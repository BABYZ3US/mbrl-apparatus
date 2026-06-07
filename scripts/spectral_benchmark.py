"""Spectral reward solver benchmark — closed-form RFF ridge vs the current way.

Same data protocol as scripts/transversality_test.py (competent-policy Pendulum
rollouts, rich_reward in fixed REF_MU/REF_SD standardized coords, shuffled
split). Per (n, seed) cell, three arms:
  (a) mlp_hutchinson — MLP 2x256 + 2-probe Hutchinson H^2 penalty, 1500 epochs
      Adam (the current way, lam=1e-2 as in transversality_test).
  (b) spectral — SpectralReward closed-form fit, lam swept over
      {1e-6, 1e-4, 1e-2, 1}, best-on-validation; also reports the EXACT penalty
      value (1/M) sum c_j^2 |w_j|^4 at the chosen lam.
  (c) spectral_poly — same solver with POLYNOMIAL per-band ridge weights
      (models/spectral.py poly_weights): weights_j = sum_d lam*coefs[d]*|w_j|^(2*degrees[d]).
      Sweep = SPECTRAL_LAMS x POLY_CONFIGS (shapes below), best-on-validation.
      This is the supervised analog of the Trainer's lambda-polynomial band
      equalizer (static coefs here — no schedule time axis in a single fit).

Wall-clock per cell is FIT time only (the spectral arms' clocks cover their
whole sweeps — the honest cost of best-on-validation); data collection excluded
from all arms.

Resumable: each finished cell is appended to results/spectral_benchmark_cells.jsonl
and skipped on re-run, so the grid can be filled in chunks under a wall-clock
budget (--budget). When all cells are done the merged table is printed and saved
to results/spectral_benchmark.json.

    python scripts/spectral_benchmark.py --budget 30      # run a chunk
    python scripts/spectral_benchmark.py --report         # table only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from transversality_test import collect_competent, rich_reward, REF_MU, REF_SD

NS = [128, 512, 2048]
SEEDS = [0, 1, 2]
SPECTRAL_LAMS = [1e-6, 1e-4, 1e-2, 1.0]
# (c) lambda-polynomial shapes: P(|w|^2) = sum_d coefs[d] * |w|^(2*degrees[d]),
# overall scale swept by lam. quartic-only ([2],[1.0]) is arm (b) — not repeated.
POLY_CONFIGS = [
    {"name": "quad+quartic",   "degrees": [1, 2],    "coefs": [1.0, 1.0]},
    {"name": "quartic+sextic", "degrees": [2, 3],    "coefs": [1.0, 1.0]},
    {"name": "full-123",       "degrees": [1, 2, 3], "coefs": [1.0, 1.0, 1.0]},
    {"name": "high-clamp",     "degrees": [1, 3],    "coefs": [0.1, 10.0]},
]
MLP_LAM = 1e-2          # the transversality_test default dose
MLP_EPOCHS = 1500
CELLS_PATH = Path("results/spectral_benchmark_cells.jsonl")
OUT_PATH = Path("results/spectral_benchmark.json")


def make_data(n: int, seed: int):
    """train (n) / val (2048) / test (rest) in fixed standardized coords."""
    X, A, _ = collect_competent(n + 4096, seed)
    XA = np.concatenate([X, A], axis=1)
    XA = ((XA - REF_MU) / REF_SD).astype(np.float32)
    r = rich_reward(torch.from_numpy(XA)).numpy().astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(XA))
    tr, va, te = perm[:n], perm[n:n + 2048], perm[n + 2048:]
    t = torch.from_numpy
    return (t(XA[tr]), t(r[tr]), t(XA[va]), t(r[va]), t(XA[te]), t(r[te]))


def run_mlp(n: int, seed: int, time_left: float = 1e9) -> dict | None:
    """(a) the current way: MLP 2x256 + 2-probe Hutchinson penalty, Adam.

    Chunk-resumable: large-n cells exceed a single wall-clock budget, so if
    `time_left` runs out mid-training the model/optimizer/probe-RNG state is
    stashed and None is returned; the next invocation resumes the epochs.
    Reported wall_s accumulates across chunks (fit time only)."""
    from mbrl.models.encoder import mlp
    from mbrl.regularization.hutchinson import hvp_penalty

    torch.manual_seed(seed)
    xa_tr, r_tr, _, _, xa_te, r_te = make_data(n, seed)
    R = mlp([xa_tr.shape[1], 256, 256, 1])
    opt = torch.optim.Adam(R.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    fn = lambda x: R(x).squeeze(-1)

    stash = CELLS_PATH.parent / f"spectral_benchmark_mlp_n{n}_s{seed}.pt"
    epochs_done, wall_prev = 0, 0.0
    if stash.exists():
        st = torch.load(stash, weights_only=False)
        R.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        epochs_done, wall_prev = st["epochs"], st["wall"]

    t0 = time.perf_counter()
    for e in range(epochs_done, MLP_EPOCHS):
        loss = F.mse_loss(R(xa_tr).squeeze(-1), r_tr) \
             + MLP_LAM * hvp_penalty(fn, xa_tr, 2, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if time.perf_counter() - t0 > time_left:
            torch.save({"model": R.state_dict(), "opt": opt.state_dict(),
                        "gen": gen.get_state(), "epochs": e + 1,
                        "wall": wall_prev + (time.perf_counter() - t0)}, stash)
            print(f"stash: mlp_hutchinson n={n} seed={seed} at epoch {e + 1}"
                  f"/{MLP_EPOCHS} — re-run to continue")
            return None
    wall = wall_prev + (time.perf_counter() - t0)
    stash.unlink(missing_ok=True)
    with torch.no_grad():
        mse = F.mse_loss(R(xa_te).squeeze(-1), r_te).item()
    return {"arm": "mlp_hutchinson", "n": n, "seed": seed, "lam": MLP_LAM,
            "test_mse": mse, "wall_s": wall}


def run_spectral(n: int, seed: int) -> dict:
    """(b) SpectralReward closed-form ridge; lam chosen on validation MSE."""
    from mbrl.models.spectral import SpectralReward

    xa_tr, r_tr, xa_va, r_va, xa_te, r_te = make_data(n, seed)
    d = xa_tr.shape[1]
    best = None  # (val_mse, lam, model)
    t0 = time.perf_counter()
    for lam in SPECTRAL_LAMS:
        sr = SpectralReward(d, n_features=512, sigma_w=1.0, seed=seed).fit(
            xa_tr, r_tr, lam)
        with torch.no_grad():
            val = F.mse_loss(sr.predict(xa_va), r_va).item()
        if best is None or val < best[0]:
            best = (val, lam, sr)
    wall = time.perf_counter() - t0  # whole sweep: the honest cost
    val_mse, lam, sr = best
    with torch.no_grad():
        mse = F.mse_loss(sr.predict(xa_te), r_te).item()
    return {"arm": "spectral", "n": n, "seed": seed, "lam": lam,
            "val_mse": val_mse, "test_mse": mse, "wall_s": wall,
            "penalty_exact": sr.hessian_frobenius_sq()}


def run_spectral_poly(n: int, seed: int) -> dict:
    """(c) SpectralReward closed-form ridge with lambda-polynomial per-band
    weights; (lam, shape) chosen on validation MSE."""
    from mbrl.models.spectral import SpectralReward, poly_weights

    xa_tr, r_tr, xa_va, r_va, xa_te, r_te = make_data(n, seed)
    d = xa_tr.shape[1]
    best = None  # (val_mse, lam, name, model)
    t0 = time.perf_counter()
    for lam in SPECTRAL_LAMS:
        for cfg in POLY_CONFIGS:
            sr = SpectralReward(d, n_features=512, sigma_w=1.0, seed=seed)
            w = poly_weights(sr.w2.sqrt(), cfg["degrees"],
                             [lam * c for c in cfg["coefs"]])
            sr.fit(xa_tr, r_tr, weights=w)
            with torch.no_grad():
                val = F.mse_loss(sr.predict(xa_va), r_va).item()
            if best is None or val < best[0]:
                best = (val, lam, cfg["name"], sr)
    wall = time.perf_counter() - t0  # whole (lam x shape) sweep: the honest cost
    val_mse, lam, name, sr = best
    with torch.no_grad():
        mse = F.mse_loss(sr.predict(xa_te), r_te).item()
    return {"arm": "spectral_poly", "n": n, "seed": seed, "lam": lam,
            "poly": name, "val_mse": val_mse, "test_mse": mse, "wall_s": wall,
            "penalty_exact": sr.hessian_frobenius_sq()}


ARMS = ("mlp_hutchinson", "spectral", "spectral_poly")


def all_cells():
    return [(arm, n, s) for n in NS for s in SEEDS for arm in ARMS]


def load_done() -> dict:
    done = {}
    if CELLS_PATH.exists():
        for line in CELLS_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["arm"], row["n"], row["seed"])] = row
    return done


def report(done: dict):
    print(f"\n{'n':>6} {'arm':>16} {'test_mse':>12} {'wall_s':>8} "
          f"{'lam*':>14} {'penalty*':>10}  poly*")
    for n in NS:
        for arm in ARMS:
            rows = [done[(arm, n, s)] for s in SEEDS if (arm, n, s) in done]
            if not rows:
                continue
            mse = np.mean([r["test_mse"] for r in rows])
            wall = np.mean([r["wall_s"] for r in rows])
            if arm == "mlp_hutchinson":
                print(f"{n:>6} {arm:>16} {mse:>12.4f} {wall:>8.2f} "
                      f"{'-':>14} {'-':>10}")
                continue
            lams = ",".join(f"{r['lam']:g}" for r in rows)
            pen = np.mean([r["penalty_exact"] for r in rows])
            polys = ",".join(r.get("poly", "-") for r in rows) \
                if arm == "spectral_poly" else ""
            print(f"{n:>6} {arm:>16} {mse:>12.4f} {wall:>8.2f} "
                  f"{lams:>14} {pen:>10.4f}  {polys}")
    print("  lam* = per-seed best-on-validation lam; poly* = per-seed best "
          "shape; penalty* = exact (1/M) sum c^2 |w|^4 at the chosen weights")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=30.0,
                   help="wall-clock budget (s); stop starting new cells beyond it")
    p.add_argument("--report", action="store_true", help="print table only")
    args = p.parse_args()

    CELLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    cells = all_cells()

    if not args.report:
        t_start = time.perf_counter()
        for arm, n, s in cells:
            if (arm, n, s) in done:
                continue
            elapsed = time.perf_counter() - t_start
            if elapsed > args.budget:
                break
            if arm == "mlp_hutchinson":
                row = run_mlp(n, s, time_left=args.budget - elapsed)
                if row is None:  # chunk budget hit mid-cell; stashed
                    break
            elif arm == "spectral_poly":
                row = run_spectral_poly(n, s)
            else:
                row = run_spectral(n, s)
            with CELLS_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[(arm, n, s)] = row
            print(f"done: {arm} n={n} seed={s} mse={row['test_mse']:.4f} "
                  f"wall={row['wall_s']:.2f}s")

    remaining = [c for c in cells if c not in done]
    report(done)
    if remaining:
        print(f"\n{len(remaining)} cells remaining — re-run to continue: "
              + ", ".join(f"{a}/n{n}/s{s}" for a, n, s in remaining))
    else:
        OUT_PATH.write_text(json.dumps(list(done.values()), indent=1))
        print(f"\nall {len(cells)} cells done — wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
