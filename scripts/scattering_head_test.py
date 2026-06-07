"""Bridge run 6 — the scattering (rational) reward head vs the linear champion.

Motivated by the math project's HP-candidate-11 note (the Eisenstein scattering
matrix phi(s) = xi(2s-1)/xi(2s): a RATIO whose pole structure carries the
spectrum). Architecture question, stated falsifiably: does a rational head
R = N/D (models/spectral.py RationalSpectralReward, SK closed-form iterations)
beat the validated linear ladder+poly head where the target has RESONANCE
structure — sharp localized rewards, the RL analog of goal/contact bonuses?

Arms (matched total feature count M=512, matched sweep budget, same noisy-
train / noisy-val / clean-test protocol as the recipe test):
  linear   — sigma-ladder RFF + lambda-polynomial weights (run-3/5 champion form)
  rational — N/D with M/2 features each, SAME ladder + poly weights per block

Target families per cell:
  smooth    — rich_reward (the recipe-test target; parity expected)
  resonant  — rich_reward + 3 bounded spikes A*eps/(eps + |x - x_k|^2) with
              centers SAMPLED FROM THE DATA MANIFOLD (on-policy resonances)

PRE-REGISTERED CRITERIA (recorded in the ledger before results):
  (i)  rational beats linear on resonant targets (overall AND near-spike test
       MSE) in a majority of cells;
  (ii) rational is within noise of linear on smooth targets (no tax);
  (iii) resonance recovery: the model's 1/|D| peaks land on the true spike
        centers — contrast = mean resonance score at centers / at random
        test points > 3.
  FALSIFIER: (i) fails => the scattering form is decorative for reward
  modeling and run 6 is recorded as NOT SUPPORTED, whatever (iii) shows.

    python scripts/scattering_head_test.py --budget 30    # chunked, resumable
    python scripts/scattering_head_test.py --report
"""
from __future__ import annotations

import argparse
import json
import subprocess as _sp
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from transversality_test import collect_competent, rich_reward, REF_MU, REF_SD
from mbrl.models.spectral import (SpectralReward, RationalSpectralReward,
                                  poly_weights)


def _git_sha() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).resolve().parents[1],
                                text=True).strip()
    except Exception:
        return "nogit"


RESULTS_DIR = Path("results/bridge") / _git_sha()
CELLS_PATH = RESULTS_DIR / "scattering_cells.jsonl"
OUT_PATH = RESULTS_DIR / "scattering_test.json"

NS = [512, 2048]
SEEDS = [0, 1, 2, 3, 4]
TARGETS = ("smooth", "resonant")
EXTREME_TARGETS = ("goal", "sharp")   # run 8: where linear features ring
NOISE_SIGMA = 1.0
LADDER = [0.25, 0.5, 1.0, 2.0]
LAMS = [1e-4, 1e-2, 1.0, 100.0]
SHAPES = [
    {"name": "quartic",        "degrees": [2],    "coefs": [1.0]},
    {"name": "quartic+sextic", "degrees": [2, 3], "coefs": [1.0, 1.0]},
    {"name": "high-clamp",     "degrees": [1, 3], "coefs": [0.1, 10.0]},
]
N_FEATURES = 512
SPIKE_A, SPIKE_EPS, N_SPIKES = 3.0, 0.05, 3


def make_cell(n: int, seed: int, target: str):
    X, A, _ = collect_competent(n + 4096, seed)
    XA = ((np.concatenate([X, A], axis=1) - REF_MU) / REF_SD).astype(np.float32)
    XA_t = torch.from_numpy(XA)
    r = rich_reward(XA_t)
    centers = None
    if target == "resonant":
        idx = np.random.default_rng(seed + 50).choice(len(XA), N_SPIKES,
                                                      replace=False)
        centers = XA_t[idx]                      # resonances ON the manifold
        d2 = torch.cdist(XA_t, centers).pow(2)   # (N, K)
        r = r + (SPIKE_A * SPIKE_EPS / (SPIKE_EPS + d2)).sum(-1)
    elif target == "goal":                       # run 8: discontinuous bonus
        idx = np.random.default_rng(seed + 50).choice(len(XA), N_SPIKES,
                                                      replace=False)
        centers = XA_t[idx]
        d2 = torch.cdist(XA_t, centers).pow(2)
        r = r + 5.0 * (d2 < 0.25).any(-1).float()    # indicator on a ball
    elif target == "sharp":                      # run 8: near-pole spikes
        idx = np.random.default_rng(seed + 50).choice(len(XA), N_SPIKES,
                                                      replace=False)
        centers = XA_t[idx]
        d2 = torch.cdist(XA_t, centers).pow(2)
        r = r + (30.0 * 0.002 / (0.002 + d2)).sum(-1)
    g = torch.Generator().manual_seed(seed + 9000)
    noise = NOISE_SIGMA * torch.randn(len(r), generator=g)
    perm = np.random.default_rng(seed).permutation(len(XA))
    tr, va, te = perm[:n], perm[n:n + 2048], perm[n + 2048:]
    pack = lambda idx: (XA_t[idx], (r + noise)[idx])
    xa_te = XA_t[te]
    return (*pack(tr), *pack(va), xa_te, r[te], centers)


def run_cell(n: int, seed: int, target: str) -> dict:
    xa_tr, y_tr, xa_va, y_va, xa_te, y_te, centers = make_cell(n, seed, target)
    d = xa_tr.shape[1]
    row = {"n": n, "seed": seed, "target": target}

    # near-spike test subset (where resonance modeling matters)
    near = None
    if centers is not None:
        near = (torch.cdist(xa_te, centers).min(-1).values < 0.5)

    def eval_model(m) -> dict:
        with torch.no_grad():
            pred = m.predict(xa_te)
            out = {"test_mse": F.mse_loss(pred, y_te).item()}
            if near is not None and near.any():
                out["near_spike_mse"] = F.mse_loss(pred[near], y_te[near]).item()
            return out

    # ---- linear champion: ladder + poly, best-on-validation ----
    t0 = time.perf_counter()
    best = None
    for sh in SHAPES:
        for lam in LAMS:
            sr = SpectralReward(d, N_FEATURES, LADDER, seed=seed)
            w = poly_weights(sr.w2.sqrt(), sh["degrees"],
                             [lam * c for c in sh["coefs"]])
            sr.fit(xa_tr, y_tr, weights=w)
            val = F.mse_loss(sr.predict(xa_va), y_va).item()
            if best is None or val < best[0]:
                best = (val, f"{sh['name']}@lam={lam:g}", sr)
    row["linear"] = {"choice": best[1], **eval_model(best[2]),
                     "wall_s": time.perf_counter() - t0}

    # ---- rational (scattering) head: same ladder/poly per block ----
    t0 = time.perf_counter()
    best = None
    # den_anchor is a regularizer — swept on validation like lam (v3; the
    # rational arm's sweep is 2x the linear arm's, recorded in the ledger)
    for sh in SHAPES:
        for lam in LAMS:
            for anchor in (0.03, 0.3):
                rr = RationalSpectralReward(d, N_FEATURES, LADDER, seed=seed)
                wn = poly_weights(rr.num.w2.sqrt(), sh["degrees"],
                                  [lam * c for c in sh["coefs"]])
                wd = poly_weights(rr.den.w2.sqrt(), sh["degrees"],
                                  [lam * c for c in sh["coefs"]])
                rr.fit(xa_tr, y_tr, weights_num=wn, weights_den=wd,
                       den_anchor=anchor)
                val = F.mse_loss(rr.predict(xa_va), y_va).item()
                if best is None or val < best[0]:
                    best = (val, f"{sh['name']}@lam={lam:g}@a={anchor:g}", rr)
    rr = best[2]
    ent = {"choice": best[1], **eval_model(rr),
           "clamp_rate": rr.clamp_rate_, "wall_s": time.perf_counter() - t0}
    if centers is not None:   # criterion (iii): resonance recovery
        rand = xa_te[torch.randperm(len(xa_te),
                                    generator=torch.Generator().manual_seed(1))[:512]]
        ent["resonance_contrast"] = float(
            rr.resonance_score(centers).mean() / rr.resonance_score(rand).mean())
    row["rational"] = ent
    return row


def load_done() -> dict:
    done = {}
    if CELLS_PATH.exists():
        for line in CELLS_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["n"], r["seed"], r["target"])] = r
    return done


def report(done: dict):
    seen = sorted({r['target'] for r in done.values()})
    for target in seen:
        rows = [r for r in done.values() if r["target"] == target]
        if not rows:
            continue
        lin = np.mean([r["linear"]["test_mse"] for r in rows])
        rat = np.mean([r["rational"]["test_mse"] for r in rows])
        wins = sum(1 for r in rows
                   if r["rational"]["test_mse"] < r["linear"]["test_mse"])
        print(f"\n[{target}] linear {lin:.4f} vs rational {rat:.4f} — "
              f"rational wins {wins}/{len(rows)}")
        if target in ("resonant", "goal", "sharp"):
            nlin = np.mean([r["linear"].get("near_spike_mse", np.nan) for r in rows])
            nrat = np.mean([r["rational"].get("near_spike_mse", np.nan) for r in rows])
            nwins = sum(1 for r in rows if r["rational"].get("near_spike_mse", 9e9)
                        < r["linear"].get("near_spike_mse", 9e9))
            contrast = np.mean([r["rational"].get("resonance_contrast", np.nan)
                                for r in rows])
            clamp = np.mean([r["rational"]["clamp_rate"] for r in rows])
            print(f"  near-spike: linear {nlin:.4f} vs rational {nrat:.4f} "
                  f"({nwins}/{len(rows)})")
            print(f"  resonance recovery contrast = {contrast:.1f} "
                  f"(criterion: > 3); D-clamp rate = {clamp:.3%}")
    # pre-registered verdict
    res = [r for r in done.values() if r["target"] in ("resonant", "goal", "sharp")]
    smo = [r for r in done.values() if r["target"] == "smooth"]
    if res and smo and len(res) + len(smo) == len(NS) * len(SEEDS) * 2:
        i_ok = (sum(1 for r in res if r["rational"]["test_mse"]
                    < r["linear"]["test_mse"]) > len(res) / 2
                and sum(1 for r in res if r["rational"].get("near_spike_mse", 9e9)
                        < r["linear"].get("near_spike_mse", 0)) > len(res) / 2)
        print(f"\ncriterion (i) resonant majority win: {'YES' if i_ok else 'NO'}"
              f" — run 6 {'SUPPORTED' if i_ok else 'NOT SUPPORTED'} "
              "(per pre-registration; see ledger)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=30.0)
    p.add_argument("--report", action="store_true")
    p.add_argument("--extreme", action="store_true",
                   help="run 8: goal/sharp targets (new pre-registration)")
    args = p.parse_args()
    global CELLS_PATH, OUT_PATH
    targets = TARGETS
    if args.extreme:
        targets = EXTREME_TARGETS
        CELLS_PATH = RESULTS_DIR / "scattering_extreme_cells.jsonl"
        OUT_PATH = RESULTS_DIR / "scattering_extreme_test.json"
    CELLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    cells = [(n, s, t) for t in targets for n in NS for s in SEEDS]
    if not args.report:
        t0 = time.perf_counter()
        for n, s, t in cells:
            if (n, s, t) in done or time.perf_counter() - t0 > args.budget:
                continue
            row = run_cell(n, s, t)
            with CELLS_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[(n, s, t)] = row
            print(f"done: {t} n={n} s={s} lin="
                  f"{row['linear']['test_mse']:.4f} rat="
                  f"{row['rational']['test_mse']:.4f}")
    report(done)
    remaining = [c for c in cells if c not in done]
    if remaining:
        print(f"\n{len(remaining)} cells remaining — re-run to continue")
    else:
        OUT_PATH.write_text(json.dumps(list(done.values()), indent=1))
        print(f"\nall cells done — wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
