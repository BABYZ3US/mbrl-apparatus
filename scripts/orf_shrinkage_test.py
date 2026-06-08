"""Bridge run 12 — research-cycle candidates from approximation theory.

Cycle 1 of the continuous research loop. Sources: ORF/GORF variance reduction
(Yu et al. 2016; surveys 2004.11154, 1911.09158 — leverage-score sampling
noted as cycle-2 candidate), Donoho–Johnstone wavelet shrinkage (universal
threshold, near-minimax over Besov classes), and the math project's
incremental-filtration structure (framework_feedback F3: measure each
component's signal against ITS OWN noise, not a global dose).

Arms, all on the calibrated champion form (cal_low ladder + high-clamp poly,
matched sweep budget, smooth + resonant targets from the run-6 protocol):
  champion        — iid RFF (control, current best)
  orf             — orthogonalized frequency frame (candidate A)
  shrink          — champion + per-coefficient soft-threshold (candidate B)
  orf+shrink      — both

PRE-REGISTERED CRITERIA: a candidate ships into the champion config iff it
beats the champion arm in a MAJORITY of cells with mean relative benefit
> +2% (above seed noise) and never catastrophically loses a cell (> -20%).
FALSIFIER: neither clears the bar => both recorded NOT SUPPORTED; cycle 2
moves to leverage-score sampling.

    python scripts/orf_shrinkage_test.py --budget 30
"""
from __future__ import annotations

import argparse
import json
import subprocess as _sp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from transversality_test import collect_competent, rich_reward, REF_MU, REF_SD
from mbrl.models.spectral import (SpectralReward, calibrate_sigma_ladder,
                                  orthogonalize_features, poly_weights)


def _git_sha() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).resolve().parents[1],
                                text=True).strip()
    except Exception:
        return "nogit"


RESULTS_DIR = Path("results/bridge") / _git_sha()
CELLS_PATH = RESULTS_DIR / "orf_shrinkage_cells.jsonl"
OUT_PATH = RESULTS_DIR / "orf_shrinkage_test.json"

NS = [512, 2048]
SEEDS = [0, 1, 2, 3, 4]
TARGETS = ("smooth", "resonant")
NOISE_SIGMA = 1.0
LAMS = [1e-4, 1e-2, 1.0, 100.0]
SHAPES = [
    {"name": "quartic",        "degrees": [2],    "coefs": [1.0]},
    {"name": "quartic+sextic", "degrees": [2, 3], "coefs": [1.0, 1.0]},
    {"name": "high-clamp",     "degrees": [1, 3], "coefs": [0.1, 10.0]},
]
N_FEATURES = 512
# candidate B (DJ shrinkage) DROPPED PRE-RUN: requires orthonormal design;
# rings in the correlated RFF basis (tests/test_spectral.py pins it).
# Cycle-2 successor: shrinkage in the Phi-SVD basis (adaptive TSVD).
ARMS = ("champion", "orf")


def make_cell(n: int, seed: int, target: str):
    X, A, _ = collect_competent(n + 4096, seed)
    XA = ((np.concatenate([X, A], axis=1) - REF_MU) / REF_SD).astype(np.float32)
    XA_t = torch.from_numpy(XA)
    r = rich_reward(XA_t)
    if target == "resonant":
        idx = np.random.default_rng(seed + 50).choice(len(XA), 3, replace=False)
        d2 = torch.cdist(XA_t, XA_t[idx]).pow(2)
        r = r + (3.0 * 0.05 / (0.05 + d2)).sum(-1)
    g = torch.Generator().manual_seed(seed + 9000)
    noise = NOISE_SIGMA * torch.randn(len(r), generator=g)
    perm = np.random.default_rng(seed).permutation(len(XA))
    tr, va, te = perm[:n], perm[n:n + 2048], perm[n + 2048:]
    return (XA_t[tr], (r + noise)[tr], XA_t[va], (r + noise)[va],
            XA_t[te], r[te])


def run_cell(n: int, seed: int, target: str) -> dict:
    xa_tr, y_tr, xa_va, y_va, xa_te, y_te = make_cell(n, seed, target)
    d = xa_tr.shape[1]
    ladder, _ = calibrate_sigma_ladder(xa_tr, y_tr, seed=seed)  # champion form
    row = {"n": n, "seed": seed, "target": target}

    for arm in ARMS:
        t0 = time.perf_counter()
        best = None
        for sh in SHAPES:
            for lam in LAMS:
                sr = SpectralReward(d, N_FEATURES, ladder, seed=seed)
                if "orf" in arm:
                    sr = orthogonalize_features(sr)
                w = poly_weights(sr.w2.sqrt(), sh["degrees"],
                                 [lam * c for c in sh["coefs"]])
                sr.fit(xa_tr, y_tr, weights=w)
                val = F.mse_loss(sr.predict(xa_va), y_va).item()
                if best is None or val < best[0]:
                    best = (val, f"{sh['name']}@lam={lam:g}", sr)
        _, choice, sr = best
        with torch.no_grad():
            mse = F.mse_loss(sr.predict(xa_te), y_te).item()
        row[arm] = {"choice": choice, "test_mse": mse,
                    "wall_s": time.perf_counter() - t0,
                    "nonzero_coefs": int((sr.c.abs() > 1e-10).sum())}
    return row


def load_done():
    done = {}
    if CELLS_PATH.exists():
        for line in CELLS_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["n"], r["seed"], r["target"])] = r
    return done


def report(done: dict):
    rows = list(done.values())
    if not rows:
        return
    for arm in ARMS[1:]:
        wins = sum(1 for r in rows
                   if r[arm]["test_mse"] < r["champion"]["test_mse"])
        rel = np.mean([(r["champion"]["test_mse"] - r[arm]["test_mse"])
                       / r["champion"]["test_mse"] for r in rows])
        worst = min((r["champion"]["test_mse"] - r[arm]["test_mse"])
                    / r["champion"]["test_mse"] for r in rows)
        ship = wins > len(rows) / 2 and rel > 0.02 and worst > -0.20
        print(f"{arm:>11} vs champion: wins {wins}/{len(rows)}, mean "
              f"{rel:+.1%}, worst cell {worst:+.1%} -> "
              f"{'SHIP' if ship else 'NOT SUPPORTED'}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=30.0)
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    CELLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    cells = [(n, s, t) for t in TARGETS for n in NS for s in SEEDS]
    if not args.report:
        t0 = time.perf_counter()
        for n, s, t in cells:
            if (n, s, t) in done or time.perf_counter() - t0 > args.budget:
                continue
            row = run_cell(n, s, t)
            with CELLS_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[(n, s, t)] = row
            print(f"done: {t} n={n} s={s} " + " ".join(
                f"{a}={row[a]['test_mse']:.4f}" for a in ARMS))
    report(done)
    remaining = [c for c in cells if c not in done]
    if remaining:
        print(f"\n{len(remaining)} cells remaining — re-run to continue")
    else:
        OUT_PATH.write_text(json.dumps(list(done.values()), indent=1))
        print(f"\nall cells done — wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
