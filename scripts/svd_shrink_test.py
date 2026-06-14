"""Bridge run 12B — shrinkage in the Phi-SVD basis (adaptive TSVD).

Research-cycle 2, the THEORETICALLY CORRECT form of run-12 candidate B.
Candidate B (Donoho-Johnstone universal-threshold shrinkage, scripts/
orf_shrinkage_test.py) was DROPPED PRE-RUN: DJ near-minimaxity requires an
ORTHONORMAL basis; in the correlated RFF design cancellation pairs ring when
one side is zeroed (tests/test_spectral.py pins MSE x3000). The fix, named in
the ledger's cycle-2 queue: do the shrinkage in the orthonormal basis that the
problem actually provides — the economy SVD of the penalty-whitened design
(Rosasco spectral filtering / adaptive TSVD). U is orthonormal, so beta = U^T y
carries iid target noise per component, the exact DJ setting.

Sources web-searched first: Rosasco spectral-filtering family (MIT 9.520
class07; Wikipedia "Regularization by spectral filtering" — Tikhonov /
Landweber / TSVD as filters g(s) on singular values); Donoho-Johnstone 1994
universal threshold tau = sigma*sqrt(2 log n) with MAD/0.6745 noise estimate.

Arms, on the calibrated champion form (cal_low ladder + poly sweep, matched
SHAPES x LAMS budget, smooth + resonant run-6 targets):
  champion    — iid RFF closed-form ridge (control, current best)
  svd_shrink  — same fit, DJ soft-threshold in the Phi-SVD basis (kappa=1,
                parameter-free => matched sweep budget)

PRE-REGISTERED CRITERIA (ledger default bar, fixed BEFORE results exist):
svd_shrink ships into the champion config iff it beats champion in a MAJORITY
of the 20 cells (n in {512,2048} x 5 seeds x {smooth,resonant}) with mean
relative test-MSE benefit > +2% AND no cell worse than -20%.
FALSIFIER: bar not cleared => NOT SUPPORTED, recorded; the supervised
positivity/shrinkage program for candidate B is then closed (3rd orthonormal-
frame instance: runs 4, 6/8, 12), cycle-2 moves to leverage-score sampling.

    python scripts/svd_shrink_test.py --budget 30      # chunked, resumable
    python scripts/svd_shrink_test.py --report         # adjudicate only
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

from orf_shrinkage_test import (NS, SEEDS, TARGETS, LAMS, SHAPES, N_FEATURES,
                                make_cell)
from mbrl.models.spectral import (SpectralReward, calibrate_sigma_ladder,
                                  poly_weights, svd_shrink_fit)

# svd_shrink sweep = LAMS (4, shared with champion) x KAPPAS (3) = 12 configs,
# MATCHED to the champion's SHAPES (3) x LAMS (4) = 12. kappa=0 is the pure
# Tikhonov spectral filter (no shrinkage), kappa=1 the DJ universal threshold,
# kappa=2 a conservative variant; validation picks (lam, kappa).
KAPPAS = (0.0, 1.0, 2.0)


def _git_sha() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).resolve().parents[1],
                                text=True).strip()
    except Exception:
        return "nogit"


RESULTS_DIR = Path("results/bridge") / _git_sha()
CELLS_PATH = RESULTS_DIR / "svd_shrink_cells.jsonl"
OUT_PATH = RESULTS_DIR / "svd_shrink_test.json"
ARMS = ("champion", "svd_shrink")


def run_cell(n: int, seed: int, target: str) -> dict:
    xa_tr, y_tr, xa_va, y_va, xa_te, y_te = make_cell(n, seed, target)
    d = xa_tr.shape[1]
    ladder, _ = calibrate_sigma_ladder(xa_tr, y_tr, seed=seed)  # champion form
    row = {"n": n, "seed": seed, "target": target}

    for arm in ARMS:
        t0 = time.perf_counter()
        best = None
        if arm == "svd_shrink":
            configs = [("kappa", lam, kappa) for lam in LAMS for kappa in KAPPAS]
        else:
            configs = [(sh, lam, None) for sh in SHAPES for lam in LAMS]
        for cfg, lam, kappa in configs:
            sr = SpectralReward(d, N_FEATURES, ladder, seed=seed)
            if arm == "svd_shrink":
                svd_shrink_fit(sr, xa_tr, y_tr, lam=lam, kappa=kappa)
                tag = f"lam={lam:g}@kappa={kappa:g}"
            else:
                w = poly_weights(sr.w2.sqrt(), cfg["degrees"],
                                 [lam * c for c in cfg["coefs"]])
                sr.fit(xa_tr, y_tr, weights=w)
                tag = f"{cfg['name']}@lam={lam:g}"
            val = F.mse_loss(sr.predict(xa_va), y_va).item()
            if best is None or val < best[0]:
                best = (val, tag, sr)
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
