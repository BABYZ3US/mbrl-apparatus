"""Bridge run 13 — leverage-score feature sampling (research cycle 2).

The sole remaining cycle-2 queue item after candidate A (ORF, run 12 — sub-
threshold) and candidate B (DJ shrinkage in the Phi-SVD basis, run 12B — NOT
SUPPORTED). Idea (Bach 2017, "On the Equivalence between Kernel Quadrature
Rules and Random Feature Expansions"): the number of random features needed to
match full-kernel performance is governed by the kernel's LEVERAGE function;
sampling features from the leverage-tilted distribution (vs iid from the base
spectral measure) needs provably fewer of them. Practical empirical-leverage
schemes: Rudi & Rosasco 2017 (generalization of RFF learning); Rudi-Camoriano-
Rosasco 2018 (fast leverage-score sampling). Web-searched before implementing.

ONE CHANGE vs champion: the feature SET only. Both arms share the calibrated
cal_low ladder, the poly-band penalty, the closed-form ridge, and the SAME
SHAPES x LAMS = 12 validation sweep. The leverage arm draws a POOL_MULT x larger
pool from the ladder measure, computes the empirical ridge leverage score of
each pooled feature on the (label-free) training design, and importance-samples
N_FEATURES of them WITHOUT replacement proportional to leverage. Nothing else
differs — so any test-MSE gap is attributable to leverage-tilted vs iid feature
placement at the champion's budget.

Arms (calibrated champion form, run-6 smooth + resonant targets, 20 cells):
  champion   — iid RFF from the cal_low ladder (control, current best)
  leverage   — leverage-sampled features from a POOL_MULT x pool, same recipe

PRE-REGISTERED hyperparameters (fixed BEFORE results, chosen label-free):
  POOL_MULT = 4   (pool = 2048; more candidates => more for leverage to choose)
  LAM_LEV   = 1.0 (central, clearly selective on a label-free dry run:
                   d_eff 5.5-116, leverage CV 1.0-1.8 across the 4 regimes;
                   the reward's effective dimension is small, d_eff << M, in
                   every cell — recorded so a null is interpretable: at matched
                   M=512 the low-rank signal may already be over-covered by iid
                   features, and leverage's theoretical edge is at M ~ d_eff,
                   a reduced-budget question this matched test does not ask).

PRE-REGISTERED CRITERIA (ledger default bar, fixed before any cell ran):
leverage ships into the champion config iff it beats champion in a MAJORITY of
the 20 cells with mean relative test-MSE > +2% AND worst cell > -20%.
FALSIFIER: bar not cleared => NOT SUPPORTED; recorded; cycle-2 supervised queue
is then exhausted (candidates A, B, and leverage all closed) and the loop moves
to a fresh literature pass / the RL-loop questions.

    python scripts/leverage_sample_test.py --budget 30      # chunked, resumable
    python scripts/leverage_sample_test.py --report         # adjudicate only
"""
from __future__ import annotations

import argparse
import copy
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
                                  leverage_sample, poly_weights)

POOL_MULT = 4
LAM_LEV = 1.0


def _git_sha() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).resolve().parents[1],
                                text=True).strip()
    except Exception:
        return "nogit"


RESULTS_DIR = Path("results/bridge") / _git_sha()
CELLS_PATH = RESULTS_DIR / "leverage_sample_cells.jsonl"
OUT_PATH = RESULTS_DIR / "leverage_sample_test.json"
ARMS = ("champion", "leverage")


def run_cell(n: int, seed: int, target: str) -> dict:
    xa_tr, y_tr, xa_va, y_va, xa_te, y_te = make_cell(n, seed, target)
    d = xa_tr.shape[1]
    ladder, _ = calibrate_sigma_ladder(xa_tr, y_tr, seed=seed)  # champion form
    row = {"n": n, "seed": seed, "target": target}

    # leverage arm: select the feature basis ONCE per cell (label-free), then
    # run the identical poly-band sweep on it (copy.copy per config = fresh c,
    # shared W/b/w2/w4 — read-only in fit/predict).
    base = SpectralReward(d, N_FEATURES * POOL_MULT, ladder, seed=seed)
    base = leverage_sample(base, xa_tr, N_FEATURES, LAM_LEV,
                           generator=torch.Generator().manual_seed(seed + 1234))

    for arm in ARMS:
        t0 = time.perf_counter()
        best = None
        for sh in SHAPES:
            for lam in LAMS:
                if arm == "leverage":
                    sr = copy.copy(base)
                else:
                    sr = SpectralReward(d, N_FEATURES, ladder, seed=seed)
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
    row["leverage"]["lev_info"] = base.lev_info
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
    # per-target + leverage diagnostics (for honesty, not adjudication)
    for tgt in TARGETS:
        sub = [r for r in rows if r["target"] == tgt]
        if not sub:
            continue
        rel = np.mean([(r["champion"]["test_mse"] - r["leverage"]["test_mse"])
                       / r["champion"]["test_mse"] for r in sub])
        wins = sum(1 for r in sub
                   if r["leverage"]["test_mse"] < r["champion"]["test_mse"])
        deff = np.mean([r["leverage"]["lev_info"]["d_eff"] for r in sub])
        print(f"  [{tgt:>8}] leverage wins {wins}/{len(sub)}, mean {rel:+.1%}, "
              f"mean d_eff {deff:.1f}")


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
