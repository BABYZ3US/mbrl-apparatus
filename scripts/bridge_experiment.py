"""Bridge experiment (claims_ledger.md, Weil-positivity entry — conjecture tier).

Tests the ledger's closed-form prediction: if the clamp's operative property is
positivity of the curvature quadratic functional, then a positivity-constrained
spectral solve should reproduce the clamped-trace penalty's empirical ordering

    Lap-2 clamped (+41)  >  Frobenius (-40)  >  Lap-2 unclamped (-79)

(original_findings_report.md sec.3, HalfCheetah returns) in a closed-form
supervised setting — same ordering, no Hutchinson during the fit, no clamp
heuristic. Lower test MSE here plays the role of higher return there.

The three arms, all single linear solves in the RFF basis of models/spectral.py
(phi_j(x) = sqrt(2/M) cos(w_j.x + b_j), Delta phi_j(x) = -|w_j|^2 phi_j(x)):

  (a) frobenius_diag — the expectation penalty lam * diag(|w|^4): isotropic,
      positive, but blind to WHERE curvature sits on the data. The Frobenius
      analog (always-non-negative estimator, no per-sample coherence).
  (b) lap2_positive — the EXACT per-sample squared-Laplacian Gram penalty:
      Delta R(x_n) = d_n . c with d_n = -(phi(x_n) * |w|^2), so
      mean_n (Delta R(x_n))^2 = c' G c,  G = D'D / N  (PSD by construction).
      The curvature DENSITY is non-negative pointwise — positivity as the
      constraint class, which is what the clamp was restoring stochastically.
      The clamped-trace analog. Penalty matrix lam * M * G.
  (c) lap2_indefinite — the probe-pair product estimator as a closed-form
      quadratic: per sample, v'H(x)v = q_v(x) . c with
      q_{v,j}(x) = -phi_j(x) (w_j . v)^2, and the sign-indefinite penalty
      B = sym(Q1'Q2) / N with independent Rademacher v1_n, v2_n per sample.
      Unbiased for the same expectation but NOT positive — the unclamped
      analog. Penalty matrix lam * M * B (possibly indefinite; that is the
      point).

Scale matching: E_phases[M * G] = E_probes,phases[M * B] = diag(|w|^4), so all
three arms coincide in expectation and differ ONLY in per-sample positivity /
coherence structure. Any ordering observed is attributable to that structure.

Protocol: competent-policy Pendulum data (transversality_test protocol), TRAIN
and VALIDATION targets corrupted with N(0, NOISE_SIGMA^2) (model selection has
no clean access), TEST targets clean — the denoising setting of report sec.4.4,
where regularization structure has work to do. n up to 8192 (the closed-form
solver makes large-n cheap — no Adam epochs).

Success criterion (per ledger): test-MSE ordering
    lap2_positive < frobenius_diag < lap2_indefinite
in the majority of (n, seed) cells, and in the n-aggregated means. A null
result (no separation, or positive arm loses) counts AGAINST the bridge.
RUN 1 RESULT: NOT SUPPORTED, 0/9 (see ledger). Diagnosis: data-null
directions of the Gram form go unpenalized.

  (d) hybrid_diag_gram — wide cut + sharp transverse cut: the convex
      combination alpha * diag(|w|^4) + (1-alpha) * M * G (both terms already
      share the expectation scale, so alpha trades coverage for data
      specificity). The diag term covers every band (kills the null-direction
      confound of run 1); the Gram term cuts sharply along the data's own
      curvature directions. Per cell we also record the TRANSVERSALITY ANGLE
      between the two constraint forms (Frobenius-space angle, the
      regularization/transversality.py notion applied to explicit matrices) —
      the multi-kernel theory predicts the hybrid's benefit over diag-only
      should grow with this misalignment; report() prints the Spearman
      correlation across cells.

Resumable like spectral_benchmark.py:
    python scripts/bridge_experiment.py --budget 30   # run a chunk
    python scripts/bridge_experiment.py --report      # table only
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

NS = [512, 2048, 8192]
SEEDS = [0, 1, 2]
LAMS = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4]  # wide: Gram arms need heavier doses
                                          # (eigen-spread differs from diag)
NOISE_SIGMA = 1.0
N_FEATURES = 512
ALPHAS = [0.05, 0.2, 0.5, 0.8, 0.95]   # hybrid: weight on the wide diag cut
ARMS = ("frobenius_diag", "lap2_positive", "lap2_indefinite",
        "hybrid_diag_gram")
CELLS_PATH = Path("results/bridge_experiment_cells.jsonl")
OUT_PATH = Path("results/bridge_experiment.json")


def make_data(n: int, seed: int):
    """train (n) / val (2048) / test (rest); train+val targets noisy, test clean."""
    X, A, _ = collect_competent(n + 4096, seed)
    XA = np.concatenate([X, A], axis=1)
    XA = ((XA - REF_MU) / REF_SD).astype(np.float32)
    r = rich_reward(torch.from_numpy(XA))
    g = torch.Generator().manual_seed(seed + 9000)
    noise = NOISE_SIGMA * torch.randn(len(r), generator=g)
    perm = np.random.default_rng(seed).permutation(len(XA))
    tr, va, te = perm[:n], perm[n:n + 2048], perm[n + 2048:]
    t = lambda idx: torch.from_numpy(XA[idx])
    return (t(tr), (r + noise)[tr], t(va), (r + noise)[va], t(te), r[te])


def penalty_matrix(arm: str, sr, Phi: torch.Tensor, seed: int) -> torch.Tensor:
    """(M, M) penalty quadratic form P with E[P] = diag(|w|^4) for every arm."""
    M = sr.M
    if arm == "frobenius_diag":
        return torch.diag(sr.w4)
    if arm == "lap2_positive":
        D = -(Phi * sr.w2)                       # (N, M): Delta R(x_n) = D_n . c
        return M * (D.T @ D) / Phi.shape[0]      # PSD Gram of the exact density
    if arm == "lap2_indefinite":
        g = torch.Generator().manual_seed(seed + 7000)
        N, d = Phi.shape[0], sr.in_dim
        v1 = torch.randint(0, 2, (N, d), generator=g, dtype=torch.float32) * 2 - 1
        v2 = torch.randint(0, 2, (N, d), generator=g, dtype=torch.float32) * 2 - 1
        S1 = (v1 @ sr.W.T).pow(2)                # (N, M): (w_j . v1_n)^2
        S2 = (v2 @ sr.W.T).pow(2)
        Q1, Q2 = -(Phi * S1), -(Phi * S2)        # v'H(x_n)v = Q_n . c
        B = (Q1.T @ Q2 + Q2.T @ Q1) / (2.0 * N)  # sign-indefinite sym product
        return M * B
    raise ValueError(arm)


def run_cell(n: int, seed: int) -> dict:
    """All arms on shared data/features; (lam[, alpha]) best-on-(noisy)-val."""
    import math

    from mbrl.models.spectral import SpectralReward

    xa_tr, r_tr, xa_va, r_va, xa_te, r_te = make_data(n, seed)
    d = xa_tr.shape[1]
    sr = SpectralReward(d, n_features=N_FEATURES, sigma_w=1.0, seed=seed)
    Phi_tr, Phi_va, Phi_te = sr.features(xa_tr), sr.features(xa_va), sr.features(xa_te)
    A0 = Phi_tr.T @ Phi_tr
    rhs = Phi_tr.T @ r_tr
    eye = torch.eye(sr.M)

    P_diag = penalty_matrix("frobenius_diag", sr, Phi_tr, seed)
    P_gram = penalty_matrix("lap2_positive", sr, Phi_tr, seed)

    def solve_eval(P, lam):
        """-> (val_mse, c) or None on a degenerate solve."""
        try:
            c = torch.linalg.solve(A0 + lam * P + 1e-8 * eye, rhs)
        except RuntimeError:
            return None
        if not torch.isfinite(c).all():
            return None
        return F.mse_loss(Phi_va @ c, r_va).item(), c

    row = {"n": n, "seed": seed, "noise_sigma": NOISE_SIGMA}
    # transversality angle between the wide (diag) and sharp (Gram) cuts:
    # Frobenius-space angle between the two penalty quadratic forms
    cosang = float((P_diag * P_gram).sum()
                   / (P_diag.norm() * P_gram.norm() + 1e-30))
    row["angle_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))

    for arm in ARMS:
        t0 = time.perf_counter()
        best = None  # (val_mse, lam, alpha, c)
        if arm == "hybrid_diag_gram":
            sweep = [(lam, a) for lam in LAMS for a in ALPHAS]
        else:
            P = penalty_matrix(arm, sr, Phi_tr, seed)
            sweep = [(lam, None) for lam in LAMS]
        for lam, a in sweep:
            if a is not None:
                P = a * P_diag + (1.0 - a) * P_gram
            out = solve_eval(P, lam)
            if out is None:
                continue
            val, c = out
            if best is None or val < best[0]:
                best = (val, lam, a, c)
        wall = time.perf_counter() - t0
        if best is None:
            row[arm] = {"degenerate": True, "wall_s": wall}
            continue
        val_mse, lam, a, c = best
        test_mse = F.mse_loss(Phi_te @ c, r_te).item()
        entry = {"lam": lam, "val_mse": val_mse, "test_mse": test_mse,
                 "wall_s": wall}
        if a is not None:
            entry["alpha"] = a
        if arm == "lap2_indefinite":  # diagnostic: how indefinite was B?
            entry["P_min_eig"] = float(torch.linalg.eigvalsh(
                penalty_matrix(arm, sr, Phi_tr, seed)).min())
        row[arm] = entry
    return row


def load_done() -> dict:
    done = {}
    if CELLS_PATH.exists():
        for line in CELLS_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["n"], r["seed"])] = r
    return done


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def report(done: dict):
    print(f"\n{'n':>6} {'arm':>17} {'test_mse':>12} {'lam*':>16}  alpha*")
    orderings = []
    for n in NS:
        rows = [done[(n, s)] for s in SEEDS if (n, s) in done]
        if not rows:
            continue
        means = {}
        for arm in ARMS:
            ok = [r[arm] for r in rows if arm in r and not r[arm].get("degenerate")]
            if not ok:
                print(f"{n:>6} {arm:>17} {'DEGENERATE':>12}")
                continue
            mse = np.mean([e["test_mse"] for e in ok])
            lams = ",".join(f"{e['lam']:g}" for e in ok)
            alphas = ",".join(f"{e['alpha']:g}" for e in ok if "alpha" in e)
            means[arm] = mse
            print(f"{n:>6} {arm:>17} {mse:>12.4f} {lams:>16}  {alphas}")
        if len(means) == 3:
            ok_order = (means["lap2_positive"] < means["frobenius_diag"]
                        < means["lap2_indefinite"])
            orderings.append(ok_order)
            print(f"{'':>6} ordering lap2_positive < frobenius_diag < "
                  f"lap2_indefinite: {'YES' if ok_order else 'NO'}")
    # per-cell tally
    cells = [r for r in done.values()
             if all(a in r and not r[a].get("degenerate") for a in ARMS)]
    if cells:
        hits = sum(1 for r in cells
                   if r["lap2_positive"]["test_mse"] < r["frobenius_diag"]["test_mse"]
                   < r["lap2_indefinite"]["test_mse"])
        pos_beats_fro = sum(1 for r in cells
                            if r["lap2_positive"]["test_mse"]
                            < r["frobenius_diag"]["test_mse"])
        print(f"\nfull ordering in {hits}/{len(cells)} cells; "
              f"lap2_positive beats frobenius_diag in {pos_beats_fro}/{len(cells)}")
        print("ledger criterion: majority of cells + per-n means. "
              + ("SUPPORTED" if hits > len(cells) / 2 and orderings
                 and all(orderings) else "NOT SUPPORTED"))
        # hybrid: wide + sharp transverse cuts
        hyb_wins = sum(1 for r in cells
                       if r["hybrid_diag_gram"]["test_mse"]
                       < r["frobenius_diag"]["test_mse"])
        benefit = [(r["frobenius_diag"]["test_mse"] - r["hybrid_diag_gram"]["test_mse"])
                   / r["frobenius_diag"]["test_mse"] for r in cells]
        angles = [r["angle_deg"] for r in cells if "angle_deg" in r]
        print(f"hybrid beats frobenius_diag in {hyb_wins}/{len(cells)} cells; "
              f"mean relative benefit {np.mean(benefit):+.1%}")
        if len(angles) == len(cells) and len(cells) >= 4:
            rho = spearman(angles, benefit)
            print(f"diag-vs-Gram transversality angle: "
                  f"{np.mean(angles):.1f} deg (range {min(angles):.1f}-"
                  f"{max(angles):.1f}); Spearman(angle, hybrid benefit) = "
                  f"{rho:+.2f} (n={len(cells)} cells — low power, directional only)")


def angle_sweep(n: int = 512, seeds=(0, 1, 2, 3, 4),
                sigmas=(0.5, 1.0, 2.0, 4.0)):
    """Vary the RFF bandwidth sigma_w to MOVE the diag-vs-Gram transversality
    angle, and test whether the hybrid's benefit over diag-only tracks it
    (wide cut + sharp transverse cut: benefit should grow with misalignment).
    n=512 (the regime where the hybrid's benefit lives). Data shared across
    sigmas within a seed — the angle is a property of the basis, not the data."""
    import math

    from mbrl.models.spectral import SpectralReward

    pts = []  # (angle_deg, rel_benefit, sigma, seed)
    for seed in seeds:
        xa_tr, r_tr, xa_va, r_va, xa_te, r_te = make_data(n, seed)
        d = xa_tr.shape[1]
        for sig in sigmas:
            sr = SpectralReward(d, n_features=N_FEATURES, sigma_w=sig, seed=seed)
            Phi_tr, Phi_va, Phi_te = (sr.features(xa_tr), sr.features(xa_va),
                                      sr.features(xa_te))
            A0, rhs = Phi_tr.T @ Phi_tr, Phi_tr.T @ r_tr
            eye = torch.eye(sr.M)
            P_diag = penalty_matrix("frobenius_diag", sr, Phi_tr, seed)
            P_gram = penalty_matrix("lap2_positive", sr, Phi_tr, seed)
            cosang = float((P_diag * P_gram).sum()
                           / (P_diag.norm() * P_gram.norm() + 1e-30))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))

            def best(sweep):
                b = None
                for P, lam in sweep:
                    try:
                        c = torch.linalg.solve(A0 + lam * P + 1e-8 * eye, rhs)
                    except RuntimeError:
                        continue
                    if not torch.isfinite(c).all():
                        continue
                    val = F.mse_loss(Phi_va @ c, r_va).item()
                    if b is None or val < b[0]:
                        b = (val, F.mse_loss(Phi_te @ c, r_te).item())
                return b
            diag_b = best([(P_diag, lam) for lam in LAMS])
            hyb_b = best([(a * P_diag + (1 - a) * P_gram, lam)
                          for lam in LAMS for a in ALPHAS])
            if diag_b is None or hyb_b is None:
                continue
            rel = (diag_b[1] - hyb_b[1]) / diag_b[1]
            pts.append((ang, rel, sig, seed))
            print(f"sigma_w={sig:g} seed={seed}: angle={ang:.1f} deg  "
                  f"diag={diag_b[1]:.4f} hybrid={hyb_b[1]:.4f}  "
                  f"benefit={rel:+.1%}")
    if len(pts) >= 4:
        angs = [p[0] for p in pts]
        bens = [p[1] for p in pts]
        print(f"\nangle range {min(angs):.1f}-{max(angs):.1f} deg; "
              f"Spearman(angle, hybrid benefit) = {spearman(angs, bens):+.2f} "
              f"(n={len(pts)})")
        Path("results").mkdir(exist_ok=True)
        Path("results/bridge_angle_sweep.json").write_text(json.dumps(
            [{"angle_deg": a, "rel_benefit": b, "sigma_w": s, "seed": sd}
             for a, b, s, sd in pts], indent=1))
        print("wrote results/bridge_angle_sweep.json")


# ---------------- recipe test: sigma parameterized over the transform ----------------
SIGMA_LADDER = [0.25, 0.5, 1.0, 2.0]   # log-spaced bandwidths, M/4 features each
RECIPE_LAMS = [1e-4, 1e-2, 1.0, 100.0]
RECIPE_ALPHAS = [0.2, 0.5, 0.8]
RECIPE_SHAPES = [                       # the lambda transverse polynomial shapes
    {"name": "quartic",        "degrees": [2],       "coefs": [1.0]},
    {"name": "quad+quartic",   "degrees": [1, 2],    "coefs": [1.0, 1.0]},
    {"name": "quartic+sextic", "degrees": [2, 3],    "coefs": [1.0, 1.0]},
    {"name": "full-123",       "degrees": [1, 2, 3], "coefs": [1.0, 1.0, 1.0]},
    {"name": "high-clamp",     "degrees": [1, 3],    "coefs": [0.1, 10.0]},
]
RECIPE_CELLS = Path("results/bridge_recipe_cells.jsonl")
RECIPE_OUT = Path("results/bridge_recipe_test.json")
RECIPE_NS = [512, 2048]
RECIPE_SEEDS = [0, 1, 2, 3, 4]


def _ladder_rff(d: int, seed: int):
    """RFF basis with sigma PARAMETERIZED OVER THE TRANSFORM — now native in
    SpectralReward (sigma_w as a list); identical W to the original in-place
    block scaling (same RNG stream, scale applied after the draw)."""
    from mbrl.models.spectral import SpectralReward
    return SpectralReward(d, n_features=N_FEATURES, sigma_w=SIGMA_LADDER,
                          seed=seed)


def recipe_cell(n: int, seed: int) -> dict:
    """Four recipe arms, shared data; selection on noisy validation.
    Arms: (1) single sigma=0.5 + quartic diag (run-2 best diag recipe);
    (2) sigma ladder + quartic diag; (3) sigma ladder + lambda polynomial
    diag (shapes x lam); (4) sigma ladder + lambda polynomial + Gram
    transverse cut (shapes x lam x alpha). Recipe search, NOT scale-matched —
    validation does the arbitration."""
    from mbrl.models.spectral import SpectralReward, poly_weights

    xa_tr, r_tr, xa_va, r_va, xa_te, r_te = make_data(n, seed)
    d = xa_tr.shape[1]
    row = {"n": n, "seed": seed}

    def best_of(sr, sweeps):
        Phi_tr, Phi_va, Phi_te = (sr.features(xa_tr), sr.features(xa_va),
                                  sr.features(xa_te))
        A0, rhs = Phi_tr.T @ Phi_tr, Phi_tr.T @ r_tr
        eye = torch.eye(sr.M)
        b = None
        for tag, P in sweeps(sr, Phi_tr):
            try:
                c = torch.linalg.solve(A0 + P + 1e-8 * eye, rhs)
            except RuntimeError:
                continue
            if not torch.isfinite(c).all():
                continue
            val = F.mse_loss(Phi_va @ c, r_va).item()
            if b is None or val < b[0]:
                b = (val, tag, F.mse_loss(Phi_te @ c, r_te).item())
        return {"val_mse": b[0], "choice": b[1], "test_mse": b[2]} if b else \
            {"degenerate": True}

    def diag_sweep(sr, _Phi):
        for lam in RECIPE_LAMS:
            yield f"lam={lam:g}", lam * torch.diag(sr.w4)

    def poly_sweep(sr, _Phi):
        for sh in RECIPE_SHAPES:
            w = poly_weights(sr.w2.sqrt(), sh["degrees"], sh["coefs"])
            for lam in RECIPE_LAMS:
                yield f"{sh['name']}@lam={lam:g}", lam * torch.diag(w)

    def poly_gram_sweep(sr, Phi):
        Pg = penalty_matrix("lap2_positive", sr, Phi, seed)
        for sh in RECIPE_SHAPES:
            w = poly_weights(sr.w2.sqrt(), sh["degrees"], sh["coefs"])
            Pd = torch.diag(w)
            for lam in RECIPE_LAMS:
                for a in RECIPE_ALPHAS:
                    yield (f"{sh['name']}@lam={lam:g}@a={a:g}",
                           lam * (a * Pd + (1 - a) * Pg))
        # alpha=1 limit is arm 3; alpha=0 limit was refuted in run 1

    sr05 = SpectralReward(d, n_features=N_FEATURES, sigma_w=0.5, seed=seed)
    row["single05_quartic"] = best_of(sr05, diag_sweep)
    srL = _ladder_rff(d, seed)
    row["ladder_quartic"] = best_of(srL, diag_sweep)
    row["ladder_poly"] = best_of(srL, poly_sweep)
    row["ladder_poly_gram"] = best_of(srL, poly_gram_sweep)
    return row


RECIPE_ARMS = ("single05_quartic", "ladder_quartic", "ladder_poly",
               "ladder_poly_gram")


def recipe_report(done: dict):
    print(f"\n{'n':>6} {'arm':>18} {'test_mse':>12}  choice* (per seed)")
    for n in RECIPE_NS:
        rows = [done[(n, s)] for s in RECIPE_SEEDS if (n, s) in done]
        if not rows:
            continue
        for arm in RECIPE_ARMS:
            ok = [r[arm] for r in rows if not r[arm].get("degenerate")]
            if not ok:
                continue
            mse = np.mean([e["test_mse"] for e in ok])
            ch = ";".join(e["choice"] for e in ok)
            print(f"{n:>6} {arm:>18} {mse:>12.4f}  {ch}")
    cells = [r for r in done.values()
             if all(not r[a].get("degenerate") for a in RECIPE_ARMS)]
    if cells:
        for arm in RECIPE_ARMS[1:]:
            wins = sum(1 for r in cells if r[arm]["test_mse"]
                       < r["single05_quartic"]["test_mse"])
            rel = np.mean([(r["single05_quartic"]["test_mse"] - r[arm]["test_mse"])
                           / r["single05_quartic"]["test_mse"] for r in cells])
            print(f"{arm} vs single05_quartic: wins {wins}/{len(cells)}, "
                  f"mean relative benefit {rel:+.1%}")


def recipe_run(budget: float):
    RECIPE_CELLS.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if RECIPE_CELLS.exists():
        for line in RECIPE_CELLS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["n"], r["seed"])] = r
    cells = [(n, s) for n in RECIPE_NS for s in RECIPE_SEEDS]
    t0 = time.perf_counter()
    for n, s in cells:
        if (n, s) in done or time.perf_counter() - t0 > budget:
            continue
        row = recipe_cell(n, s)
        with RECIPE_CELLS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done[(n, s)] = row
        print(f"done: n={n} seed={s} "
              + " ".join(f"{a}={row[a].get('test_mse', float('nan')):.4f}"
                         for a in RECIPE_ARMS))
    recipe_report(done)
    remaining = [c for c in cells if c not in done]
    if remaining:
        print(f"\n{len(remaining)} cells remaining — re-run to continue")
    else:
        RECIPE_OUT.write_text(json.dumps(list(done.values()), indent=1))
        print(f"\nall {len(cells)} cells done — wrote {RECIPE_OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=30.0)
    p.add_argument("--report", action="store_true")
    p.add_argument("--angle-sweep", action="store_true",
                   help="sigma_w sweep: move the diag-vs-Gram angle, test "
                        "whether hybrid benefit tracks it")
    p.add_argument("--recipe", action="store_true",
                   help="sigma-ladder + lambda-polynomial (+ Gram) recipe test")
    args = p.parse_args()
    if args.angle_sweep:
        angle_sweep()
        return
    if args.recipe:
        recipe_run(args.budget)
        return
    CELLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    cells = [(n, s) for n in NS for s in SEEDS]

    if not args.report:
        t0 = time.perf_counter()
        for n, s in cells:
            if (n, s) in done or time.perf_counter() - t0 > args.budget:
                continue
            row = run_cell(n, s)
            with CELLS_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done[(n, s)] = row
            msg = " ".join(f"{a}={row[a].get('test_mse', float('nan')):.4f}"
                           for a in ARMS)
            print(f"done: n={n} seed={s} {msg}")

    report(done)
    remaining = [c for c in cells if c not in done]
    if remaining:
        print(f"\n{len(remaining)} cells remaining — re-run to continue")
    else:
        OUT_PATH.write_text(json.dumps(list(done.values()), indent=1))
        print(f"\nall {len(cells)} cells done — wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
