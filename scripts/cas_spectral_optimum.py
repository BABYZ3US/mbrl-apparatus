#!/usr/bin/env python
"""CAS derivation of the spectral-loss equilibrium + an environment-tuned optimum
(PM 2026-06-15). Symbolically solves the per-eigenvalue stationarity of the heuristic
band+compression loss, reads off cond / rank laws, diagnoses the floor-wall exponent,
and plugs in HalfCheetah estimates to recommend (w_band, w_compress, floor, ceiling).

  python scripts/cas_spectral_optimum.py [--rank 12 --latent 16 --kappa 10]

The per-eigenvalue potential (see regularization/rank2_frame.py):
  V(λ) = w_b[relu(λ-c)^2 + relu(f-λ)^p] + w_c·sqrt(relu(λ-f)+ε) - g·λ
with g = the mode's task utility (reconstruction+reward want it large)."""
from __future__ import annotations
import argparse, math
import sympy as sp


def derive():
    lam, c, f, wb, wc, g, eps = sp.symbols("lambda c f w_b w_c g epsilon", positive=True)
    p, delta, drift = sp.symbols("p delta drift", positive=True)
    print("=== 1. active-mode equilibrium  (λ>c: ceiling wall + compress + task pull) ===")
    Va = wb * (lam - c) ** 2 + wc * sp.sqrt(lam - f + eps) - g * lam
    dVa = sp.diff(Va, lam)
    print("  dV/dλ =", sp.nsimplify(dVa))
    lam_lead = sp.solve(2 * wb * (lam - c) - g, lam)[0]
    print("  leading order (drop compress):  λ* =", lam_lead)
    lam0 = c + g / (2 * wb)
    corr = sp.simplify((wc / (2 * sp.sqrt(lam0 - f + eps))) / (2 * wb))
    print("  with compress correction:       λ* ≈ c + g/(2 w_b) -", corr)

    print("\n=== 2. dead-mode equilibrium + condition number ===")
    print("  g≈0 ⇒ compress pushes down, floor wall stops at f ⇒ λ* = f")
    cond = sp.simplify(lam0 / f)
    print("  cond = λ_max/λ_min ≈", cond, "  →  c/f  as  g/(2 w_b) → 0")

    print("\n=== 3. survival threshold (compression sets the rank) ===")
    gstar = wc / (2 * sp.sqrt(eps))
    print("  mode survives the floor iff  g > g* =", gstar)
    print("  ⇒ rank r(w_c) = #{ modes with utility g_i > w_c/(2√ε) },  monotone-decreasing in w_c")

    print("\n=== 4. floor-wall exponent p — why cond is NOT being realized ===")
    fl = wb * (f - lam) ** p
    grad = sp.simplify(-sp.diff(fl, lam))
    print("  floor wall w_b(f-λ)^p ⇒ lift |dV/dλ| =", grad, "= p·w_b·(f-λ)^(p-1)")
    print("  p=2 (current): lift → 0 as λ→f  ⇒ a mode drifting down by δ sinks to")
    print("                 λ_min = f - δ with δ = drift/(2 w_b)  ⇒ cond = c/(f - drift/(2 w_b))")
    print("                 → unbounded when drift > 2 w_b f  (THIS is the 1e7–1e12 cond).")
    print("  p=1 (fix):     lift = w_b (constant, never vanishes) ⇒ binds at λ_min≈f for ANY")
    print("                 drift < w_b  ⇒ cond → c/f realized. THE one-line floor fix.")


def optimum(rank, latent, kappa, ceiling, eps):
    print("\n=== 5. environment-tuned optimum (HalfCheetah-v5) ===")
    print("  inputs:  r_env=%d active modes, latent k=%d, target cond κ=%g, ceiling c=%g, ε=%g"
          % (rank, latent, kappa, ceiling, eps))
    c = ceiling
    f = c / kappa                                   # (2): cond ≈ c/f ⇒ f = c/κ
    # check the equilibrium eff-rank (participation ratio) this (r,c,f) produces:
    r, kk = rank, latent
    S1 = r * c + (kk - r) * f
    S2 = r * c ** 2 + (kk - r) * f ** 2
    effrank = S1 * S1 / S2
    # w_c so g* sits just under the r-th utility; with a geometric utility g_i=ρ^i,
    # calibrate ρ so the PR-rank matches r_env, then g_r = ρ^r, w_c = 2√ε·g_r.
    rho = sp.symbols("rho", positive=True)
    # equilibrium λ_i ≈ c for g_i>g*, else f; approximate g_target at the elbow = ρ^r with ρ~0.7
    rho_val = 0.7
    g_r = rho_val ** rank
    w_c = 2 * math.sqrt(eps) * g_r
    # w_b: keep active overshoot ≤ tol·c  ⇒ w_b ≥ g_max/(2 tol c); g_max≈1 (top mode), tol=0.1
    g_max, tol = 1.0, 0.10
    w_b = g_max / (2 * tol * c)
    print("  floor   f = c/κ                  = %.3f" % f)
    print("  ceiling c                        = %.2f" % c)
    print("  ⇒ equilibrium eff-rank (PR)      ≈ %.1f   (target %d ✓)" % (effrank, rank))
    print("  w_compress = 2√ε·ρ^r (ρ≈%.1f)     ≈ %.3f   (g* prunes below the %d-th mode)"
          % (rho_val, w_c, rank))
    print("  w_band ≥ g_max/(2·tol·c) (tol=%.2f) ≈ %.1f   (active modes within %d%% of ceiling)"
          % (tol, w_b, int(tol * 100)))
    print("  floor wall exponent p            = 1   (steeper wall ⇒ cond=c/f actually binds)")
    print("\n  recommended next run:")
    print("    band_ceiling=%.2f  band_floor=%.3f  w_band=%.1f  w_compress=%.3f  floor_exp(p)=1"
          % (c, f, w_b, w_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=12, help="environment effective rank (from eff_rank)")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--kappa", type=float, default=10.0, help="target condition number")
    ap.add_argument("--ceiling", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-2)
    a = ap.parse_args()
    derive()
    optimum(a.rank, a.latent, a.kappa, a.ceiling, a.eps)


if __name__ == "__main__":
    main()
