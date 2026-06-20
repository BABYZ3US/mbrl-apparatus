"""CAS reproduction — derivations.md Section 6 (separation form).

The operator cross-entropy / Stein's loss separates over the generalized spectrum:

    D(G || Sigma_hat) = tr(Sigma_hat^{-1} G) - log det(Sigma_hat^{-1} G) - d
                      = sum_i ( nu_i - log nu_i - 1 ) = sum_i phi(nu_i),

with nu_i = eig(Sigma_hat^{-1} G) (the generalized eigenvalues, read off a triangularization),
phi(nu) = nu - log nu - 1 the scalar Itakura-Saito / log-det per-mode loss, and the determinant
factor det(Sigma_hat^{-1} G) = prod_i nu_i (the "matrix product of per-mode loss operators").
Restricting the spectrum to positive rationals makes the loss exact (rational minus logs of
rationals); the critical ratio r = 1/5 lives entirely in Q.

Run:  python stein_loss_separation.py
"""
import numpy as np
import sympy as sp


def separation_numeric(d=6, seed=0):
    """General (non-diagonal) check: matrix Stein's loss == sum of scalar per-mode losses."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)); G = A @ A.T / d + 0.2 * np.eye(d)
    B = rng.standard_normal((d, d)); S = B @ B.T / d + 0.2 * np.eye(d)
    M = np.linalg.solve(S, G)                       # Sigma_hat^{-1} G
    D_matrix = np.trace(M) - np.log(np.linalg.det(M)) - d
    nu = np.linalg.eigvals(M).real                  # generalized eigenvalues
    D_sep = np.sum(nu - np.log(nu) - 1)
    assert abs(D_matrix - D_sep) < 1e-9
    assert abs(np.linalg.det(M) - np.prod(nu)) < 1e-8
    print(f"  general d={d}: D_matrix={D_matrix:.6f}  sum phi(nu_i)={D_sep:.6f}  "
          f"det = prod nu_i  OK")


def separation_exact_rational():
    """Positive RATIONAL generalized spectrum (e.g. the critical-ratio amplification 5/4 and its
    reciprocal): the loss is exact (rational minus logs of rationals)."""
    nu = [sp.Rational(5, 4), sp.Rational(4, 5), sp.Integer(1)]   # 1 contributes phi=0
    phi = [n - sp.log(n) - 1 for n in nu]
    D = sp.simplify(sum(phi))
    prod = sp.prod(nu)                       # the matrix-product (determinant) factor
    print(f"  rational spectrum nu = {nu}")
    print(f"  det factor  prod nu_i = {prod}")
    print(f"  D = sum phi(nu_i) = {D}   (exact: rational - logs of rationals)")
    print(f"  numeric D = {float(D):.6f}")
    # exactness: D has no floating point; its transcendental part is a Z-combination of log(prime)
    trans = sp.expand_log(D, force=True)
    print(f"  expanded (log-prime basis): {trans}")


if __name__ == "__main__":
    print("=== Stein's loss separates over the generalized spectrum (derivations Section 6) ===")
    for d in (4, 6, 10):
        separation_numeric(d)
    print()
    separation_exact_rational()
    print("PASS")
