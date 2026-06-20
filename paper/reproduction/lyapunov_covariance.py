"""CAS reproduction — derivations.md Section 2, Proposition 3.

The discrete Lyapunov (Stein) equation  Sigma = A Sigma A^T + Q  and its normal-operator
diagonalization  Sigma_ii = q_i / (1 - |lambda_i|^2).

Run:  python lyapunov_covariance.py
"""
import sympy as sp
import numpy as np


def per_mode():
    """One eigen-mode: Sigma = lambda^2 Sigma + q  =>  Sigma = q/(1 - lambda^2)."""
    lam, q, Sigma = sp.symbols("lambda q Sigma", real=True, positive=True)
    sol = sp.solve(sp.Eq(Sigma, lam**2 * Sigma + q), Sigma)[0]
    assert sp.simplify(sol - q / (1 - lam**2)) == 0
    print(f"  per-mode Stein solution:  Sigma = {sol}   == q/(1-lambda^2)  OK")


def matrix_2x2():
    """Full symmetric (=> normal) 2x2 A and symbolic Q: solve the Stein equation and verify
    the closed form equals the operator series  Sigma = sum_k A^k Q (A^T)^k."""
    a, b, c = sp.symbols("a b c", real=True)
    A = sp.Matrix([[a, b], [b, c]])
    q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)
    Q = sp.Matrix([[q1, q2], [q2, q3]])
    s1, s2, s3 = sp.symbols("s1 s2 s3", real=True)
    S = sp.Matrix([[s1, s2], [s2, s3]])
    resid = S - (A * S * A.T + Q)
    sol = sp.solve([resid[0, 0], resid[0, 1], resid[1, 1]], [s1, s2, s3], dict=True)[0]
    Ssol = S.subs(sol)
    assert sp.simplify(Ssol - (A * Ssol * A.T + Q)) == sp.zeros(2, 2)
    print("  2x2 symmetric Stein equation solved; residual is identically zero  OK")

    subs = {a: 0.5, b: 0.1, c: 0.3, q1: 1.0, q2: 0.0, q3: 1.0}
    Sn = np.array(Ssol.subs(subs)).astype(float)
    An = np.array(A.subs(subs)).astype(float)
    Qn = np.array(Q.subs(subs)).astype(float)
    series = sum(
        np.linalg.matrix_power(An, k) @ Qn @ np.linalg.matrix_power(An.T, k)
        for k in range(300)
    )
    assert np.allclose(Sn, series, atol=1e-6)
    print("  closed form matches the convergent series sum_k A^k Q (A^T)^k to 1e-6  OK")


if __name__ == "__main__":
    print("=== Lyapunov/Stein covariance closed form (derivations Section 2) ===")
    per_mode()
    matrix_2x2()
    print("PASS")
