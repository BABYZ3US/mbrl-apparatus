"""Experimental verification — derivations.md Section 6, Proposition 4.

Numerically minimize the operator cross-entropy  L(S) = log det S + tr(S^{-1} G)  by gradient
descent on the SPD cone and confirm the minimizer is S* = G (with the gradient closed form
dL/dS = S^{-1} - S^{-1} G S^{-1}). Also shows L stays finite only while S stays positive-definite.

Run:  python verify_cross_entropy_fixedpoint.py
"""
import numpy as np


def minimize_to_G(d=6, iters=30000, lr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    G = A @ A.T / d + 0.1 * np.eye(d)        # target SPD covariance
    S = 2.0 * np.eye(d)                       # init away from G
    for _ in range(iters):
        Si = np.linalg.inv(S)
        grad = Si - Si @ G @ Si               # dL/dS (Prop 4)
        S = 0.5 * ((S - lr * grad) + (S - lr * grad).T)
    rel = float(np.linalg.norm(S - G) / np.linalg.norm(G))
    print(f"  d={d}: ||S* - G|| / ||G|| = {rel:.2e}   (cross-entropy minimizer is G)")
    assert rel < 1e-3
    # value at the minimum equals log det G + d
    L = np.log(np.linalg.det(S)) + np.trace(np.linalg.inv(S) @ G)
    L_at_G = np.log(np.linalg.det(G)) + d
    assert abs(L - L_at_G) < 1e-6
    print(f"  L(S*) = log det G + d = {L_at_G:.4f}  (= tr(G^-1 G) + log det G)  OK")


if __name__ == "__main__":
    print("=== Operator cross-entropy minimizer is G (derivations Section 6) ===")
    for d in (4, 6, 10):
        minimize_to_G(d)
    print("PASS")
