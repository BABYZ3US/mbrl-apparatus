"""CAS reproduction — derivations.md Section 7.

The discrete-time LQR Riccati recursion and its transpose-duality with the Lyapunov/Stein
equation of Section 2 (estimation <-> control).

Run:  python lqr_riccati_duality.py
"""
import sympy as sp


def scalar_dare():
    """Scalar discrete algebraic Riccati equation and its stabilizing root."""
    A, B, M, R, P = sp.symbols("A B M R P", positive=True)
    dare = sp.Eq(P, A**2 * P - (A * B * P) ** 2 / (R + B**2 * P) + M)
    sols = sp.solve(dare, P)
    print("  scalar DARE  P = A^2 P - (A B P)^2/(R + B^2 P) + M")
    for s in sols:
        print("    root:", sp.simplify(s))
    # gain and closed-loop, symbolic
    Ps = sp.symbols("P", positive=True)
    K = (B * Ps * A) / (R + B**2 * Ps)
    print(f"  optimal gain K = {K};  control a = -K d;  value V(d) = P d^2")
    # cost-to-go positivity: with M,R,P > 0 the value V = P d^2 >= 0  (P > 0 <=> det(op_p) > 0)
    print("  P > 0  <=>  V(d) = P d^2 positive-definite  <=>  det(op_p) > 0  (derivations Sec 4)")


def transpose_duality():
    """The Lyapunov map X -> A X A^T and the Riccati map X -> A^T X A are exchanged by A -> A^T."""
    a, b, c, d = sp.symbols("a b c d", real=True)
    x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4", real=True)
    A = sp.Matrix([[a, b], [c, d]])
    X = sp.Matrix([[x1, x2], [x3, x4]])
    lyap_map = A * X * A.T          # estimation (covariance propagation, Sec 2 / cross-entropy Sec 6)
    riccati_map = A.T * X * A       # control   (cost-to-go propagation, Sec 7)
    lyap_under_transpose = (A.T) * X * (A.T).T
    assert sp.simplify(lyap_under_transpose - riccati_map) == sp.zeros(2, 2)
    print("  X -> A X A^T  under  A -> A^T  equals  X -> A^T X A   "
          "(estimation <-> control duality)  OK")


if __name__ == "__main__":
    print("=== LQR Riccati and Lyapunov-Riccati duality (derivations Section 7) ===")
    scalar_dare()
    transpose_duality()
    print("PASS")
