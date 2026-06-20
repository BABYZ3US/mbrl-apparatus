"""CAS reproduction — derivations.md Section 6, Theorem 1 and Proposition 4.

The operator cross-entropy  L = log det Sigma_hat + tr(Sigma_hat^{-1} G)  is the Gaussian
cross-entropy of N(0,G) under N(0,Sigma_hat); its KL form is Stein's loss; its unique minimizer
is Sigma_hat = G.

Run:  python operator_cross_entropy.py
"""
import sympy as sp


def cross_entropy_and_fixed_point():
    """2x2: build L = log det S + tr(S^{-1} G); show grad_S L = 0 exactly at S = G."""
    s1, s2, s3 = sp.symbols("s1 s2 s3", real=True)
    g1, g2, g3 = sp.symbols("g1 g2 g3", real=True)
    S = sp.Matrix([[s1, s2], [s2, s3]])
    G = sp.Matrix([[g1, g2], [g2, g3]])
    L = sp.log(S.det()) + (S.inv() * G).trace()
    print("  L = log det(Sigma_hat) + tr(Sigma_hat^{-1} G)        [Theorem 1]")
    grad_at_G = [sp.simplify(sp.diff(L, v).subs({s1: g1, s2: g2, s3: g3}))
                 for v in (s1, s2, s3)]
    assert all(g == 0 for g in grad_at_G), grad_at_G
    print("  grad_{Sigma_hat} L = 0  exactly at  Sigma_hat = G   [Prop 4 fixed point]  OK")


def stein_loss_nonnegative():
    """Scalar KL = 1/2[gamma/sigma - log(gamma/sigma) - 1]: minimum at sigma=gamma, value 0."""
    sigma, gamma = sp.symbols("sigma gamma", positive=True)
    KL = sp.Rational(1, 2) * (gamma / sigma - sp.log(gamma / sigma) - 1)
    crit = sp.solve(sp.diff(KL, sigma), sigma)
    assert gamma in crit
    assert sp.simplify(KL.subs(sigma, gamma)) == 0
    # second derivative positive at the minimum => it is a minimum (KL >= 0)
    assert sp.simplify(sp.diff(KL, sigma, 2).subs(sigma, gamma)) > 0
    print("  KL = 1/2[gamma/sigma - log(gamma/sigma) - 1] >= 0, min 0 at sigma=gamma  OK")


def subsumes_positivity():
    """tr(Sigma_hat^{-1} G) -> +inf as Sigma_hat loses rank (the built-in det>0 barrier)."""
    eps = sp.symbols("eps", positive=True)
    # Sigma_hat with a vanishing eigenvalue eps, G = I (mass in that direction)
    S = sp.Matrix([[eps, 0], [0, 1]])
    G = sp.eye(2)
    term = (S.inv() * G).trace()
    lim = sp.limit(term, eps, 0, "+")
    assert lim == sp.oo
    print("  tr(Sigma_hat^{-1} G) -> +oo as an eigenvalue -> 0  (positivity barrier built in)  OK")


if __name__ == "__main__":
    print("=== Operator cross-entropy = Stein's loss (derivations Section 6) ===")
    cross_entropy_and_fixed_point()
    stein_loss_nonnegative()
    subsumes_positivity()
    print("PASS")
