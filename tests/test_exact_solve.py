"""The 2-adic head: exact p-adic (Dixon) rational solve of the ill-conditioned spectral head-refit
ridge (utils.exact_solve), and its wiring into SpectralReward.fit. Off by default; these tests
exercise the engine + the opt-in path. Pure-Python bignum ⇒ kept to small n for speed."""
import sys
from fractions import Fraction
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models.spectral import SpectralReward
from mbrl.utils.exact_solve import (exact_solve, exact_solve_fractions,
                                     padic_dixon_solve, _to_integer_system)


def _fraction_gauss(A, b):
    """Exact reference: fraction Gaussian elimination."""
    n = len(A)
    M = [[Fraction(A[i][j]) for j in range(n)] + [Fraction(b[i])] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [v / M[c][c] for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + 1)]
    return [M[i][n] for i in range(n)]


def test_dixon_matches_fraction_reference_and_zero_residual():
    """padic_dixon_solve (odd prime) == exact fraction reference, A x − b == 0 over ℚ."""
    A = [[4, 1, 0, 2], [1, 5, 1, 0], [0, 1, 6, 1], [2, 0, 1, 7]]   # SPD-ish integer
    b = [3, -2, 5, 1]
    x = padic_dixon_solve(A, b, p=2147483647)
    assert x == _fraction_gauss(A, b)
    res = [sum(Fraction(A[i][j]) * x[j] for j in range(4)) - b[i] for i in range(4)]
    assert all(v == 0 for v in res)


def test_to_integer_system_is_lossless_and_solution_preserving():
    """Float→int scaling by the dyadic common denominator preserves the solution exactly."""
    A = torch.tensor([[1.5, 0.25], [0.5, 2.0]], dtype=torch.float64)
    b = torch.tensor([1.0, -0.75], dtype=torch.float64)
    A_int, b_int, den = _to_integer_system(A, b)
    # A_int = A·den, b_int = b·den, both exact integers
    for i in range(2):
        assert b_int[i] == round(float(b[i]) * den)
        for j in range(2):
            assert A_int[i][j] == round(float(A[i][j]) * den)
    # solving the integer system gives the same x as the float system
    x = padic_dixon_solve(A_int, b_int, p=2147483647)
    ref = torch.linalg.solve(A, b)
    assert torch.allclose(torch.tensor([float(v) for v in x], dtype=torch.float64), ref, atol=1e-12)


def _vandermonde(n):
    return [[(i + 1) ** j for j in range(n)] for i in range(n)]


def test_exact_solve_recovers_truth_where_float32_fails():
    """On an ill-conditioned Vandermonde (cond ~1e9), the p-adic lift recovers the EXACT integer
    solution, while a float32 solve is materially wrong — the whole point of the 2-adic head."""
    n = 8
    A = _vandermonde(n)
    x_true = [(-1) ** i * (i + 2) for i in range(n)]
    b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    At64 = torch.tensor(A, dtype=torch.float64)
    bt64 = torch.tensor(b, dtype=torch.float64)

    sol = exact_solve_fractions(At64, bt64)
    assert sol == [Fraction(v) for v in x_true]              # EXACT recovery over ℚ

    x_exact = exact_solve(At64, bt64)
    err_exact = (x_exact - torch.tensor(x_true, dtype=torch.float64)).abs().max().item()
    x_f32 = torch.linalg.solve(torch.tensor(A, dtype=torch.float32),
                               torch.tensor(b, dtype=torch.float32))
    err_f32 = (x_f32 - torch.tensor(x_true, dtype=torch.float32)).abs().max().item()
    assert err_exact < 1e-8                                  # exact solve is exact
    assert err_f32 > err_exact                               # float32 is strictly worse here


def test_exact_solve_matches_torch_when_well_conditioned():
    """On a benign SPD system the exact solve agrees with torch.linalg.solve."""
    torch.manual_seed(0)
    n = 6
    Q = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))[0]
    A = Q @ torch.diag(torch.linspace(1.0, 4.0, n, dtype=torch.float64)) @ Q.T
    b = torch.randn(n, dtype=torch.float64)
    assert torch.allclose(exact_solve(A, b), torch.linalg.solve(A, b), atol=1e-9)


def test_spectral_head_exact_solve_matches_float_and_is_finite():
    """SpectralReward.fit with exact_solve=True trains a finite head and, on a well-conditioned
    ridge, matches the float fit. Small M to keep the pure-Python lift fast."""
    torch.manual_seed(0)
    X = torch.randn(64, 2)
    y = torch.randn(64)
    h_float = SpectralReward(2, n_features=16, sigma_w=1.0, seed=0, exact_solve=False).fit(X, y, lam=1.0)
    h_exact = SpectralReward(2, n_features=16, sigma_w=1.0, seed=0, exact_solve=True).fit(X, y, lam=1.0)
    assert h_exact.exact_solve is True and h_float.exact_solve is False
    assert torch.isfinite(h_exact.c).all()
    assert torch.allclose(h_exact.c, h_float.c, atol=1e-4)
