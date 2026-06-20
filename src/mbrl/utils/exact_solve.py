"""Exact rational linear solve by p-adic (Hensel/Dixon) lifting — the "2-adic head".

Target: the closed-form spectral head-refit `SpectralReward.fit`, which solves the (M, M) ridge
system  A c = b ,  A = ΦᵀΦ + diag(weights) ,  b = Φᵀy .  Empirically that Gram is ILL-conditioned
(log10 cond ≈ 12, measured by scripts/analyze_loss_dynamics.py): in the float32 the model trains in
(~7 decimal digits) a direct `torch.linalg.solve` loses ALL precision. This lift instead solves the
GIVEN (exact-dyadic) system over ℚ with NO conditioning-induced error — O(n³) ring operations
(Dixon keeps the residual bounded, so it is the same order as one float solve, but exact).

On the prime p (the project's binary / p=2 program). Every float is an exact dyadic rational, so the
float system is cleared to integers by scaling with the common denominator (a power of 2):
A_int = A·den, b_int = b·den solve the SAME system (A_int x = b_int ⇔ A x = b). But that makes
det(A_int) = 2^(sM)·det(A) EVEN, so the matrix is singular mod 2 and the binary (p=2) Dixon lift
degenerates — p=2 works ONLY for the near-identity operator solves (det odd; see
paper/reproduction/padic_gpu.py), NOT for this Gram. So the head-refit lifts with an ODD prime p
chosen so A_int is invertible mod p (standard Dixon prime selection); p=2 remains available for the
near-identity case. The result is identical exact ℚ regardless of which prime carries the lift.

Cost: pure-Python arbitrary-precision integers (no overflow, O(n³) ring ops transparent), but
bignum-heavy — a correctness / certification tool, NOT yet a fast hot-path solver at M≈512 (the
GPU int64 = ℤ/2^64 Newton path in padic_gpu.py, with a GF(2) seed for the general case, is the
speed follow-up). Off by default; enabled via model spectral.exact_solve.
"""
from fractions import Fraction
import math

import torch

# Odd Mersenne/NTT-friendly primes for the Dixon lift; tried in order until one does not divide det.
_DIXON_PRIMES = (2147483647, 2147483629, 2147483587, 998244353, 167772161, 2013265921)


def _to_integer_system(A, b):
    """Lossless float→integer: each float is a dyadic rational; scale A and b by the common
    denominator `den` (a power of 2) so A_int = A·den, b_int = b·den are integers with the SAME
    solution. Returns (A_int rows, b_int, den). den cancels (both sides scaled) — kept for tests."""
    n = len(b)
    Af = [[Fraction(float(A[i][j])) for j in range(n)] for i in range(n)]
    bf = [Fraction(float(b[i])) for i in range(n)]
    den = 1
    for row in Af:
        for v in row:
            den = den // math.gcd(den, v.denominator) * v.denominator   # lcm
    for v in bf:
        den = den // math.gcd(den, v.denominator) * v.denominator
    A_int = [[int(v * den) for v in row] for row in Af]
    b_int = [int(v * den) for v in bf]
    return A_int, b_int, den


def _inverse_mod(A, p, w):
    """A⁻¹ mod p^w by Gauss-Jordan with pivots coprime to p (units mod p^w). O(n³).
    Returns None if A is singular mod p (det ≡ 0) — caller retries with another prime."""
    n = len(A)
    mod = p ** w
    M = [[A[i][j] % mod for j in range(n)] + [1 if j == i else 0 for j in range(n)]
         for i in range(n)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] % p != 0), None)   # a unit pivot mod p
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        inv = pow(M[c][c], -1, mod)
        M[c] = [(x * inv) % mod for x in M[c]]
        for r in range(n):
            f = M[r][c]
            if r != c and f:
                M[r] = [(M[r][k] - f * M[c][k]) % mod for k in range(2 * n)]
    return [row[n:] for row in M]


def _ratrec(a, m):
    """Rational reconstruction: p/q ≡ a (mod m), |p|, q ≤ √(m/2) (Wang's bound)."""
    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 < 0:
        r1, s1 = -r1, -s1
    return Fraction(r1, s1)


def padic_dixon_solve(A, b, p, w=30):
    """Exact rational solution of integer system A x = b by p-adic (base p^w) Dixon lifting.
    Requires A invertible mod p; returns list[Fraction] or None if singular mod p. O(n³)."""
    n = len(A)
    base = p ** w
    C = _inverse_mod(A, p, w)                                       # O(n³)
    if C is None:
        return None
    H = 1                                                          # Hadamard bound on |det|
    for i in range(n):
        H *= math.isqrt(sum(A[i][j] * A[i][j] for j in range(n))) + 1
    bb = max(1, max((abs(v) for v in b), default=1))
    digits = math.ceil(math.log(2 * H * H * bb + 2, base)) + 1     # k = O(n/w) digits
    r = list(b)
    X = [0] * n
    modulus = 1
    for _ in range(digits):                                        # O(digits · n²) = O(n³/w)
        xi = [sum(C[i][j] * (r[j] % base) for j in range(n)) % base for i in range(n)]
        for i in range(n):
            X[i] += xi[i] * modulus
        r = [(r[i] - sum(A[i][j] * xi[j] for j in range(n))) // base for i in range(n)]
        modulus *= base
    return [_ratrec(X[i] % modulus, modulus) for i in range(n)]


def exact_solve_fractions(A, b):
    """Exact A x = b for float tensors → list[Fraction]. Lifts with the first Dixon prime under
    which the integer system is nonsingular. Raises if singular mod every trial prime."""
    A_int, b_int, _ = _to_integer_system(A.detach().cpu(), b.detach().cpu())
    for p in _DIXON_PRIMES:
        sol = padic_dixon_solve(A_int, b_int, p)
        if sol is not None:
            return sol
    raise RuntimeError("exact_solve: integer system singular mod every trial prime "
                       "(matrix may be exactly singular)")


def exact_solve(A, b):
    """Exact rational solve of the float system A x = b, returned as a tensor on A's device/dtype.
    Drop-in for `torch.linalg.solve(A, b)` (b a vector) when conditioning makes the float solve
    untrustworthy. The solve carries ZERO error; remaining error is only the input rounding."""
    sol = exact_solve_fractions(A, b)
    return torch.tensor([float(s) for s in sol], dtype=A.dtype, device=A.device)
