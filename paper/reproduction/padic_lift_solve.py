"""Exact rational linear solve by 2-adic (p=2) Hensel/Dixon lifting, in O(n^3).

Solves  A x = b  over Q exactly (A integer, nonsingular mod 2) without forming a single fraction
until the end -- the residuals stay bounded, so it is O(n^3) RING operations, the same order as one
floating-point solve, but exact. This is the engine under the conservative-case operator loss
(derivations.md Section 6 Remark): tr(Sigma_hat^{-1} G) - d is rational, so solving Sigma_hat Y = G
column-wise with this routine certifies the loss in fixed-width binary.

Complexity (n x n, base beta = 2^w, det odd):
  * inverse of A mod beta (Gauss-Jordan with odd pivots):           O(n^3)
  * lift to base^k, k = O(n/w) digits (Hadamard bound), O(n^2)/digit: O(n^3 / w)
  * rational reconstruction (n components):                          O(n * polylog)
  -> O(n^3) ring operations. (Bit complexity carries the usual log(n||A||) factors.)
Why it does not blow up: Dixon's residual update r <- (r - A x_i)/beta keeps |r| <= ||A|| * n
bounded at every step, unlike fraction-free Gaussian elimination whose intermediates grow like det.

Pure Python (arbitrary-precision ints, no overflow) so the O(n^3) operation count is transparent.

Run:  python padic_lift_solve.py
"""
from fractions import Fraction
import math
import time


def _inverse_mod_2adic(A, w):
    """A^{-1} mod 2^w by Gauss-Jordan choosing ODD pivots. Requires det(A) odd. O(n^3)."""
    n = len(A)
    m = 1 << w
    M = [[A[i][j] % m for j in range(n)] + [1 if j == i else 0 for j in range(n)]
         for i in range(n)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] & 1), None)   # an odd (=unit) pivot
        if piv is None:
            raise ValueError("A is not invertible mod 2 (det even): pick another base or precondition")
        M[c], M[piv] = M[piv], M[c]
        inv = pow(M[c][c], -1, m)
        M[c] = [(x * inv) % m for x in M[c]]
        for r in range(n):
            f = M[r][c]
            if r != c and f:
                M[r] = [(M[r][k] - f * M[c][k]) % m for k in range(2 * n)]
    return [row[n:] for row in M]


def _ratrec(a, m):
    """Rational reconstruction: p/q == a (mod m), |p|,q <= sqrt(m/2) (Wang)."""
    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 < 0:
        r1, s1 = -r1, -s1
    return Fraction(r1, s1)


def padic_solve(A, b, w=30):
    """Exact rational solution of A x = b via 2-adic (base 2^w) Dixon lifting. O(n^3)."""
    n = len(A)
    base = 1 << w
    C = _inverse_mod_2adic(A, w)                                  # O(n^3)
    H = 1                                                          # Hadamard bound on |det|
    for i in range(n):
        H *= math.isqrt(sum(A[i][j] * A[i][j] for j in range(n))) + 1
    bb = max(1, abs(max(b, key=abs))) if b else 1
    digits = math.ceil(math.log(2 * H * H * bb + 2, base)) + 1     # k = O(n/w)
    r = list(b)
    X = [0] * n
    modulus = 1
    for _ in range(digits):                                       # O(digits * n^2) = O(n^3/w)
        xi = [sum(C[i][j] * (r[j] % base) for j in range(n)) % base for i in range(n)]
        for i in range(n):
            X[i] += xi[i] * modulus
        r = [(r[i] - sum(A[i][j] * xi[j] for j in range(n))) // base for i in range(n)]
        modulus *= base
    return [_ratrec(X[i] % modulus, modulus) for i in range(n)]


def conservative_loss_via_lift(Sigma, G, w=30):
    """Exact conservative-case operator loss  D = tr(Sigma^{-1} G) - n  by 2-adic lift: solve
    Sigma y = G[:,j] for each column j and accumulate the diagonal entry y[j]. This equals the full
    Stein's loss when det(Sigma^{-1} G) = 1 (logs cancel); tr - n is exact rational regardless.
    n single O(n^3) solves -> O(n^4) for the whole trace (a Hensel matrix-inverse lift would give the
    trace in O(n^3 log n)). Returns a Fraction."""
    n = len(Sigma)
    tr = Fraction(0)
    for j in range(n):
        y = padic_solve(Sigma, [G[i][j] for i in range(n)], w)
        tr += y[j]
    return tr - n


# ---- verification + O(n^3) scaling -----------------------------------------------------------

def _exact_solve_reference(A, b):
    """Reference: fraction Gaussian elimination (exact) for correctness checking."""
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


def _odd_det_matrix(n, seed):
    """Random integer matrix with odd determinant (invertible mod 2): unit lower x unit upper."""
    rng = _LCG(seed)
    L = [[(1 if i == j else (rng() % 5 - 2) if i > j else 0) for j in range(n)] for i in range(n)]
    U = [[(1 if i == j else (rng() % 5 - 2) if i < j else 0) for j in range(n)] for i in range(n)]
    return [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


class _LCG:
    def __init__(self, s): self.s = s & 0xFFFFFFFF
    def __call__(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s


if __name__ == "__main__":
    print("=== 2-adic (p=2) Dixon lift: exact rational solve in O(n^3) ===")
    # correctness vs exact reference
    for n, seed in [(6, 1), (12, 2), (20, 3)]:
        A = _odd_det_matrix(n, seed)
        b = [(_LCG(seed * 7 + i)() % 19 - 9) for i in range(n)]
        x = padic_solve(A, b)
        xref = _exact_solve_reference(A, b)
        assert x == xref, (n, x, xref)
        # residual is exactly zero (A x == b over Q)
        res = [sum(Fraction(A[i][j]) * x[j] for j in range(n)) - b[i] for i in range(n)]
        assert all(v == 0 for v in res)
        print(f"  n={n:3d}: exact solve matches reference, A x - b == 0 exactly  OK")

    # O(n^3) scaling (time should ~8x per doubling of n)
    print("  --- timing (expect ~n^3: ratio ~8x per doubling) ---")
    prev = None
    for n in (16, 32, 64):
        A = _odd_det_matrix(n, 99)
        b = [(_LCG(n + i)() % 99 - 49) for i in range(n)]
        t = time.perf_counter()
        padic_solve(A, b)
        dt = time.perf_counter() - t
        ratio = f"  ({dt/prev:.1f}x vs n/2)" if prev else ""
        print(f"  n={n:3d}: {dt*1e3:8.1f} ms{ratio}")
        prev = dt

    # loss application: exact tr(Sigma^{-1} G) - n via the lift, checked against the reference
    n = 8
    S = _odd_det_matrix(n, 5)
    Gm = _odd_det_matrix(n, 6)
    D = conservative_loss_via_lift(S, Gm)
    Yref = [_exact_solve_reference(S, [Gm[i][j] for i in range(n)]) for j in range(n)]
    Dref = sum(Yref[j][j] for j in range(n)) - n
    assert D == Dref
    print(f"  loss tr(Sigma^-1 G) - n computed exactly by lift = {D}  (matches reference)  OK")
    print("PASS")
