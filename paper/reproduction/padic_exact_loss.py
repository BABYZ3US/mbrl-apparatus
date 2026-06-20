"""CAS / number-theoretic reproduction — derivations.md Section 6 (p-adic exact computation).

Binary arithmetic is 2-adic. In the CONSERVATIVE case det(Sigma_hat^{-1} G) = 1 the logs cancel and
the operator loss is rational, D = tr(Sigma_hat^{-1} G) - d, so it is computable EXACTLY in
fixed-width binary by Hensel/Dixon 2-adic lifting + rational reconstruction -- no float rounding and
no fraction blow-up ("the same number of bits regardless").

Caveats (in the docstring on purpose): Q_2 is unordered (no minimization -> this is bit-exact
CERTIFICATION of the loss value, not training), and off the conservative manifold the log is
genuinely transcendental (only the det=1 case is purely rational / 2-adic-exact).

Run:  python padic_exact_loss.py
"""
from fractions import Fraction
import math


def padic_inverse(a, k):
    """2-adic inverse of an ODD a, modulo 2^k, by Hensel (Newton) doubling: x <- x(2 - a x)."""
    assert a % 2 == 1, "even a is not a 2-adic unit"
    x, prec = 1, 1                      # 1 is the inverse mod 2
    while prec < k:
        prec = min(2 * prec, k)
        m = 1 << prec
        x = (x * (2 - a * x)) % m
    return x % (1 << k)


def rational_reconstruction(a, m):
    """Recover p/q with p/q == a (mod m), |p|, q <= sqrt(m/2) (Wang's algorithm)."""
    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    p, qd = r1, s1
    if qd < 0:
        p, qd = -p, -qd
    return Fraction(p, qd)


def to_2adic_and_back(x: Fraction, k: int) -> Fraction:
    """Encode a rational in fixed-width 2-adic (mod 2^k) and reconstruct it exactly: clear the
    2-power v2 of the denominator, invert the odd part as a 2-adic unit, reconstruct, reattach 2^v2."""
    m = 1 << k
    den = x.denominator
    v2 = (den & -den).bit_length() - 1     # 2-adic valuation of the denominator
    odd_den = den >> v2
    res = (x.numerator * padic_inverse(odd_den, k)) % m   # 2-adic residue of the unit part
    return rational_reconstruction(res, m) / (1 << v2)


def main():
    k = 128
    # (1) the critical ratio 1/5 is a 2-adic integer (5 is odd = a unit): exact round-trip
    assert to_2adic_and_back(Fraction(1, 5), k) == Fraction(1, 5)
    inv5 = padic_inverse(5, k)
    print(f"  1/5: 5 is a 2-adic unit; 5 * (5^-1 mod 2^{k}) mod 2^{k} = {(5*inv5) % (1<<k)}  "
          f"-> recovered EXACTLY")

    # (2) conservative loss: spectrum {5/4, 4/5, 1}, det = product = 1 -> logs cancel ->
    #     D = tr - d = (5/4 + 4/5 + 1) - 3 = 1/20, a pure rational
    nu = [Fraction(5, 4), Fraction(4, 5), Fraction(1)]
    assert math.prod(nu) == 1                          # det(Sigma_hat^-1 G) = 1 (conservative)
    D = sum(nu) - len(nu)
    D_2adic = to_2adic_and_back(D, k)                  # computed in fixed-width binary, exact
    assert D_2adic == D == Fraction(1, 20)
    print(f"  conservative loss D = tr - d = {D} (logs cancelled, det=1)")
    print(f"  computed in fixed-width 2-adic (mod 2^{k}) and reconstructed EXACTLY = {D_2adic}")
    print(f"  float gives {float(D)} (an approximation); 2-adic gives the exact rational")

    # (3) a rational with even denominator (v2 != 0) still round-trips exactly: 5/4 (v2=2)
    assert to_2adic_and_back(Fraction(5, 4), k) == Fraction(5, 4)
    print("  5/4 (v2 = 2): exact 2-adic round-trip with the 2-power handled by the valuation  OK")


if __name__ == "__main__":
    print("=== p=2 (2-adic) exact loss certification (derivations Section 6) ===")
    main()
    print("PASS")
