"""CAS reproduction — derivations.md Section 3.

The latent covariance band  mu in [f, c]  maps, through the normal-operator relation
mu = q/(1 - |lambda|^2), to an ANNULUS on the operator eigenvalues:
   ceiling  mu <= c  <=>  |lambda|^2 <= 1 - q/c   (anti-freeze)
   floor    mu >= f  <=>  |lambda|^2 >= 1 - q/f   (anti-collapse)

Run:  python band_to_annulus.py
"""
import sympy as sp


def main():
    lam2, q, f, c = sp.symbols("lambda2 q f c", positive=True)  # lam2 = |lambda|^2
    mu = q / (1 - lam2)

    ceil = sp.solve(sp.Eq(mu, c), lam2)[0]
    flo = sp.solve(sp.Eq(mu, f), lam2)[0]
    assert sp.simplify(ceil - (1 - q / c)) == 0
    assert sp.simplify(flo - (1 - q / f)) == 0
    print(f"  mu = c  =>  |lambda|^2 = {ceil}   (= 1 - q/c, anti-freeze ceiling < 1)  OK")
    print(f"  mu = f  =>  |lambda|^2 = {flo}   (= 1 - q/f, anti-collapse floor)  OK")

    # the critical ratio r* = 1/5 and its covariance amplification 1/(1 - r*)
    r_star = sp.Rational(1, 5)
    amp = 1 / (1 - r_star)
    assert amp == sp.Rational(5, 4)
    print(f"  critical ratio r* = 1/5 = 1 - 4/5  =>  covariance amplification 1/(1-r*) = {amp}  OK")
    # |lambda| corresponding to r* (the init_shift used in experiment)
    print(f"  |lambda*| = sqrt(1/5) = {sp.nsimplify(sp.sqrt(r_star))} ~ {float(sp.sqrt(r_star)):.7f}")


if __name__ == "__main__":
    print("=== Band <=> operator eigenvalue annulus (derivations Section 3) ===")
    main()
    print("PASS")
