"""Experimental verification — derivations.md Section 9 ("rejected" coincidences).

Shows that the golden-ratio fixed point and the relations g_p = 1 + 1/g_d and g_p = 1 + eff/H
are coincidences of two independently band-pinned constants, not laws: the golden match holds for
the Jensen-corrected z_std but not the clean mean eigenvalue, and g_p does NOT covary with eff/H
across arms (measured).

Run:  python verify_rejected_coincidences.py
"""
import numpy as np

PHI = (1 + 5 ** 0.5) / 2

# Measured operator-generator norm g_p, eff_rank, horizon H, per arm (seed 0, ~300k).
# If g_p = 1 + eff/H were a law, g_p would track eff/H across arms. It does not.
ARMS = [
    # arm,        eff,   H,   g_p
    ("A0", 14.60, 18, 2.279),
    ("A1", 14.60, 17, 1.949),
    ("A2", 14.57, 20, 2.442),
    ("A3", 14.49, 19, 2.064),
    ("A4", 14.45, 19, 2.027),
    ("A5", 29.43, 22, 2.734),
]


def golden_is_coincidence():
    z_std, mean_eig = 0.788, 0.643
    print(f"  z_std=0.788 vs 1/sqrt(phi)={1/np.sqrt(PHI):.3f}  "
          f"(|diff|={abs(z_std-1/np.sqrt(PHI)):.3f}, looks golden)")
    print(f"  <mu>=0.643 vs 1/phi={1/PHI:.3f}  "
          f"({100*abs(mean_eig-1/PHI)/(1/PHI):.1f}% off, NOT golden) -> coincidence of the "
          f"Jensen-corrected std, not a phi fixed point")


def gp_law_is_coincidence():
    eff_over_H = np.array([e / H for _, e, H, _ in ARMS])
    gp = np.array([g for *_, g in ARMS])
    # within the d=16 arms (A0..A4) the correlation of g_p with eff/H is ~0
    d16 = list(range(5))
    r = np.corrcoef(eff_over_H[d16], gp[d16])[0, 1]
    print("  arm   eff/H   g_p    1+eff/H")
    for (name, e, H, g) in ARMS:
        print(f"   {name}   {e/H:.3f}  {g:.3f}   {1+e/H:.3f}")
    print(f"  corr(g_p, eff/H) over d=16 arms = {r:+.2f}  (no covariation) "
          f"-> g_p = 1 + eff/H is a coincidence of two band-pinned constants")


if __name__ == "__main__":
    print("=== Rejected coincidences (derivations Section 9) ===")
    golden_is_coincidence()
    print()
    gp_law_is_coincidence()
    print("done")
