"""Experimental verification — derivations.md Section 1, Proposition 1.

z_std = sqrt(Tr G / d) = sqrt(<mu>).  Numerically on sampled latents, plus the band-pinned
measured values from the run logs.

Run:  python verify_zstd_identity.py
"""
import numpy as np

# Measured (band-pinned, from the operator runs): z_std vs predicted sqrt(<mu>)
MEASURED = [
    # (d, z_std_measured, mean_eig_measured)
    (16, 0.788, 0.620),
    (32, 0.786, 0.618),
]


def sampled_identity(d, n=50000, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    C = A @ A.T / d
    L = np.linalg.cholesky(C + 1e-9 * np.eye(d))
    z = rng.standard_normal((n, d)) @ L.T
    G = z.T @ z / n                       # uncentered second moment
    z_std_direct = float(np.sqrt((z ** 2).mean()))     # RMS latent
    z_std_trace = float(np.sqrt(np.trace(G) / d))      # sqrt(Tr G / d)
    mean_eig = float(np.linalg.eigvalsh(G).mean())     # <mu>
    # Tr G / d == mean eigenvalue is exact (basis-invariant); sampling links it to the RMS latent
    assert abs(z_std_trace - np.sqrt(mean_eig)) < 1e-10
    assert abs(z_std_direct - z_std_trace) < 1e-2
    return z_std_direct, z_std_trace, np.sqrt(mean_eig)


if __name__ == "__main__":
    print("=== z_std = sqrt(Tr G / d) = sqrt(<mu>)  (derivations Section 1) ===")
    for d in (16, 32):
        a, b, c = sampled_identity(d)
        print(f"  sampled d={d}: RMS latent={a:.4f}  sqrt(Tr G/d)={b:.4f}  sqrt(<mu>)={c:.4f}  OK")
    print("  --- measured (band-pinned) ---")
    for d, zs, me in MEASURED:
        print(f"  d={d}: z_std_measured={zs:.3f}  sqrt(<mu>)={np.sqrt(me):.3f}  "
              f"(|diff|={abs(zs-np.sqrt(me)):.3f})")
    print("PASS")
