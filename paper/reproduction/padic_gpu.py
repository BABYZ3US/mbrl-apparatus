"""GPU-vectorized 2-adic lift: batched exact integer inverse / solve mod 2^64, entirely in torch
int64 -- because two's-complement int64 IS the ring Z/2^64 (overflow = the 2-adic truncation), so
the mod-2^64 reduction is FREE. The Newton inverse step  X <- X(2I - A X)  is pure batched matmul,
so the whole lift is one GPU kernel sequence, vectorized over a batch of systems -- no CPU round
trip, no fraction blow-up. derivations.md Section 6 (Remark).

Requirement: A ≡ I (mod 2), so the seed X0 = I converges (the near-identity operators A = rawA + cI
are exactly this shape). The general case needs a GF(2) seed (batched Gauss-Jordan mod 2) -- a
drop-in follow-up. The final rational reconstruction is a light CPU step on the mod-2^64 residues.

Device-agnostic: pass CUDA tensors and it runs on the GPU unchanged.

Run:  python padic_gpu.py
"""
import math
from fractions import Fraction
import torch

M64 = 1 << 64


def _bmm(A, X):
    """Batched int64 matmul; int64 overflow gives the EXACT product in Z/2^64 (ring homomorphism:
    sums of products reduce mod 2^64 correctly). A:(B,n,k) X:(B,k,m) -> (B,n,m)."""
    return (A.unsqueeze(-1) * X.unsqueeze(-3)).sum(-2)


def padic_inverse_gpu(A: torch.Tensor, steps: int = 6) -> torch.Tensor:
    """Batched A^{-1} mod 2^64 by 2-adic Newton. A:(B,n,n) int64 with A ≡ I (mod 2).
    Each step doubles the valid precision (err E <- E^2), so `steps`=6 reaches mod 2^(2^6)=2^64.
    Pure batched matmul -> GPU-ideal; int64 overflow does the mod-2^64 reduction for free."""
    B, n, _ = A.shape
    I = torch.eye(n, dtype=torch.int64, device=A.device).expand(B, n, n).contiguous()
    X = I.clone()
    twoI = I << 1
    for _ in range(steps):
        X = _bmm(X, twoI - _bmm(A, X))
    return X


def _ratrec(a: int, m: int) -> Fraction:
    a %= m
    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 < 0:
        r1, s1 = -r1, -s1
    return Fraction(r1, s1)


def padic_solve_gpu(A: torch.Tensor, b: torch.Tensor, steps: int = 6):
    """Exact rational solve A x = b, batched: GPU 2-adic inverse + matvec, then CPU rational
    reconstruction of the mod-2^64 residues. A:(B,n,n) int64 (A ≡ I mod 2), b:(B,n) int64.
    Returns list[B] of list[n] of Fraction. (Use steps>6 + multi-limb for denominators > ~2^31.)"""
    Ainv = padic_inverse_gpu(A, steps)                         # GPU, batched
    x_mod = _bmm(Ainv, b.unsqueeze(-1)).squeeze(-1)            # x mod 2^64, GPU
    return [[_ratrec(int(v) & (M64 - 1), M64) for v in row] for row in x_mod.cpu().tolist()]


# ---- verification + batched demo -------------------------------------------------------------

def _rand_unit_diag(B, n, scale=3, seed=0):
    """Random integer batch with A ≡ I (mod 2): A = I + 2*N, so every A is invertible mod 2."""
    g = torch.Generator().manual_seed(seed)
    N = torch.randint(-scale, scale + 1, (B, n, n), generator=g, dtype=torch.int64)
    return torch.eye(n, dtype=torch.int64).expand(B, n, n) + 2 * N


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== GPU 2-adic lift (torch int64 = Z/2^64), device={dev} ===")

    # (1) inverse correctness: A * A^{-1} == I exactly mod 2^64, batched
    A = _rand_unit_diag(64, 8, seed=1).to(dev)
    Ai = padic_inverse_gpu(A)
    I = torch.eye(8, dtype=torch.int64, device=dev).expand_as(A)
    assert torch.equal(_bmm(A, Ai), I), "A A^{-1} != I mod 2^64"
    print("  batch=64 n=8: A @ A^{-1} == I exactly (mod 2^64) for ALL systems  OK")

    # (2) exact rational solve vs an independent Fraction reference
    A = _rand_unit_diag(5, 6, seed=2)
    b = torch.randint(-9, 10, (5, 6), dtype=torch.int64)
    X = padic_solve_gpu(A.to(dev), b.to(dev))
    for k in range(5):
        Ak = [[Fraction(int(A[k, i, j])) for j in range(6)] for i in range(6)]
        # residual A x - b must be exactly zero
        res = [sum(Ak[i][j] * X[k][j] for j in range(6)) - int(b[k, i]) for i in range(6)]
        assert all(v == 0 for v in res), (k, res)
    print("  batch=5 n=6: exact rational solve, A x - b == 0 exactly for ALL systems  OK")

    # (3) vectorization: invert a big batch in one kernel sequence
    import time
    A = _rand_unit_diag(4096, 16, seed=3).to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    _ = padic_inverse_gpu(A)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"  batch=4096 n=16: inverted in {1e3*(time.perf_counter()-t):.1f} ms "
          f"({dev}; 6 Newton steps, all on-device)")
    print("PASS")
