"""Spectral reward solver: random Fourier features + closed-form weighted ridge.

The FFT idea, correctly realized for SCATTERED data: a regular-grid FFT does not
apply to replay-buffer samples, but a random Fourier feature expansion gives the
same payoff — the curvature penalty becomes DIAGONAL in frequency space, so the
H^2-penalized fit is a single linear solve instead of 1500 Adam epochs with
Hutchinson probes.

Math (derivation of the exact penalty constant):
    R(x)            = sum_j c_j phi_j(x),   phi_j(x) = sqrt(2/M) cos(w_j . x + b_j)
    grad^2 phi_j(x) = -sqrt(2/M) cos(w_j . x + b_j) w_j w_j^T            (rank-1)
    ||grad^2 R||_F^2 = (2/M) sum_{j,k} c_j c_k cos(th_j) cos(th_k) (w_j . w_k)^2

Taking the expectation over the i.i.d. uniform phases b_j ~ U[0, 2pi) (equivalently,
over x for spread-out data): E[cos th_j cos th_k] = (1/2) delta_{jk}, so the cross
terms vanish EXACTLY in expectation and

    E ||grad^2 R||_F^2 = (1/M) sum_j c_j^2 |w_j|^4.

Because each feature's Hessian is rank-1, ||w w^T||_F^2 = |w|^4 = (tr w w^T)^2, and

    E [(Delta R)^2]    = (1/M) sum_j c_j^2 |w_j|^4   — the SAME diagonal form:

the Frobenius and squared-Laplacian penalties coincide exactly-in-expectation in
this basis (the null-Lagrangian equivalence, R5: same Euler-Lagrange; here it is
not merely the same EL equation but the same penalty VALUE, because the rank-1
feature Hessians kill the off-diagonal (sum_i k_i k_j, i != j) cross terms that
distinguish (tr H)^2 from ||H||_F^2 pointwise).

The penalized least-squares objective  ||Phi c - y||^2 + lam * sum_j c_j^2 |w_j|^4
is then closed-form ridge with a diagonal weight:

    c = (Phi^T Phi + lam * diag(|w|^4))^{-1} Phi^T y      (one (M, M) solve)

— no Hutchinson probes, no gradient descent, exact penalty, O(N M^2 + M^3).

What does NOT transfer: the clamped trace estimator's max(est, 0) coherence
rectifier (configs/base.yaml clamp_trace) is nonlinear in coefficient space and
has no diagonal form here — open question; this module implements the UNCLAMPED
penalty exactly.

Integrated into the Trainer via the `spectral:` config block (spectral.enabled
=> the reward path is an ensemble of these heads, refit closed-form from a
rolling cache with poly_weights per-band schedules); also standalone for the
supervised benchmark (scripts/spectral_benchmark.py).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def poly_weights(omega_norms: Tensor, degrees, coefs) -> Tensor:
    """Per-feature polynomial penalty weights sum_d coefs[d] * |w_j|^(2*degrees[d]).

    Generalizes the pure quartic |w|^4 (degrees=[2], coefs=[1.0] — the H^2
    penalty) to a polynomial P(|w|^2): different degrees weight different
    frequency BANDS (low degree -> low frequencies barely touched, high degree
    -> high frequencies clamped hard), and time-varying coefs (e.g. per-degree
    phase-shifted lambda schedules in the Trainer) tune bands at different
    points of training.
    """
    if len(degrees) != len(coefs):
        raise ValueError(f"degrees and coefs must align, got {len(degrees)} vs {len(coefs)}")
    w2 = torch.as_tensor(omega_norms, dtype=torch.float32).pow(2)
    out = torch.zeros_like(w2)
    for d, c in zip(degrees, coefs):
        out = out + float(c) * w2.pow(int(d))
    return out


def snr_band_weights(Phi: Tensor, y: Tensor, omega_norms: Tensor,
                     n_bands: int = 8, snr_clip: tuple = (1e-3, 1e3),
                     generator: "torch.Generator | None" = None):
    """Explicit SNR penalty: per-band Wiener ridge weights, no hand-tuned shape.

    The ledger's Wiener-filter identity (Tier 1) says the optimal filter passes
    bands with SNR > 1 and suppresses SNR < 1. In the (near-orthogonal) RFF
    basis E[Phi'Phi] = (N/M) I, so ridge with per-feature weight theta_j
    shrinks the OLS coefficient by (N/M) / (N/M + theta_j); choosing

        theta_j = (N/M) / SNR_band(j)

    gives the Wiener shrinkage SNR/(1+SNR) per band — cutoff exactly at
    SNR = 1, derived from the data instead of dialed in by hand.

    Per-band SNR via INCREMENTAL split-half cross-fitting (no clean targets
    needed): bands processed low -> high |w|; each band's signal/noise is
    measured on the residual after lower bands' fits are subtracted (naive
    per-feature split-half is broken by feature correlation — low-frequency
    signal leaks into high-band coefficients consistently across halves and
    fakes SNR >> 1 in dead bands). Split halves separate target noise;
    residualization separates redundant signal. Bands = |w| quantile bins
    (align with the sigma ladder's blocks when one is in use, but binning is
    ladder-agnostic, so it also works for scalar sigma_w).

    Returns (weights (M,), info) — info has per-band |w| centers, SNRs, and
    sigma_eff = center/sqrt(d) plus the interpolated sigma where SNR crosses 1
    (the user's hypothesis, 2026-06-08: crossing at sigma = 1)."""
    N, M = Phi.shape
    perm = torch.randperm(N, generator=generator)
    A, B = perm[: N // 2], perm[N // 2:]

    w = torch.as_tensor(omega_norms, dtype=torch.float32)
    edges = torch.quantile(w, torch.linspace(0, 1, n_bands + 1))
    weights = torch.empty(M)
    centers, snrs = [], []
    # INCREMENTAL (residual) SNR, bands low -> high. The naive per-feature
    # split-half estimate is broken by feature correlation: a high-band
    # feature's diag-approx coefficient picks up LEAKAGE from low-frequency
    # signal, consistent across halves, so dead bands fake SNR >> 1 (observed:
    # min band SNR ~ 10, no crossing, under-regularized everywhere). Per-band
    # Wiener is only meaningful for the signal a band adds BEYOND lower bands,
    # so: estimate each band's split-half SNR on the residual after the
    # already-processed bands' (Wiener-shrunk) fit is subtracted.
    resid = y.clone()
    order = torch.argsort(edges[:-1])     # low |w| first (quantiles are sorted)
    for b in order.tolist():
        hi_ok = (w <= edges[b + 1]) if b == n_bands - 1 else (w < edges[b + 1])
        mask = (w >= edges[b]) & hi_ok
        if not mask.any():
            continue
        Pb = Phi[:, mask]
        Mb = int(mask.sum())
        cA = (M / len(A)) * (Pb[A].T @ resid[A])
        cB = (M / len(B)) * (Pb[B].T @ resid[B])
        sig = (cA * cB).mean().clamp_min(0.0)     # incremental signal power
        noi = ((cA - cB).pow(2) / 4.0).mean().clamp_min(1e-12)
        snr = float((sig / noi).clamp(*snr_clip))
        weights[mask] = (N / M) / snr
        centers.append(float(w[mask].mean()))
        snrs.append(snr)
        # subtract this band's Wiener-regularized fit from the residual:
        # small ridge solve on the band block only (Mb x Mb, trivial); the
        # ridge weight (N/M)/SNR IS the Wiener shrinkage — no extra factor
        cb = torch.linalg.solve(Pb.T @ Pb + (N / M) / max(snr, 1e-6)
                                * torch.eye(Mb), Pb.T @ resid)
        resid = resid - Pb @ cb
    # |w| where SNR crosses 1 (log-linear interpolation between band centers).
    # sigma_eff = w_at_snr1 / sqrt(in_dim): |w| ~ sigma*sqrt(d) for N(0, s^2 I)
    # rows — divide by sqrt(in_dim) caller-side to test the sigma=1 hypothesis.
    import math as _math
    info = {"band_centers": centers, "band_snrs": snrs,
            "edges": [float(e) for e in edges]}
    for i in range(1, len(snrs)):
        a, b_ = snrs[i - 1], snrs[i]
        if (a - 1.0) * (b_ - 1.0) <= 0 and a != b_:
            la, lb = _math.log(max(a, 1e-12)), _math.log(max(b_, 1e-12))
            t = (0.0 - la) / (lb - la)
            info["w_at_snr1"] = centers[i - 1] + t * (centers[i] - centers[i - 1])
            break
    return weights, info


def calibrate_sigma_ladder(X: Tensor, y: Tensor, mults=(0.5, 1.0, 2.0, 4.0),
                           probe_lo: float = 0.05, probe_hi: float = 8.0,
                           probe_rungs: int = 12, n_features: int = 512,
                           seed: int = 0):
    """Set the sigma ladder FROM THE DATA: measure the SNR=1 crossing sigma*
    with a wide log-spaced probe basis, then place the production rungs at
    sigma* x mults (bridge run 4: the crossing is measurable to ~3% across
    seeds; this uses the SNR machinery for measurement — what it is
    demonstrably good at — while the validated lambda-polynomial stays the
    penalty). Returns (ladder: list[float], info) with info["sigma_star"],
    the probe band SNRs, and info["calibrated"]=False on the no-crossing
    fallback (sigma* = geometric middle of bands with SNR > 1, or 1.0)."""
    import math as _math

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    d = X.shape[1]
    probe = [_math.exp(t) for t in torch.linspace(
        _math.log(probe_lo), _math.log(probe_hi), probe_rungs).tolist()]
    sr = SpectralReward(d, n_features=n_features, sigma_w=probe, seed=seed)
    _, info = snr_band_weights(sr.features(X), y, sr.w2.sqrt(),
                               n_bands=probe_rungs,
                               generator=torch.Generator().manual_seed(seed + 9))
    if "w_at_snr1" in info:
        sigma_star = info["w_at_snr1"] / _math.sqrt(d)
        calibrated = True
    else:  # no crossing inside the probe range: fall back conservatively
        live = [c for c, s in zip(info["band_centers"], info["band_snrs"]) if s > 1]
        sigma_star = (live[len(live) // 2] / _math.sqrt(d)) if live else 1.0
        calibrated = False
    ladder = [float(sigma_star * m) for m in mults]
    info = {"sigma_star": float(sigma_star), "calibrated": calibrated,
            "probe_band_snrs": info["band_snrs"], "ladder": ladder}
    return ladder, info


class SpectralReward:
    """R(x) = sum_j c_j sqrt(2/M) cos(w_j . x + b_j), w_j ~ N(0, sigma_w^2 I),
    b_j ~ U[0, 2pi).
    The H^2 (Frobenius-Hessian) penalty is DIAGONAL in this basis:
    E_x ||grad^2 R||_F^2 = (1/M) sum_j c_j^2 |w_j|^4 — exact (over the uniform
    phases; see module docstring for the constant's derivation), no Hutchinson
    probes, no gradient descent. Penalized fit is closed-form ridge:
    c = (Phi^T Phi + lam * diag(|w|^4))^{-1} Phi^T y.
    """

    def __init__(self, in_dim: int, n_features: int = 512,
                 sigma_w: "float | list[float]" = 1.0,
                 seed: int = 0, device: str = "cpu"):
        """sigma_w: scalar bandwidth, or a list = SIGMA LADDER (sigma
        parameterized over the transform): feature block k (M/K features)
        is drawn at bandwidth sigma_w[k], giving a multi-scale frame with
        genuinely separated frequency bands for poly_weights to dose
        (bridge run 3: ladder x polynomial = the winning recipe; ladder
        alone +6.6%, ladder + polynomial +33.7% over single sigma).
        Scalar path is bitwise-identical to the original (same RNG stream,
        scale applied after the draw)."""
        self.in_dim, self.M = in_dim, n_features
        self.sigma_w = sigma_w
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(seed)  # CPU generator: reproducible across backends
        W = torch.randn(n_features, in_dim, generator=g)
        if isinstance(sigma_w, (int, float)):
            W = W * float(sigma_w)
        else:                                   # sigma ladder over blocks
            sigmas = [float(s) for s in sigma_w]
            if not sigmas:
                raise ValueError("sigma_w ladder must be non-empty")
            K = len(sigmas)
            blk = n_features // K
            if blk == 0:
                raise ValueError(f"n_features={n_features} < ladder rungs {K}")
            for k, sig in enumerate(sigmas):    # last block absorbs remainder
                W[k * blk: n_features if k == K - 1 else (k + 1) * blk] *= sig
        self.W = W.to(self.device)
        self.b = (2.0 * math.pi * torch.rand(n_features, generator=g)).to(self.device)
        self.c = torch.zeros(n_features, device=self.device)
        # |w_j|^2 and |w_j|^4 — diagonal curvature weights (precomputed once)
        self.w2 = self.W.pow(2).sum(-1)
        self.w4 = self.w2.pow(2)
        self.lam = None  # last fit's lam (for reporting)

    # ---------------- features / prediction ----------------
    def features(self, X: Tensor) -> Tensor:
        """(N, M) random Fourier features sqrt(2/M) cos(W x + b). Pure torch and
        differentiable in X — predict() can be handed to hvp_penalty directly."""
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return math.sqrt(2.0 / self.M) * torch.cos(X @ self.W.T + self.b)

    def predict(self, X: Tensor) -> Tensor:
        """(N,) predictions Phi(X) c. Differentiable w.r.t. X (no detach)."""
        return self.features(X) @ self.c

    __call__ = predict

    # ---------------- closed-form penalized fit ----------------
    def fit(self, X: Tensor, y: Tensor, lam: float | None = None,
            weights: Tensor | None = None) -> "SpectralReward":
        """Solve c = (Phi^T Phi + diag(weights) + 1e-8 I)^{-1} Phi^T y.

        weights: per-feature penalty vector (M,). Default (weights=None) is
        the classic H^2 quartic lam * |w|^4; pass poly_weights(...) for
        polynomial / per-band schedules. The 1e-8 ridge floor keeps the system
        well-conditioned when weights ~ 0 (some |w_j| can be tiny, and
        Phi^T Phi is rank <= N for N < M)."""
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        Phi = self.features(X)                                   # (N, M)
        if weights is None:
            if lam is None:
                raise ValueError("fit() needs lam or an explicit weights vector")
            weights = lam * self.w4
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        A = Phi.T @ Phi + torch.diag(weights + 1e-8)
        self.c = torch.linalg.solve(A, Phi.T @ y)
        self.lam = float(lam) if lam is not None else None
        return self

    # ---------------- exact penalty values ----------------
    def hessian_frobenius_sq(self) -> float:
        """EXACT E_x ||grad^2 R||_F^2 = (1/M) sum_j c_j^2 |w_j|^4 (expectation
        over the uniform phases; cross terms vanish — verified against an
        autograd Hutchinson estimate in tests/test_spectral.py)."""
        return float((self.c.pow(2) * self.w4).sum().item() / self.M)

    def laplacian_trace_sq(self) -> float:
        """EXACT E_x [(Delta R)^2] = (1/M) sum_j c_j^2 |w_j|^4 — identical to
        hessian_frobenius_sq(): rank-1 feature Hessians make ||H_j||_F^2 =
        (tr H_j)^2 per feature, and the j != k cross terms vanish in
        expectation — the null-Lagrangian equivalence holds exactly-in-
        expectation in this basis (not just at the Euler-Lagrange level, R5).
        NOTE: the UNCLAMPED quantity; the max(est, 0) coherence rectifier of
        laplacian_trace_penalty is nonlinear in c and has no diagonal form."""
        return self.hessian_frobenius_sq()
