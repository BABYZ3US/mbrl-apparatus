"""SpectralReward: closed-form RFF ridge with an exact diagonal H^2 penalty.

The load-bearing test is the autograd cross-check: hessian_frobenius_sq()
(the exact closed form (1/M) sum c_j^2 |w_j|^4) must match the Hutchinson
HVP estimate (mbrl.regularization.hutchinson.hvp_penalty, many probes) of
E_x ||grad^2 R||_F^2 on random data within 10% — same estimator the Trainer
uses, so the two penalty implementations are mutually validating.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from mbrl.models.spectral import SpectralReward
from mbrl.regularization.hutchinson import hvp_penalty


def test_exact_penalty_matches_autograd_hutchinson():
    """Closed-form penalty vs autograd: build the predictor as a differentiable
    torch function, Hutchinson-estimate E_x ||grad^2 R||_F^2 with many probes
    on a big random batch, and require agreement within 10%."""
    torch.manual_seed(0)
    d = 4
    sr = SpectralReward(in_dim=d, n_features=256, sigma_w=1.0, seed=0)
    sr.c = torch.randn(256, generator=torch.Generator().manual_seed(1))

    exact = sr.hessian_frobenius_sq()
    assert exact > 0

    # spread-out x: the cross terms (the only approximation when averaging
    # over x instead of the phases) decay with the data spread
    X = 2.0 * torch.randn(4096, d, generator=torch.Generator().manual_seed(2))
    gen = torch.Generator().manual_seed(3)
    est = hvp_penalty(sr.predict, X, n_probes=64, generator=gen,
                      create_graph=False).item()

    assert est == pytest.approx(exact, rel=0.10), (exact, est)


def test_fit_recovers_smooth_function_and_penalty_monotone_in_lam():
    torch.manual_seed(0)
    d = 3
    g = torch.Generator().manual_seed(0)
    X = torch.randn(512, d, generator=g)
    y = torch.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2]
    Xte = torch.randn(512, d, generator=g)
    yte = torch.sin(Xte[:, 0]) + 0.5 * Xte[:, 1] ** 2 - Xte[:, 2]

    sr = SpectralReward(in_dim=d, n_features=512, sigma_w=1.0, seed=0)
    sr.fit(X, y, lam=1e-4)
    mse = torch.mean((sr.predict(Xte) - yte) ** 2).item()
    assert mse < 0.05  # smooth target, plenty of data: near-interpolation

    # heavier curvature penalty -> smaller exact penalty value (monotone)
    pens = []
    for lam in (1e-6, 1e-2, 1.0):
        pens.append(SpectralReward(d, 512, 1.0, seed=0).fit(X, y, lam)
                    .hessian_frobenius_sq())
    assert pens[0] > pens[1] > pens[2] > 0


def test_laplacian_equals_frobenius_null_lagrangian():
    """In the RFF basis the (Delta R)^2 and ||H||_F^2 expectations coincide
    exactly (rank-1 feature Hessians) — the null-Lagrangian equivalence."""
    sr = SpectralReward(in_dim=5, n_features=128, seed=7)
    sr.c = torch.randn(128, generator=torch.Generator().manual_seed(7))
    assert sr.laplacian_trace_sq() == pytest.approx(sr.hessian_frobenius_sq(), rel=0)


def test_seed_determinism_and_predict_shape():
    a = SpectralReward(4, 64, seed=3)
    b = SpectralReward(4, 64, seed=3)
    assert torch.equal(a.W, b.W) and torch.equal(a.b, b.b)
    X = torch.randn(10, 4)
    a.fit(X, torch.randn(10), lam=1e-2)
    out = a.predict(X)
    assert out.shape == (10,) and torch.isfinite(out).all()
    # predict stays differentiable in X (needed for the autograd cross-check)
    Xg = X.clone().requires_grad_(True)
    a.predict(Xg).sum().backward()
    assert Xg.grad is not None and torch.isfinite(Xg.grad).all()


def test_sigma_ladder_blocks_and_scalar_equivalence():
    """sigma_w as a list = ladder: block k scaled by sigma[k]; scalar path
    bitwise-matches the pre-ladder construction (scale after draw, same RNG)."""
    ladder = [0.25, 0.5, 1.0, 2.0]
    unit = SpectralReward(4, 64, sigma_w=1.0, seed=3)
    lad = SpectralReward(4, 64, sigma_w=ladder, seed=3)
    assert torch.equal(lad.b, unit.b)                  # same phase stream
    blk = 64 // len(ladder)
    for k, sig in enumerate(ladder):
        sl = slice(k * blk, 64 if k == len(ladder) - 1 else (k + 1) * blk)
        assert torch.allclose(lad.W[sl], sig * unit.W[sl])
    # scalar path unchanged: sigma_w=0.5 == 0.5 * unit draw
    half = SpectralReward(4, 64, sigma_w=0.5, seed=3)
    assert torch.allclose(half.W, 0.5 * unit.W)
    # ladder produces separated bands: |w| spread much wider than single sigma
    assert lad.w2.sqrt().max() / lad.w2.sqrt().min() \
        > 2 * (unit.w2.sqrt().max() / unit.w2.sqrt().min())
    # fit still works end-to-end
    X = torch.randn(128, 4, generator=torch.Generator().manual_seed(0))
    lad.fit(X, X.sum(-1), lam=1e-4)
    assert torch.isfinite(lad.predict(X)).all()


def test_sigma_ladder_validation():
    with pytest.raises(ValueError):
        SpectralReward(4, 64, sigma_w=[], seed=0)
    with pytest.raises(ValueError):
        SpectralReward(4, 2, sigma_w=[1.0, 2.0, 3.0], seed=0)


def test_snr_band_weights_wiener_behavior():
    """Known band structure: low-frequency signal + white target noise. The
    SNR weights must (a) be finite/positive, (b) penalize the signal-free
    high-|w| bands harder than the signal-carrying low bands, (c) yield a fit
    on noisy targets at least as good as the unweighted near-interpolator."""
    from mbrl.models.spectral import snr_band_weights

    g = torch.Generator().manual_seed(0)
    d = 3
    X = torch.randn(4096, d, generator=g)
    y_clean = torch.sin(X[:, 0]) + 0.5 * X[:, 1]          # smooth, low-frequency
    y = y_clean + 1.0 * torch.randn(len(X), generator=g)  # SNR ~ O(1)

    sr = SpectralReward(d, n_features=256, sigma_w=[0.25, 0.5, 1.0, 4.0], seed=0)
    Phi = sr.features(X)
    theta, info = snr_band_weights(Phi, y, sr.w2.sqrt(), n_bands=8,
                                   generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(theta).all() and (theta > 0).all()
    snrs = info["band_snrs"]
    assert len(snrs) >= 4
    # low bands carry the signal: SNR should decrease from the lowest band
    # to the highest (compare first vs last)
    assert snrs[0] > snrs[-1]
    # weights inversely follow SNR: highest band penalized harder than lowest
    w = sr.w2.sqrt()
    lo = theta[w <= info["edges"][1]].mean()
    hi = theta[w >= info["edges"][-2]].mean()
    assert hi > lo

    # Wiener fit beats the (nearly unregularized) tiny-quartic fit on clean targets
    Xte = torch.randn(2048, d, generator=g)
    yte = torch.sin(Xte[:, 0]) + 0.5 * Xte[:, 1]
    fit_snr = SpectralReward(d, 256, [0.25, 0.5, 1.0, 4.0], seed=0).fit(X, y, weights=theta)
    fit_raw = SpectralReward(d, 256, [0.25, 0.5, 1.0, 4.0], seed=0).fit(X, y, lam=1e-9)
    mse_snr = torch.mean((fit_snr.predict(Xte) - yte) ** 2).item()
    mse_raw = torch.mean((fit_raw.predict(Xte) - yte) ** 2).item()
    assert mse_snr < mse_raw


def test_rational_head_recovers_planted_rational():
    """SK iterations: a planted N/D target is fit better by the rational head
    than by a linear head at MATCHED total features; D-guard exposes clamping."""
    from mbrl.models.spectral import RationalSpectralReward

    g = torch.Generator().manual_seed(0)
    d = 3
    X = torch.randn(3000, d, generator=g)
    c0 = torch.tensor([0.6, -0.4, 0.2])
    y = torch.sin(X[:, 0]) / (0.3 + (X - c0).pow(2).sum(-1))   # one resonance
    Xte = torch.randn(1500, d, generator=g)
    yte = torch.sin(Xte[:, 0]) / (0.3 + (Xte - c0).pow(2).sum(-1))

    rr = RationalSpectralReward(d, 128, [0.5, 1.0], seed=0)
    wn = 1e-3 * rr.num.w4
    wd = 1e-3 * rr.den.w4
    rr.fit(X, y, weights_num=wn, weights_den=wd, den_anchor=0.05)
    mse_rat = torch.mean((rr.predict(Xte) - yte) ** 2).item()

    sr = SpectralReward(d, 128, [0.5, 1.0], seed=0).fit(X, y, lam=1e-3)
    mse_lin = torch.mean((sr.predict(Xte) - yte) ** 2).item()

    assert torch.isfinite(rr.predict(Xte)).all()
    assert rr.clamp_rate_ < 0.2          # guard rail mostly inactive
    assert mse_rat < mse_lin             # rational wins on a rational target
    # resonance recovery: 1/|D| larger at the planted center than randomly
    score_c = rr.resonance_score(c0.unsqueeze(0)).item()
    score_r = rr.resonance_score(Xte[:256]).mean().item()
    assert score_c > score_r


def test_orf_preserves_norms_and_orthogonalizes():
    """ORF: row norms preserved exactly (ladder/bands untouched); within-chunk
    directions orthogonal; predictions still finite and fit works."""
    from mbrl.models.spectral import orthogonalize_features

    sr = SpectralReward(4, 64, [0.25, 0.5, 1.0, 2.0], seed=0)
    norms_before = sr.w2.sqrt().clone()
    sr = orthogonalize_features(sr)
    assert torch.allclose(sr.w2.sqrt(), norms_before, atol=1e-5)
    D = sr.W[:4] / sr.W[:4].norm(dim=-1, keepdim=True)   # first chunk dirs
    off = (D @ D.T - torch.eye(4)).abs().max()
    assert off < 1e-5
    X = torch.randn(256, 4, generator=torch.Generator().manual_seed(1))
    sr.fit(X, X.sum(-1), lam=1e-3)
    assert torch.isfinite(sr.predict(X)).all()


def test_shrink_coefs_rings_in_correlated_basis():
    """DOCUMENTED FAILURE (run 12 candidate B, dropped before running): DJ
    universal-threshold shrinkage requires an ORTHONORMAL basis. In the
    correlated RFF design, cancellation pairs are everywhere — zeroing one
    side leaves the partner ringing. This test pins the failure mode so the
    candidate isn't re-proposed naively; the correct form (shrinkage in the
    Phi-SVD basis = adaptive TSVD, Rosasco spectral filtering) is cycle-2."""
    from mbrl.models.spectral import shrink_coefs

    g = torch.Generator().manual_seed(0)
    X = torch.randn(2048, 3, generator=g)
    y_clean = torch.sin(X[:, 0])
    y = y_clean + 1.0 * torch.randn(len(X), generator=g)
    sr = SpectralReward(3, 256, [0.5, 1.0], seed=0)
    w = 1e-3 * sr.w4
    sr.fit(X, y, weights=w)
    mse_before = torch.mean((sr.predict(X) - y_clean) ** 2).item()
    sr = shrink_coefs(sr, X, y, w)
    nz = int((sr.c.abs() > 1e-10).sum())
    mse_after = torch.mean((sr.predict(X) - y_clean) ** 2).item()
    assert nz < 256                       # it does shrink...
    assert mse_after > mse_before * 2     # ...and RINGS: the documented failure


def test_svd_shrink_does_not_ring_and_reduces_to_ridge():
    """Run-12B (cycle 2): DJ shrinkage in the orthonormal Phi-SVD basis is the
    corrected candidate B (adaptive TSVD / Rosasco spectral filtering). SAME
    correlated-RFF problem as the ringing test above. Two guarantees:
    (a) kappa=0 reproduces the scalar Tikhonov ridge (Phi'Phi + lam I)^-1 Phi'y
    exactly (up to float32 SVD-vs-solve roundoff) — the arm nests a clean
    spectral-filter baseline; (b) at the DJ universal threshold it does NOT
    ring — U is orthonormal by construction, so the cancellation-pair failure
    that destroys shrink_coefs cannot occur, and the Tikhonov filter s/(s²+lam)
    bounds small-singular-value amplification."""
    from mbrl.models.spectral import svd_shrink_fit

    g = torch.Generator().manual_seed(0)
    X = torch.randn(2048, 3, generator=g)
    y_clean = torch.sin(X[:, 0])
    y = y_clean + 1.0 * torch.randn(len(X), generator=g)
    sr = SpectralReward(3, 256, [0.5, 1.0], seed=0)
    Phi = sr.features(X)

    # (a) kappa=0 == scalar Tikhonov ridge
    lam = 1.0
    c_ridge = torch.linalg.solve(Phi.T @ Phi + lam * torch.eye(256), Phi.T @ y)
    sr0 = SpectralReward(3, 256, [0.5, 1.0], seed=0)
    svd_shrink_fit(sr0, X, y, lam=lam, kappa=0.0)
    assert (sr0.c - c_ridge).norm() / c_ridge.norm() < 1e-2

    # (b) DJ universal threshold in the SVD basis: bounded, no ringing
    sr.fit(X, y, weights=1e-3 * sr.w4)
    mse_before = torch.mean((sr.predict(X) - y_clean) ** 2).item()
    svd_shrink_fit(sr, X, y, lam=1.0, kappa=1.0)
    mse_after = torch.mean((sr.predict(X) - y_clean) ** 2).item()
    assert mse_after < mse_before * 2     # contrast: shrink_coefs blows up >1000x
