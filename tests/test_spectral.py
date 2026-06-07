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
