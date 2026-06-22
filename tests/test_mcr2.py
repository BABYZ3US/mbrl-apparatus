"""MCR² expansion term as an unsupervised anti-collapse regularizer (Yu et al. 2020).

Pins for `mbrl.regularization.mcr2`:
  • coding_rate is a volume readout: ~0 for a near-collapsed batch (all rows ≈ equal),
    clearly LARGER for an isotropic random batch — it grows with the representation's spread.
  • mcr2_loss == −coding_rate exactly; gradient flows to Z (backward populates Z.grad);
    and more spread ⇒ SMALLER loss (loss(isotropic) < loss(collapsed)), so MINIMIZING the
    loss MAXIMIZES the rate (the intended anti-collapse direction).
  • logdet identity: forming cov = ZᵀZ/N then logdet(I + (d/eps²)·cov) reproduces the
    displayed ½·logdet(I + (d/(N·eps²))·ZᵀZ) to float precision.
  • shape/guards: returns a scalar; N<2 is a differentiable 0.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from mbrl.regularization.mcr2 import coding_rate, mcr2_loss


def _collapsed(n=256, d=8, seed=0, scale=1e-4):
    """Near-collapsed batch: every row ≈ the same point, with tiny isotropic noise."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, d, generator=g)
    return base.expand(n, d).clone() + scale * torch.randn(n, d, generator=g)


def _isotropic(n=256, d=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


# ---- coding_rate: a volume / spread readout ----
def test_coding_rate_collapsed_near_zero_isotropic_larger():
    """A near-collapsed Z packs into ~0 nats; an isotropic Z[256,8] fills the coding ball
    and packs into clearly more — the rate measures spread."""
    r_collapsed = coding_rate(_collapsed()).item()
    r_isotropic = coding_rate(_isotropic()).item()
    assert math.isfinite(r_collapsed) and math.isfinite(r_isotropic)
    assert r_collapsed < 1e-2                       # collapsed -> near 0
    assert r_isotropic > r_collapsed                # the required inequality
    assert r_isotropic > 1.0                        # isotropic 8-D spread is clearly large


def test_coding_rate_monotone_in_scale():
    """Scaling an isotropic batch up spreads it further (bigger eigenvalues inside the
    logdet) -> a strictly larger coding rate."""
    Z = _isotropic()
    assert coding_rate(2.0 * Z).item() > coding_rate(Z).item() > coding_rate(0.25 * Z).item()


def test_coding_rate_returns_scalar():
    r = coding_rate(_isotropic())
    assert r.dim() == 0 and r.shape == torch.Size([])


def test_logdet_identity_matches_displayed_formula():
    """coding_rate (cov = ZᵀZ/N path) equals the displayed ½·logdet(I + (d/(N·eps²))·ZᵀZ)
    computed directly on the centered batch — same matrix, same logdet, to float precision."""
    g = torch.Generator().manual_seed(2)
    N, d, eps = 200, 6, 0.5
    Z = torch.randn(N, d, generator=g)
    Zc = Z - Z.mean(0, keepdim=True)
    M = torch.eye(d) + (d / (N * eps * eps)) * (Zc.T @ Zc)   # the displayed I + (d/(N eps²)) ZᵀZ
    ref = 0.5 * torch.logdet(M)
    assert coding_rate(Z, eps=eps).item() == pytest.approx(ref.item(), rel=1e-5, abs=1e-5)


def test_guard_small_n_is_zero():
    """N<2 has no spread to measure -> a differentiable scalar 0 (tied to Z)."""
    z1 = torch.randn(1, 8, requires_grad=True)
    r = coding_rate(z1)
    assert r.item() == 0.0 and r.shape == torch.Size([])
    r.backward()                                    # stays differentiable (zero grad), no error
    assert z1.grad is not None and torch.allclose(z1.grad, torch.zeros_like(z1))


# ---- mcr2_loss: the term to add to a minimized loss ----
def test_mcr2_loss_is_negative_coding_rate():
    Z = _isotropic()
    assert mcr2_loss(Z).item() == pytest.approx(-coding_rate(Z).item(), rel=0, abs=1e-6)


def test_mcr2_loss_gradient_flows():
    """backward on the scalar mcr2_loss populates Z.grad with finite, non-zero gradient —
    the regularizer trains the representation."""
    Z = _isotropic().clone().requires_grad_(True)
    loss = mcr2_loss(Z)
    assert loss.dim() == 0
    loss.backward()
    assert Z.grad is not None and torch.isfinite(Z.grad).all()
    assert Z.grad.abs().sum().item() > 0.0          # non-trivial gradient


def test_minimizing_loss_maximizes_spread():
    """More spread DEcreases the loss: loss(isotropic) < loss(collapsed). So a minimizer
    is pushed toward the more-spread (anti-collapse) representation."""
    assert mcr2_loss(_isotropic()).item() < mcr2_loss(_collapsed()).item()
    # equivalently, the gradient at a collapsed Z points toward expanding it: stepping
    # AGAINST the loss gradient raises the coding rate
    Z = _collapsed(scale=1e-2).clone().requires_grad_(True)
    r0 = coding_rate(Z).item()
    mcr2_loss(Z).backward()
    with torch.no_grad():
        Z_step = Z - 0.5 * Z.grad                   # descend the loss = ascend the rate
    assert coding_rate(Z_step).item() > r0
