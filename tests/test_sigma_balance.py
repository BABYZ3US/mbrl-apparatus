"""sigma-scaling / entropy-balance setpoint + the compress Tikhonov ridge.

Pins: sigma_balance_penalty drives sigma=sqrt(<lambda>) toward the target (0 at the
target, grows away from it, grad flows); the compress penalty's `eps` ridge is a real
knob (larger eps -> larger floor value). Mirrors tests/ style (no fixtures)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.regularization.rank2_frame import sigma_balance_penalty, spectral_compress_penalty


def test_sigma_balance_hits_zero_at_target():
    torch.manual_seed(0)
    # isotropic std~0.8 -> sigma = sqrt(<lambda>) ~ 0.8 -> penalty ~ 0
    z = 0.8 * torch.randn(4096, 8)
    p = sigma_balance_penalty(z, sigma_target=0.8)
    assert p.item() < 1e-2, p.item()


def test_sigma_balance_grows_away_and_grad_flows():
    torch.manual_seed(1)
    # near-collapsed z -> sigma ~ 0 -> penalty ~ sigma_target^2 = 0.64
    zc = 1e-3 * torch.randn(2048, 8)
    assert sigma_balance_penalty(zc, 0.8).item() > 0.5

    z = torch.randn(256, 6, requires_grad=True)
    sigma_balance_penalty(z, 0.8).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_compress_eps_is_a_real_ridge():
    torch.manual_seed(2)
    z = 0.01 * torch.randn(64, 4)               # near-singular Gram -> floor value dominated by eps
    c_small = spectral_compress_penalty(z, floor=0.0, eps=1e-3)
    c_big = spectral_compress_penalty(z, floor=0.0, eps=1e-1)
    assert c_big.item() > c_small.item()        # larger ridge -> larger sqrt(eps) floor
    # default arg still works (legacy call signature)
    assert torch.isfinite(spectral_compress_penalty(z, 0.0)).all()
