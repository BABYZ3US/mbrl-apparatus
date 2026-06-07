import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch

from mbrl.models import AffineDynamics
from mbrl.regularization.schedule import LambdaSchedule
from mbrl.regularization.transversality import transversality_angle


def test_affine_dynamics_zero_action_curvature():
    """d^2 T / da^2 = 0 exactly (R15) — second difference in a vanishes."""
    torch.manual_seed(0)
    dyn = AffineDynamics(latent_dim=4, action_dim=2, hidden=32, depth=1)
    z = torch.randn(8, 4)
    a, da = torch.randn(8, 2), torch.randn(8, 2)
    second_diff = dyn(z, a + da) - 2 * dyn(z, a) + dyn(z, a - da)
    assert second_diff.abs().max().item() < 1e-5


def test_schedule_cuberoot_profile():
    s = LambdaSchedule(kind="cuberoot", lam0=1e-3, t0=1000, floor=1e-6)
    assert s(0) == pytest.approx(1e-3)
    assert s(1000) == pytest.approx(1e-3 * 0.5 ** (1 / 3))
    assert s(10 ** 15) == pytest.approx(1e-6)  # floor reached for huge t
    assert all(s(t) >= s(t + 1) for t in range(0, 5000, 100))  # monotone


def test_schedule_step():
    s = LambdaSchedule(kind="step", lam0=1.0, step_at=0.5, step_factor=0.1,
                       total_steps=100)
    assert s(49) == 1.0 and s(50) == pytest.approx(0.1)


def test_effective_dim_known_spectra():
    from mbrl.regularization.transversality import effective_dim
    torch.manual_seed(0)
    x = torch.randn(256, 6)
    g = torch.Generator().manual_seed(0)
    # uniform spectrum over all 6 dims: f = ||x||^2, H = 2I -> d_eff = 6
    de = effective_dim(lambda x: x.pow(2).sum(-1), x, n_probes=64, generator=g)
    assert abs(de - 6) < 0.8
    # rank-2 spectrum: H = diag(2,2,0,0,0,0) -> d_eff = 2
    de2 = effective_dim(lambda x: x[..., :2].pow(2).sum(-1), x, n_probes=64,
                        generator=g)
    assert abs(de2 - 2) < 0.5
    # spiked: one dominant direction -> d_eff ~ 1
    de1 = effective_dim(lambda x: 10 * x[..., 0].pow(2) + 0.01 * x[..., 1].pow(2),
                        x, n_probes=64, generator=g)
    assert de1 < 1.3


def test_effective_dim_heterogeneous_curvature():
    """Per-sample PR=2 everywhere but per-sample SCALE varies wildly: the
    estimator must still report ~2, not collapse below 1 (pooled-trace bug)."""
    from mbrl.regularization.transversality import effective_dim
    torch.manual_seed(0)
    x = torch.randn(256, 6)
    g = torch.Generator().manual_seed(0)
    # scale factor exp(2*x3) varies ~e^{±4} across samples (kept fp32-safe since
    # tr(H^4) ~ scale^4); rank-2 quadratic in (x0, x1) with sample-dependent scale
    f = lambda x: torch.exp(2 * x[..., 3]).detach() * (x[..., 0] ** 2 + x[..., 1] ** 2)
    de = effective_dim(f, x, n_probes=64, generator=g)
    assert 1.5 < de < 2.6, de
    assert de >= 1.0 - 1e-6  # PR can never be < 1


def test_transversality_angle_known_cases():
    torch.manual_seed(0)
    x = torch.randn(128, 4)
    f = lambda x: x.pow(2).sum(-1)            # H = 2I
    g_same = lambda x: 3 * x.pow(2).sum(-1)   # H = 6I, parallel -> 0 deg
    assert transversality_angle(f, g_same, x, n_probes=16) < 10
    # orthogonal Hessians: diag(1,1,0,0) vs diag(0,0,1,1) -> 90 deg
    h1 = lambda x: x[..., :2].pow(2).sum(-1)
    h2 = lambda x: x[..., 2:].pow(2).sum(-1)
    assert abs(transversality_angle(h1, h2, x, n_probes=16) - 90) < 10
