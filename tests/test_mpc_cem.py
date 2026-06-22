"""CEMPlanner (CEM-MPC over the learned latent model). Pins: act() shapes for a
single z0 [k] -> [A] and a batch [B, k] -> [B, A]; actions stay in [-1, 1]; and
determinism given a `generator` (same seed -> same action, no global RNG — the
resume-bitwise discipline). Trivial linear model (z' = z + 0.1*a, r = -||z||^2,
i.e. drive z -> 0) so the optimum is unambiguous and torch-only/fast (CPU)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.planning.mpc import CEMPlanner

K = 2          # latent dim (= action dim so the toy z' = z + 0.1*a is well-formed)
A = 2          # action dim


def _step_fn(z, a):
    return z + 0.1 * a                       # trivial linear latent dynamics


def _reward_fn(z, a, tau=None):
    return -(z ** 2).sum(-1)                  # drive z -> 0 (higher = better)


def _planner():
    return CEMPlanner(action_dim=A, horizon=5, pop=64, iters=3, elites=8)


def test_act_single_shape_and_bounds():
    pl = _planner()
    g = torch.Generator().manual_seed(0)
    z0 = torch.tensor([1.0, -0.5])      # [k]
    a = pl.act(z0, _step_fn, _reward_fn, generator=g)
    assert a.shape == (A,)                     # first action of the plan
    assert a.abs().max() <= 1.0 + 1e-6         # respects [-1, 1]


def test_act_batched_shape_and_bounds():
    pl = _planner()
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(4, K, generator=torch.Generator().manual_seed(7))   # [B, k]
    a = pl.act(z0, _step_fn, _reward_fn, generator=g)
    assert a.shape == (4, A)                    # [B, A], one CEM per row
    assert a.abs().max() <= 1.0 + 1e-6


def test_deterministic_given_generator():
    pl = _planner()
    z0 = torch.tensor([0.7, 0.3])
    a1 = pl.act(z0, _step_fn, _reward_fn, generator=torch.Generator().manual_seed(123))
    a2 = pl.act(z0, _step_fn, _reward_fn, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a1, a2)                  # same seed -> identical action

    # batched determinism too
    zb = torch.randn(3, K, generator=torch.Generator().manual_seed(1))
    b1 = pl.act(zb, _step_fn, _reward_fn, generator=torch.Generator().manual_seed(5))
    b2 = pl.act(zb, _step_fn, _reward_fn, generator=torch.Generator().manual_seed(5))
    assert torch.equal(b1, b2)


def test_value_fn_bootstrap_runs():
    """Optional terminal value adds γ^H V(z_H) without changing the act() shape."""
    pl = _planner()
    g = torch.Generator().manual_seed(0)
    z0 = torch.tensor([1.0, -0.5])
    a = pl.act(z0, _step_fn, _reward_fn,
               value_fn=lambda z, tau=None: -(z ** 2).sum(-1), generator=g)
    assert a.shape == (A,)
    assert a.abs().max() <= 1.0 + 1e-6


def test_drives_latent_toward_zero():
    """Sanity: from a positive z0, the chosen first action is negative on each
    dim (the only way r = -||z||^2 improves under z' = z + 0.1*a)."""
    pl = _planner()
    g = torch.Generator().manual_seed(0)
    z0 = torch.tensor([0.9, 0.9])
    a = pl.act(z0, _step_fn, _reward_fn, generator=g)
    assert (a < 0).all()                        # pushes z back toward the origin
