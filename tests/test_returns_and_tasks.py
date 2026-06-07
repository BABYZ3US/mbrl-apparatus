import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

from mbrl.training.returns import lambda_returns


def test_lambda_returns_hand_computed():
    """H=2, B=1, hand-checked recursion."""
    r = torch.tensor([[1.0], [2.0]])           # r0, r1
    v = torch.tensor([[10.0], [20.0], [30.0]])  # v0, v1, v2
    g, lam = 0.9, 0.5
    # R1 = r1 + g*((1-lam)*v2 + lam*v2) = 2 + 0.9*30 = 29
    # R0 = r0 + g*((1-lam)*v1 + lam*R1) = 1 + 0.9*(0.5*20 + 0.5*29) = 23.05
    out = lambda_returns(r, v, g, lam)
    assert out[1, 0] == pytest.approx(29.0)
    assert out[0, 0] == pytest.approx(23.05)


def test_lambda_returns_limits():
    torch.manual_seed(0)
    H, B = 10, 4
    r, v = torch.randn(H, B), torch.randn(H + 1, B)
    g = 0.99
    # lam=0 -> one-step TD targets
    td = r + g * v[1:]
    assert torch.allclose(lambda_returns(r, v, g, 0.0), td, atol=1e-6)
    # lam=1 -> Monte Carlo with terminal bootstrap
    mc = torch.empty_like(r)
    acc = v[-1]
    for t in reversed(range(H)):
        acc = r[t] + g * acc
        mc[t] = acc
    assert torch.allclose(lambda_returns(r, v, g, 1.0), mc, atol=1e-5)


def test_policy_tanh_logprob_finite_and_bounded():
    from mbrl.models import Policy
    torch.manual_seed(0)
    pi = Policy(4, 2, hidden=32, depth=1, action_scale=2.0)
    a, logp = pi.sample(torch.randn(256, 4))
    assert a.abs().max() <= 2.0 + 1e-5
    assert torch.isfinite(logp).all()


def test_pendulum_target_family():
    from mbrl.envs.tasks import make_task_env, task_split
    split = task_split("pendulum_target", n_train=8)
    assert len(split["train"]) == 8 and len(split["extrap"]) == 2
    # no overlap between train and held-out
    assert not (set(split["train"]) & set(split["interp"]))

    env = make_task_env("pendulum_target", tau=0.5)
    obs, _ = env.reset(seed=0)
    obs, r, *_ = env.step(np.array([0.1], dtype=np.float32))
    assert np.isfinite(r) and r <= 0  # negative quadratic cost
    assert env.tau.shape == (1,) and env.task_dim == 1
    env.close()


def test_multitask_trainer_smoke():
    """Task-conditioned Trainer: one model + behaviour update, finite metrics."""
    from omegaconf import OmegaConf
    from mbrl.training import Trainer
    cfg = OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "include_task": True,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100,
                                 "floor": 0.0}},
        "smoothing": {"enabled": True, "sigma": 1.5},
        "imagination": {"horizon": 5, "gamma": 0.99, "lambda_": 0.95,
                        "entropy_coef": 3e-4, "value_target_decay": 0.98},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    })
    t = Trainer(cfg, obs_dim=3, action_dim=1, task_dim=1)
    g = torch.Generator().manual_seed(1)
    batch = (torch.randn(32, 3, generator=g), torch.randn(32, 1, generator=g),
             torch.randn(32, generator=g), torch.randn(32, 3, generator=g),
             torch.rand(32, 1, generator=g))
    m = t.model_update(batch)
    b = t.behaviour_update(torch.randn(32, 4, generator=g),
                           torch.rand(32, 1, generator=g))
    import math
    assert all(math.isfinite(v) for v in (*m.values(), *b.values()))
    # include_task=False path also runs
    cfg.penalty.include_task = False
    t2 = Trainer(cfg, obs_dim=3, action_dim=1, task_dim=1)
    m2 = t2.model_update(batch)
    assert math.isfinite(m2["loss/total"])
