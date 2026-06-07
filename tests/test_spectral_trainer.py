"""Trainer-integrated spectral reward path (spectral.enabled=true).

Covers: (1) full spectral Trainer smoke (cache -> refit -> behaviour update),
(2) the data-driven symexp clamp on imagined spectral rewards, (3) poly_weights
hand values + per-degree schedule shifts, (4) bitwise save -> resume under a
spectral config, (5) policy gradients flow through the closed-form predict
(cos features are differentiable in (z, a)).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mbrl.models.spectral import SpectralReward, poly_weights
from mbrl.regularization.schedule import LambdaSchedule
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager


def make_cfg(poly=None, **spec_overrides):
    spec = {"enabled": True, "n_features": 64, "sigma_w": 1.0,
            "refit_every": 5, "cache_size": 256, "heads": 2,
            "poly": poly or {"degrees": [2], "coefs": [1.0], "shifts": [0]}}
    spec.update(spec_overrides)
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "symlog_reward": True, "reward_heads": 2},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3,
                                 "t0": 100, "floor": 0.0}},
        "spectral": spec,
        "smoothing": {"enabled": True, "sigma": 1.5},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    })


def fake_batch(n=32, obs_dim=3, act_dim=1, seed=123, r_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            r_scale * torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_spectral_trainer_smoke():
    """Enough model updates to trigger a refit (batch 32, n_features 64 ->
    refit at update 2), then a behaviour update; everything finite, the exact
    spectral penalty positive after the refit."""
    torch.manual_seed(0)
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    metrics = []
    for i in range(8):
        metrics.append(t.model_update(fake_batch(seed=100 + i)))
    # before the first refit the heads predict zeros — logged, not fatal
    assert metrics[0]["spectral/fitted"] == 0.0
    assert metrics[-1]["spectral/refits"] >= 1
    assert metrics[-1]["spectral/cache_n"] == 8 * 32
    for m in metrics:
        for k, v in m.items():
            assert np.isfinite(v), (k, v)
    assert metrics[-1]["penalty/value"] > 0  # exact mean-over-heads H^2

    z0 = t.encoder(fake_batch(seed=999)[0]).detach()
    b = t.behaviour_update(z0)
    for k, v in b.items():
        assert np.isfinite(v), (k, v)


def test_imagined_rewards_respect_symexp_clamp():
    """Absurd targets / absurd coefficients: imagined spectral rewards must
    stay inside symexp(margin * symlog_bound)."""
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    # absurd real targets drive the data-driven bound up but keep it finite
    for i in range(3):
        m = t.model_update(fake_batch(seed=200 + i, r_scale=1e8))
    assert np.isfinite(t.symlog_bound) and t.symlog_bound > 1.0
    assert np.isfinite(m["loss/total"])
    # blow up the head coefficients: predictions are absurd pre-clamp
    for h in t.spec_heads:
        h.c = 1e6 * torch.randn(h.M, generator=torch.Generator().manual_seed(7))
    z = torch.randn(64, t.encoder(torch.zeros(1, 3)).shape[-1])
    a = torch.randn(64, 1)
    r_im, dis = t._imagined_reward(z, a)
    bound = math.expm1(t.symexp_margin * t.symlog_bound)
    assert torch.isfinite(r_im).all()
    # mean - pessimism*std of per-head clamped values cannot exceed the
    # per-head bound (pessimism=0 here -> plain mean)
    assert r_im.abs().max().item() <= bound * (1 + 1e-5)  # float32 slack at the bound
    assert np.isfinite(dis.item())


def test_poly_weights_hand_values_and_shifts():
    # pure quartic: |w| in {1, 2} -> |w|^4 in {1, 16}
    w = torch.tensor([1.0, 2.0])
    assert torch.allclose(poly_weights(w, [2], [1.0]), torch.tensor([1.0, 16.0]))
    # mixed degrees: 2|w|^2 + 3|w|^4 -> at |w|=1: 5; at |w|=2: 8 + 48 = 56
    assert torch.allclose(poly_weights(w, [1, 2], [2.0, 3.0]),
                          torch.tensor([5.0, 56.0]))
    with pytest.raises(ValueError):
        poly_weights(w, [1, 2], [1.0])

    # per-degree shifts phase-shift the lambda schedule per band
    poly = {"degrees": [1, 2], "coefs": [0.5, 1.0], "shifts": [0, 50]}
    t = Trainer(make_cfg(poly=poly), obs_dim=3, action_dim=1)
    head = t.spec_heads[0]
    sched = LambdaSchedule(kind="cuberoot", lam0=1e-3, t0=100, floor=0.0)
    for tt in (0, 200):
        expect = (0.5 * sched(tt) * head.w2
                  + 1.0 * sched(tt + 50) * head.w2.pow(2))
        got = t._spectral_band_weights(head, tt)
        assert torch.allclose(got, expect, rtol=1e-6), tt
    # weights move over t, and the shift changes them at fixed t
    w0, w200 = t._spectral_band_weights(head, 0), t._spectral_band_weights(head, 200)
    assert not torch.allclose(w0, w200)
    t_noshift = Trainer(make_cfg(poly={"degrees": [1, 2], "coefs": [0.5, 1.0],
                                       "shifts": [0, 0]}), obs_dim=3, action_dim=1)
    assert not torch.allclose(w0, t_noshift._spectral_band_weights(
        t_noshift.spec_heads[0], 0))


def test_spectral_resume_bitwise_identical(tmp_path):
    """save -> resume -> identical next model_update losses (spectral cfg)."""
    cfg = make_cfg()
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    for i in range(3):  # refit fires at update 2 -> heads carry fitted c
        t1.model_update(fake_batch(seed=300 + i))
    assert t1.spec_refits >= 1
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    m_ref = t1.model_update(fake_batch(seed=303))

    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    assert t2.spec_refits == t1.spec_refits  # m_ref's update didn't refit
    m_resumed = t2.model_update(fake_batch(seed=303))

    for k in ("loss/dyn", "loss/reward", "loss/total", "penalty/value"):
        assert m_resumed[k] == pytest.approx(m_ref[k], rel=1e-6), k


def test_policy_gradients_flow_through_spectral_reward():
    torch.manual_seed(0)
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    for i in range(4):  # ensure a refit happened (c != 0)
        t.model_update(fake_batch(seed=400 + i))
    assert t.spec_refits >= 1

    # direct: predict builds a torch graph back to z and a
    k = t.spec_heads[0].in_dim - 1  # latent dim (action_dim=1, task_dim=0)
    z = torch.randn(16, k, requires_grad=True)
    a = torch.randn(16, 1, requires_grad=True)
    r_im, _ = t._imagined_reward(z, a)
    assert r_im.grad_fn is not None
    r_im.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert a.grad is not None and a.grad.abs().sum() > 0

    # end to end: behaviour_update leaves nonzero policy gradients
    z0 = t.encoder(fake_batch(seed=555)[0]).detach()
    b = t.behaviour_update(z0)
    assert np.isfinite(b["loss/policy"])
    gsum = sum(p.grad.abs().sum().item() for p in t.policy.parameters()
               if p.grad is not None)
    assert gsum > 0


def test_sigma_ladder_trainer_smoke():
    """sigma_w as an OmegaConf list (the spectral_ladder preset's run-3 recipe)
    flows through Trainer construction, refit, and stays finite."""
    torch.manual_seed(0)
    cfg = make_cfg(poly={"degrees": [1, 3], "coefs": [0.1, 10.0], "shifts": [0, 0]},
                   sigma_w=[0.25, 0.5, 1.0, 2.0])
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    for h in t.spec_heads:  # ladder applied: bands genuinely separated
        assert (h.w2.sqrt().max() / h.w2.sqrt().min()) > 4.0
    for i in range(6):
        m = t.model_update(fake_batch(seed=200 + i))
        assert np.isfinite(m["loss/total"])
    assert t.spec_refits >= 1
    assert t._spectral_penalty_value() >= 0.0
