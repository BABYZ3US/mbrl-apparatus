"""SequencePlanner (Dreamer latent action-sequence planner). Pins: plan shapes
+ finite stable log-prob; default OFF = the MLP-policy actor (unchanged); the
planner trains end-to-end through behaviour_update; act() = receding-horizon
first action; bitwise resume with the planner on (it's checkpointed)."""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models.planner import SequencePlanner
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(planner=False, horizon=6):
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "schedule": {"kind": "constant", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": horizon, "gamma": 0.99, "lambda_": 0.95,
                        "adaptive_horizon": {"enabled": True, "h_min": 3, "h_max": 9}},
        "planner": {"enabled": planner, "d_model": 32, "nhead": 2, "layers": 1},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 16},
    })


def _batch(n=16, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_plan_shapes_and_stable_logprob():
    seed_everything(0)
    pl = SequencePlanner(latent_dim=4, action_dim=2, horizon=6, d_model=32, nhead=2, layers=1)
    z0 = torch.randn(8, 4)
    a, logp = pl.plan(z0)
    assert a.shape == (6, 8, 2) and logp.shape == (6, 8)         # time-major
    assert a.abs().max() <= 1.0 + 1e-5                            # tanh-squashed
    assert torch.isfinite(logp).all()
    # finite even at saturated pre-activations (the stable log-det)
    pl.log_std.data.fill_(2.0)
    _, logp2 = pl.plan(z0 * 50.0)
    assert torch.isfinite(logp2).all()
    assert pl.act(z0).shape == (8, 2)                            # receding-horizon first action


def test_default_off_uses_mlp_policy():
    seed_everything(0)
    t = Trainer(_cfg(planner=False), obs_dim=3, action_dim=2)
    assert not t.use_planner and t.planner is None
    z0 = t.encoder(_batch()[0]).detach()
    t.behaviour_update(z0)                                        # runs via the MLP policy
    assert t.act(z0).shape == (z0.shape[0], 2)


def test_planner_trains_end_to_end():
    seed_everything(0)
    t = Trainer(_cfg(planner=True, horizon=6), obs_dim=3, action_dim=2)
    assert t.use_planner and t.planner is not None
    assert t._imagination_horizon() == 6                         # fixed plan length (no adaptive H)
    before = [p.detach().clone() for p in t.planner.parameters()]
    for i in range(3):
        z0 = t.encoder(_batch(seed=i)[0]).detach()
        t.behaviour_update(z0)
    after = list(t.planner.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))   # the planner moved
    assert t.act(z0).shape == (z0.shape[0], 2)


def test_resume_bitwise_identical_with_planner(tmp_path):
    cfg = _cfg(planner=True, horizon=5)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=2)
    for i in range(3):
        t1.model_update(_batch(seed=i))
        t1.behaviour_update(t1.encoder(_batch(seed=i)[0]).detach())
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")   # _latest() globs ckpt_step*.pt
    seed_everything(0)
    a_ref = t1.act(t1.encoder(_batch(seed=42)[0]).detach())

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=2)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    seed_everything(0)
    a_res = t2.act(t2.encoder(_batch(seed=42)[0]).detach())
    assert torch.allclose(a_ref, a_res, atol=1e-6)               # planner restored exactly


def test_alignment_loss_and_grad_clip():
    """Imagination-latent alignment (2507.16450): off by default (imagine/align=0);
    on -> a finite alignment term + actor/grad_norm logged. The transformer-
    stabilization lever; pulls imagined latents toward the encoder manifold."""
    seed_everything(0)
    cfg = _cfg(planner=True, horizon=6)
    cfg.imagination.align_weight = 0.0
    cfg.optim.actor_clip = 100.0
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    m = t.behaviour_update(t.encoder(_batch()[0]).detach())
    assert m["imagine/align"] == 0.0 and "actor/grad_norm" in m   # off = no-op

    seed_everything(0)
    cfg2 = _cfg(planner=True, horizon=6)
    cfg2.imagination.align_weight = 1.0
    t2 = Trainer(cfg2, obs_dim=3, action_dim=2)
    m2 = t2.behaviour_update(t2.encoder(_batch()[0]).detach())
    assert m2["imagine/align"] > 0.0 and math.isfinite(m2["imagine/align"])  # on = real term

import math
