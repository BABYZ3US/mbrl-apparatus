import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mbrl.regularization.schedule import LambdaSchedule
from mbrl.training import Trainer

CFG = OmegaConf.create({
    "seed": 0,
    "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
    "penalty": {"n_probes": 2, "penalize_dynamics": False,
                "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 0.0}},
    "smoothing": {"enabled": True, "sigma": 1.5},
    "imagination": {"horizon": 5, "gamma": 0.99, "lambda_": 0.95,
                    "entropy_coef": 3e-4, "value_target_decay": 0.98,
                    "ret_scale_decay": 0.99},
    "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
})


# ---------------- sin2chirp schedule ----------------
def test_sin2chirp_properties():
    T = 100_000
    s = LambdaSchedule(kind="sin2chirp", lam0=1e-2, t0=20_000, floor=1e-6,
                       period0=20_000, period_end=2_000, total_steps=T)
    ts = np.arange(0, T, 10)
    vals = np.array([s(int(t)) for t in ts])
    # strictly positive (floor) and bounded by the envelope
    assert (vals >= 1e-6 - 1e-12).all()
    env = 1e-2 * (20_000 / (20_000 + ts)) ** (1 / 3)
    assert (vals <= env + 1e-9).all()
    # amplitude decays: max over early window > max over late window
    early = vals[ts < 20_000].max()
    late = vals[ts > 80_000].max()
    assert late < early
    # frequency increases: more zero-ish crossings (near-floor dips) late
    def dips(lo, hi):
        w = vals[(ts >= lo) & (ts < hi)]
        return int(((w[1:] < 0.05 * w.max()) & (w[:-1] >= 0.05 * w.max())).sum())
    assert dips(60_000, 100_000) > dips(0, 40_000)


def test_sin2chirp_no_total_steps_constant_period():
    s = LambdaSchedule(kind="sin2chirp", lam0=1.0, t0=1e12, floor=0.0,
                       period0=100)  # huge t0 => flat envelope
    # sin^2 with period0=100: zero at t=0, max near t=25, zero near t=50
    assert s(0) == pytest.approx(0.0, abs=1e-9)
    assert s(25) == pytest.approx(1.0, rel=1e-3)
    assert s(50) == pytest.approx(0.0, abs=1e-6)


# ---------------- return normalization ----------------
def test_return_normalization_bounds_policy_gradient():
    """Same latent state, returns scaled 1000x => policy grad must NOT scale 1000x."""
    torch.manual_seed(0)

    def grad_norm_with_reward_scale(scale):
        cfg = OmegaConf.create(OmegaConf.to_container(CFG))
        t = Trainer(cfg, obs_dim=3, action_dim=1)
        with torch.no_grad():  # blow up the reward head output
            t.reward.net[-1].weight.mul_(scale)
            t.reward.net[-1].bias.add_(scale)
        for _ in range(5):  # let ret_scale EMA adapt
            t.behaviour_update(torch.randn(64, 4, generator=torch.Generator().manual_seed(7)))
        t.behaviour_update(torch.randn(64, 4, generator=torch.Generator().manual_seed(8)))
        return sum(p.grad.norm().item() for p in t.policy.parameters()
                   if p.grad is not None), t.ret_scale

    g1, s1 = grad_norm_with_reward_scale(1.0)
    g1000, s1000 = grad_norm_with_reward_scale(1000.0)
    assert s1000 > s1  # the scale tracker noticed
    assert g1000 < 100 * g1  # gradient grows far less than the 1000x reward scale


def test_ret_scale_checkpointed(tmp_path):
    from mbrl.utils.checkpoint import CheckpointManager
    t = Trainer(CFG, obs_dim=3, action_dim=1)
    t.behaviour_update(torch.randn(32, 4))
    assert t.ret_scale != 1.0
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)
    cm.save(t, env_steps=0, tag="step0")
    t2 = Trainer(CFG, obs_dim=3, action_dim=1)
    cm.resume(t2)
    assert t2.ret_scale == t.ret_scale


# ---------------- encoder isolation (user requirement) ----------------
def test_penalty_never_touches_encoder():
    """The curvature penalty acts on the reward surface over (z, a) ONLY:
    backprop of the penalty alone must leave encoder grads at zero/None."""
    from mbrl.regularization.hutchinson import hvp_penalty
    torch.manual_seed(0)
    t = Trainer(CFG, obs_dim=3, action_dim=1)
    obs = torch.randn(32, 3)
    a = torch.randn(32, 1)
    z = t.encoder(obs)
    za = torch.cat([z.detach(), a], dim=-1)  # as in model_update
    pen = hvp_penalty(t.reward.on_concat, za, n_probes=2, generator=t.gen)
    pen.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in t.encoder.parameters()), \
        "curvature penalty leaked gradients into the encoder"
    # ...and it DOES reach the reward model's weights
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in t.reward.named_parameters() if "weight" in n)
