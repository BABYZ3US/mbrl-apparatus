"""Disagreement-gated lambda (penalty.disagreement_gate): the model-paced
anneal. Pins: default OFF is a no-op; the gate is bounded in [floor, 1] and
releases toward floor as the reward heads converge; reward_heads<2 disables it
with a warning; bitwise resume holds with the gate on (dis_ema/peak checkpointed).
"""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(gate=False, floor=0.1, heads=3):
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "reward_heads": heads},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "disagreement_gate": {"enabled": gate, "floor": floor, "decay": 0.9},
                    "schedule": {"kind": "constant", "lam0": 1.0, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 32},
    })


def _batch(n=32, obs_dim=3, act_dim=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=g)
    a = torch.randn(n, act_dim, generator=g)
    r = torch.randn(n, generator=g)
    obs_next = obs + 0.1 * torch.randn(n, obs_dim, generator=g)
    return obs, a, r, obs_next


def test_default_off_is_a_noop():
    seed_everything(0)
    t = Trainer(_cfg(gate=False), obs_dim=3, action_dim=1)
    assert not t.dg_enabled
    m = t.model_update(_batch())
    assert "penalty/dg_gate" not in m            # the gate never touched lambda
    # lambda equals the constant schedule exactly
    assert m["penalty/lambda"] == pytest.approx(1.0)


def test_gate_is_bounded_and_logged():
    seed_everything(0)
    t = Trainer(_cfg(gate=True, floor=0.1), obs_dim=3, action_dim=1)
    assert t.dg_enabled
    m = t.model_update(_batch())
    assert "penalty/dg_gate" in m and "penalty/disagreement" in m
    g = m["penalty/dg_gate"]
    assert 0.1 <= g <= 1.0                        # in [floor, 1]
    # at the very first step dis_ema == dis_peak -> gate is exactly 1 (full lam)
    assert g == pytest.approx(1.0)
    assert m["penalty/lambda"] == pytest.approx(1.0 * g)


def test_gate_releases_as_heads_converge():
    """Drive the EMA below the peak (heads converging) -> gate falls to floor."""
    seed_everything(0)
    t = Trainer(_cfg(gate=True, floor=0.1), obs_dim=3, action_dim=1)
    t.model_update(_batch())                      # establishes dis_peak
    peak = t.dis_peak
    assert peak > 0
    # simulate convergence: EMA collapses far below the peak
    t.dis_ema = 0.01 * peak
    m = t.model_update(_batch(seed=1))
    # gate ~ floor + (1-floor)*small -> near the floor, lam released accordingly
    assert m["penalty/dg_gate"] < 0.2
    assert m["penalty/lambda"] < 0.2              # constant schedule lam0=1 * gate


def test_reward_heads_one_disables_with_warning():
    seed_everything(0)
    with pytest.warns(UserWarning, match="reward_heads>=2"):
        t = Trainer(_cfg(gate=True, heads=1), obs_dim=3, action_dim=1)
    assert not t.dg_enabled                        # silently degrades to schedule
    m = t.model_update(_batch())
    assert "penalty/dg_gate" not in m


def test_resume_bitwise_identical_with_gate(tmp_path):
    cfg = _cfg(gate=True)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    for i in range(3):
        t1.model_update(_batch(seed=i))
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    ema_at_save, peak_at_save = t1.dis_ema, t1.dis_peak   # the checkpointed state
    m_ref = t1.model_update(_batch(seed=7))

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    assert t2.dis_ema == pytest.approx(ema_at_save) and t2.dis_peak == pytest.approx(peak_at_save)
    m_resumed = t2.model_update(_batch(seed=7))

    for k in ("loss/dyn", "loss/total", "penalty/lambda", "penalty/dg_gate"):
        assert m_resumed[k] == pytest.approx(m_ref[k], rel=1e-6), k
