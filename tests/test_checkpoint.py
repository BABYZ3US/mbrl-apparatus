"""save -> restore -> identical next step (the Colab-resume guarantee)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from omegaconf import OmegaConf

from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager

CFG = OmegaConf.create({
    "seed": 0,
    "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
    "penalty": {"n_probes": 2, "penalize_dynamics": False,
                "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 0.0}},
    "smoothing": {"enabled": True, "sigma": 1.5},
    "imagination": {"horizon": 5, "gamma": 0.99},
    "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
})


def fake_batch(n=32, obs_dim=3, act_dim=1):
    g = torch.Generator().manual_seed(123)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_resume_bitwise_identical(tmp_path):
    t1 = Trainer(CFG, obs_dim=3, action_dim=1)
    for _ in range(3):
        t1.model_update(fake_batch())
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    m_ref = t1.model_update(fake_batch())  # the "next step" after saving

    t2 = Trainer(CFG, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)
    assert cm2.resume(t2) == 300
    m_resumed = t2.model_update(fake_batch())

    for k in ("loss/dyn", "loss/reward", "loss/total"):
        assert m_resumed[k] == pytest.approx(m_ref[k], rel=1e-6), k


def test_config_change_starts_fresh_lineage(tmp_path):
    """auto-resume under a changed config must start fresh, not crash
    (hash-scoped lineage dirs) — regression for the multitask resume error."""
    t = Trainer(CFG, obs_dim=3, action_dim=1)
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)
    cm.save(t, env_steps=500, tag="step1")

    other = OmegaConf.to_container(CFG)
    other["model"]["latent_dim"] = 8
    cm_new = CheckpointManager(tmp_path, other, every=10)
    other_cfg = OmegaConf.create(other)
    assert cm_new.resume(Trainer(other_cfg, obs_dim=3, action_dim=1)) == 0  # fresh
    # old lineage untouched and still resumable under the old config
    cm_old = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)
    assert cm_old.resume(Trainer(CFG, obs_dim=3, action_dim=1)) == 500


def test_explicit_path_hash_mismatch_raises(tmp_path):
    t = Trainer(CFG, obs_dim=3, action_dim=1)
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)
    path = cm.save(t, env_steps=0, tag="step0")
    other = OmegaConf.to_container(CFG)
    other["model"]["hidden"] = 64
    cm_bad = CheckpointManager(tmp_path, other, every=10)
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        cm_bad.resume(Trainer(CFG, obs_dim=3, action_dim=1), mode=str(path))
