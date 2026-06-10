"""The WIRED ensemble arm: algo.dynamics_ensemble consumed by the Trainer.

Pins: construction swap (affine-only), per-member regression trains every
member, the disagreement metric flows, the pessimism discount actually lowers
imagined returns, and bitwise resume holds WITH the ensemble enabled."""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models.ensemble import EnsembleAffineDynamics
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(ens: int = 3, pess: float = 0.0, dynamics: str = "affine"):
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "dynamics": dynamics},
        "algo": {"dynamics_ensemble": ens, "ensemble_pessimism": pess},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 32},
    })


def _batch(n: int = 32, obs_dim: int = 3, act_dim: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=g)
    a = torch.randn(n, act_dim, generator=g)
    r = torch.randn(n, generator=g)
    obs_next = obs + 0.1 * torch.randn(n, obs_dim, generator=g)
    return obs, a, r, obs_next


def test_ensemble_replaces_dynamics_and_trains_all_members():
    seed_everything(0)
    t = Trainer(_cfg(ens=3), obs_dim=3, action_dim=1)
    assert isinstance(t.dynamics, EnsembleAffineDynamics)
    assert t.dyn_ensemble
    before = [p.clone() for m in t.dynamics.members for p in m.parameters()]
    metrics = t.model_update(_batch())
    assert "dyn/disagreement" in metrics and metrics["dyn/disagreement"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()
               if isinstance(v, float))
    after = [p for m in t.dynamics.members for p in m.parameters()]
    changed = sum(1 for b, a_ in zip(before, after) if not torch.equal(b, a_))
    assert changed == len(after)               # every member's params moved


def test_default_path_keeps_single_dynamics():
    seed_everything(0)
    t = Trainer(_cfg(ens=0), obs_dim=3, action_dim=1)
    assert not t.dyn_ensemble
    assert not isinstance(t.dynamics, EnsembleAffineDynamics)


def test_ensemble_requires_affine_dynamics():
    with pytest.raises(ValueError, match="requires model.dynamics=affine"):
        Trainer(_cfg(ens=3, dynamics="gaussian"), obs_dim=3, action_dim=1)


def test_pessimism_lowers_imagined_returns():
    """Same seed, same updates — the pessimistic trainer's imagined returns sit
    strictly below the neutral one's (the discount is real, not cosmetic)."""
    outs = []
    for pess in (0.0, 2.0):
        seed_everything(0)
        t = Trainer(_cfg(ens=3, pess=pess), obs_dim=3, action_dim=1)
        for i in range(2):
            t.model_update(_batch(seed=i))
        seed_everything(123)                   # identical imagination stochasticity
        z0 = t.encoder(_batch(seed=9)[0])
        outs.append(t.behaviour_update(z0)["imagine/return_mean"])
    assert outs[1] < outs[0]


def test_resume_bitwise_identical_with_ensemble(tmp_path):
    cfg = _cfg(ens=3, pess=0.5)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    for i in range(3):
        t1.model_update(_batch(seed=i))
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    m_ref = t1.model_update(_batch(seed=7))

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    m_resumed = t2.model_update(_batch(seed=7))

    for k in ("loss/dyn", "loss/total", "dyn/disagreement"):
        assert m_resumed[k] == pytest.approx(m_ref[k], rel=1e-6), k
