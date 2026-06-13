"""Dual-latent controlled-operator model — shared backbone z splits into a
dynamics latent d=D(z) and a policy latent p=P(z); reward/policy/value read p.
Pins: shared=ONE operator (option 1) vs twin=TWO operators (option 3); the dual
path is GATED (default off ⇒ z-loop unchanged); guards (needs dynamics=operator,
no spectral); model+behaviour updates run and log the right diagnostics; twin
adds p-consistency + coupling; bitwise resume holds in both modes."""
import math
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models.dual_latent import DualLatent
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(enabled=True, mode="shared", couple_weight=0.0, **op):
    opw = {f"w_{k}": op.get(f"w_{k}", 0.0) for k in ("normal", "smooth", "spread", "radius")}
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "dynamics": "operator", "operator": {"structure": "none", "rank": 0, **opw},
                  "dual_latent": {"enabled": enabled, "mode": mode, "d_dim": 0, "p_dim": 0,
                                  "couple_weight": couple_weight, "p_consistency_weight": 1.0}},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "schedule": {"kind": "constant", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 4, "gamma": 0.99, "lambda_": 0.95},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 16},
    })


def _batch(n=16, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_module_shared_one_operator():
    seed_everything(0)
    dl = DualLatent(4, 2, hidden=16, depth=1, mode="shared")
    z = torch.randn(8, 4)
    assert dl.d_of(z).shape == (8, 4) and dl.p_of(z).shape == (8, 4)
    assert len(dl.operators()) == 1 and hasattr(dl, "op")
    assert dl.couple(dl.d_of(z), dl.p_of(z)).item() == 0.0      # no coupling in shared


def test_module_twin_two_operators_and_coupling():
    seed_everything(0)
    dl = DualLatent(4, 2, hidden=16, depth=1, mode="twin")
    z = torch.randn(8, 4)
    assert len(dl.operators()) == 2 and hasattr(dl, "op_d") and hasattr(dl, "op_p")
    assert dl.couple(dl.d_of(z), dl.p_of(z)).item() > 0.0       # real coupling term


def test_dual_disabled_leaves_z_path():
    seed_everything(0)
    t = Trainer(_cfg(enabled=False), obs_dim=3, action_dim=2)
    assert not t.dual_latent and t.dual is None
    m = t.model_update(_batch())
    assert "op/radius" in m                                     # the plain operator path


def test_guards_require_operator_and_reject_spectral():
    bad = _cfg(); bad.model.dynamics = "affine"
    with pytest.raises(ValueError):
        Trainer(bad, obs_dim=3, action_dim=2)
    bad2 = _cfg(); bad2.spectral = {"enabled": True}
    with pytest.raises(ValueError):
        Trainer(bad2, obs_dim=3, action_dim=2)


def test_shared_trains_and_acts():
    seed_everything(0)
    t = Trainer(_cfg(mode="shared", w_normal=0.05), obs_dim=3, action_dim=2)
    assert t.dual_latent and t.dual.mode == "shared"
    assert t.reward.k == t.dual.p_dim                           # heads read p
    m = t.model_update(_batch())
    assert {"loss/dyn", "loss/reward", "op/radius", "op/pen_normal"} <= set(m)
    bm = t.behaviour_update(t.encoder(_batch()[0]).detach())
    assert math.isfinite(bm["loss/policy"]) and math.isfinite(bm["loss/value"])
    assert t.act(t.encoder(_batch()[0]).detach()).shape == (16, 2)


def test_twin_p_consistency_and_coupling_and_dual_operators():
    seed_everything(0)
    t = Trainer(_cfg(mode="twin", couple_weight=0.1, w_normal=0.05), obs_dim=3, action_dim=2)
    m = t.model_update(_batch())
    assert "dual/p_consistency" in m and math.isfinite(m["dual/p_consistency"])
    assert "dual/couple" in m and m["dual/couple"] >= 0.0
    assert "op/radius_d" in m and "op/radius_p" in m            # two operators logged
    assert "op/pen_normal_d" in m and "op/pen_normal_p" in m
    bm = t.behaviour_update(t.encoder(_batch()[0]).detach())    # control rolls op_p
    assert math.isfinite(bm["loss/policy"])
    assert t.act(t.encoder(_batch()[0]).detach()).shape == (16, 2)


@pytest.mark.parametrize("mode", ["shared", "twin"])
def test_resume_bitwise_identical_dual(mode, tmp_path):
    cfg = _cfg(mode=mode, couple_weight=0.1, w_normal=0.05)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=2)
    for i in range(3):
        t1.model_update(_batch(seed=i))
        t1.behaviour_update(t1.encoder(_batch(seed=i)[0]).detach())
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    seed_everything(0)
    a_ref = t1.act(t1.encoder(_batch(seed=42)[0]).detach())

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=2)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    seed_everything(0)
    a_res = t2.act(t2.encoder(_batch(seed=42)[0]).detach())
    assert torch.allclose(a_ref, a_res, atol=1e-6)
