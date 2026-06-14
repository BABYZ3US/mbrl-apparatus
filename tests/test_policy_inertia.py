"""Policy inertia (optim.policy_ema_decay / policy_ema_act / policy_inertia, PM
2026-06-13): a slow EMA of the policy weights giving the policy extra inertia vs
the faster operator (two-timescale collapse stabilizer). Pins: default OFF (no
EMA, no behaviour change); EMA maintained + lags the live policy; act() uses the
EMA when policy_ema_act; the inertia anchor adds a positive term when the policy
has drifted from its EMA; bitwise resume; works in the dual/twin path."""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(ema_decay=0.0, ema_act=False, inertia=0.0, dual=False):
    model = {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99}
    if dual:
        model |= {"dynamics": "operator",
                  "operator": {"structure": "none", "rank": 0, "w_normal": 0.0,
                               "w_smooth": 0.0, "w_spread": 0.0, "w_radius": 0.0},
                  "dual_latent": {"enabled": True, "mode": "twin", "smooth_p": False,
                                  "couple_weight": 0.1, "p_consistency_weight": 1.0}}
    return OmegaConf.create({
        "seed": 0, "model": model,
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "schedule": {"kind": "constant", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 4, "gamma": 0.99, "lambda_": 0.95},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 16,
                  "policy_ema_decay": ema_decay, "policy_ema_act": ema_act,
                  "policy_inertia": inertia},
    })


def _batch(n=16, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_off_by_default():
    seed_everything(0)
    t = Trainer(_cfg(), obs_dim=3, action_dim=2)
    assert t.policy_ema is None
    t.behaviour_update(t.encoder(_batch()[0]).detach())   # runs unchanged
    assert t.act(t.encoder(_batch()[0]).detach()).shape == (16, 2)


def test_ema_maintained_and_lags():
    seed_everything(0)
    t = Trainer(_cfg(ema_decay=0.9), obs_dim=3, action_dim=2)
    assert t.policy_ema is not None
    init = [p.clone() for p in t.policy_ema.parameters()]
    for i in range(3):
        t.behaviour_update(t.encoder(_batch(seed=i)[0]).detach())
    assert any(not torch.equal(a, b) for a, b in zip(init, t.policy_ema.parameters()))      # moved
    assert any(not torch.equal(pe, p) for pe, p in                                          # lags live
               zip(t.policy_ema.parameters(), t.policy.parameters()))


def test_act_uses_ema_when_enabled():
    seed_everything(0)
    t = Trainer(_cfg(ema_decay=0.9, ema_act=True), obs_dim=3, action_dim=2)
    # make the EMA clearly different from the live policy
    with torch.no_grad():
        for pe in t.policy_ema.parameters():
            pe.add_(1.0)
    z = t.encoder(_batch()[0]).detach()
    seed_everything(1); a_ema = t.act(z)
    t.policy_ema_act = False
    seed_everything(1); a_live = t.act(z)
    assert not torch.allclose(a_ema, a_live)            # act() routed through the EMA


def test_inertia_anchor_positive_after_drift():
    seed_everything(0)
    t = Trainer(_cfg(ema_decay=0.9, inertia=0.5), obs_dim=3, action_dim=2)
    assert float(t._policy_inertia_term()) == 0.0       # equal at init -> no penalty
    with torch.no_grad():                                # drift the live policy
        for p in t.policy.parameters():
            p.add_(0.1)
    term = t._policy_inertia_term()
    assert float(term) > 0.0 and torch.isfinite(term)    # anchor pulls it back


def test_works_in_dual_twin_path():
    seed_everything(0)
    t = Trainer(_cfg(ema_decay=0.9, inertia=0.1, dual=True), obs_dim=3, action_dim=2)
    assert t.policy_ema is not None and t.dual_latent
    init = [p.clone() for p in t.policy_ema.parameters()]
    t.behaviour_update(t.encoder(_batch()[0]).detach())   # dispatches to _behaviour_update_dual
    assert any(not torch.equal(a, b) for a, b in zip(init, t.policy_ema.parameters()))


def test_resume_bitwise_with_policy_inertia(tmp_path):
    cfg = _cfg(ema_decay=0.9, ema_act=True, inertia=0.2)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=2)
    for i in range(3):
        t1.model_update(_batch(seed=i))
        t1.behaviour_update(t1.encoder(_batch(seed=i)[0]).detach())
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    seed_everything(0); a_ref = t1.act(t1.encoder(_batch(seed=42)[0]).detach())

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=2)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    seed_everything(0); a_res = t2.act(t2.encoder(_batch(seed=42)[0]).detach())
    assert torch.allclose(a_ref, a_res, atol=1e-6)       # EMA-policy acting restored exactly
