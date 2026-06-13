"""Return-gated lambda (penalty.return_gate, PM 2026-06-13): a WEAK, never-zero
multiplier on lam(t) keyed on ACTUAL eval return, fed via Trainer.observe_return.
Pins: default OFF = no-op (gate≡1); gate ∈ [floor,1] (never zero, R14); low return
⇒ ~1 (full lam), high return ⇒ ~floor (relaxed); the gate scales penalty/lambda;
ret_ema/lo/hi checkpointed (bitwise resume)."""
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(enabled=True, floor=0.5, decay=0.95):
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "dynamics": "affine", "reward_heads": 1},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "return_gate": {"enabled": enabled, "floor": floor, "decay": decay},
                    "schedule": {"kind": "constant", "lam0": 0.5, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 4, "gamma": 0.99, "lambda_": 0.95},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 16},
    })


def _batch(n=16, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_disabled_is_noop():
    seed_everything(0)
    t = Trainer(_cfg(enabled=False), obs_dim=3, action_dim=2)
    assert t.rg_gate_now == 1.0
    t.observe_return(123.0)                     # ignored when disabled
    assert t.rg_gate_now == 1.0


def test_gate_inverse_weak_and_never_zero():
    seed_everything(0)
    t = Trainer(_cfg(enabled=True, floor=0.5, decay=0.0), obs_dim=3, action_dim=2)
    # decay=0 => ret_ema tracks the latest return exactly (easy to reason about)
    t.observe_return(-1000.0)                    # first obs: span 0 -> full lam
    assert t.rg_gate_now == 1.0
    t.observe_return(+500.0)                     # new best -> ema at hi -> floor
    assert abs(t.rg_gate_now - 0.5) < 1e-6
    t.observe_return(-1000.0)                    # back to worst -> full lam again
    assert abs(t.rg_gate_now - 1.0) < 1e-6
    t.observe_return(-250.0)                     # mid -> between floor and 1, never <floor
    assert 0.5 <= t.rg_gate_now <= 1.0


def test_gate_scales_logged_lambda():
    seed_everything(0)
    t = Trainer(_cfg(enabled=True, floor=0.5, decay=0.0), obs_dim=3, action_dim=2)
    t.observe_return(-1000.0); t.observe_return(+1000.0)   # drive gate to floor
    assert abs(t.rg_gate_now - 0.5) < 1e-6
    m = t.model_update(_batch())
    assert m["penalty/return_gate"] == t.rg_gate_now
    # lam_t = schedule(=0.5 constant) * dg(1) * rg(0.5) ≈ 0.25
    assert abs(m["penalty/lambda"] - 0.25) < 1e-6


def test_resume_bitwise_with_return_gate(tmp_path):
    cfg = _cfg(enabled=True)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=2)
    for i in range(3):
        t1.model_update(_batch(seed=i))
        t1.observe_return(-300.0 + 50 * i)
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    g_ref = t1.rg_gate_now

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=2)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    assert t2.ret_ema == t1.ret_ema and t2.rg_gate_now == g_ref     # gate state restored
