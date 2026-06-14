"""Return-gated lambda (penalty.return_gate, PM 2026-06-13, rev. sigmoid form):
a WEAK, never-zero multiplier on lam(t) keyed on ACTUAL eval return, fed via
Trainer.observe_return. ABSOLUTE + SIGN-AWARE + SMOOTH: gate = floor +
(1-floor)*(1 - σ((R̄-mid)/scale)). Pins: default OFF = no-op; gate ∈ [floor,1]
(never zero); positive return -> ~floor, negative -> ~1, return≈mid -> MIDDLE of
[floor,1] (NO spike near zero — the bug in the old running-min/max form); sign is
distinguished; the per-eval slew limit caps spikes; state checkpointed."""
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(enabled=True, floor=0.5, decay=0.0, mid=0.0, scale=100.0, slew=1.0,
         shape="quadratic"):
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "dynamics": "affine", "reward_heads": 1},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "return_gate": {"enabled": enabled, "floor": floor, "decay": decay,
                                    "mid": mid, "scale": scale, "slew": slew, "shape": shape},
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
    t.observe_return(123.0)
    assert t.rg_gate_now == 1.0


def test_sigmoid_smooth_signaware_no_spike_near_zero():
    """sigmoid shape: at return≈mid the gate is the MIDDLE of [floor,1], not max;
    smooth and sign-aware (positive→floor, negative→1)."""
    seed_everything(0)
    t = Trainer(_cfg(floor=0.5, decay=0.0, mid=0.0, scale=100.0, slew=1.0, shape="sigmoid"),
                obs_dim=3, action_dim=2)
    t.observe_return(0.0)                       # return AT the midpoint
    assert abs(t.rg_gate_now - 0.75) < 1e-6     # MIDDLE of [0.5,1] — NOT a spike to max
    t.observe_return(+1000.0)
    assert abs(t.rg_gate_now - 0.5) < 1e-3      # -> floor
    t.observe_return(-1000.0)
    assert t.rg_gate_now > 0.99                 # -> ~1


def test_quadratic_fullrange_smooth_slope():
    """quadratic (default): a FULL-RANGE parabola — λ varies smoothly across the
    whole band [mid-scale, mid+scale], on BOTH sides of mid (not flat for negative
    return). gate=1 at/below mid-scale, floor at/above mid+scale, 0.875 at mid."""
    seed_everything(0)
    def gate_at(r):
        t = Trainer(_cfg(floor=0.5, decay=0.0, mid=0.0, scale=100.0, slew=1.0,
                         shape="quadratic"), obs_dim=3, action_dim=2)
        t.observe_return(r); return t.rg_gate_now
    assert gate_at(-500.0) == 1.0                             # full λ below mid-scale
    assert abs(gate_at(0.0) - 0.875) < 1e-6                   # at mid: 0.5+0.5*0.75
    assert abs(gate_at(100.0) - 0.5) < 1e-6                   # floor at mid+scale
    assert abs(gate_at(5000.0) - 0.5) < 1e-6                  # clipped to floor
    # the KEY fix: the gate VARIES in the negative regime (not flat at 1)
    assert 0.875 < gate_at(-50.0) < 1.0
    # monotone non-increasing as return climbs, bounded [floor,1]
    prev = 1.01
    for r in (-200.0, -50.0, 0.0, 50.0, 100.0, 300.0):
        g = gate_at(r)
        assert 0.5 - 1e-9 <= g <= 1.0 + 1e-9 and g <= prev + 1e-9
        prev = g


def test_cuberoot_vs_quadratic_curvature():
    """cuberoot gate (p=1/3) is CONCAVE — relaxes λ faster than the quadratic
    (p=2, convex) for the same return. Both full-range, both monotone, bounded.
    This is the gate-curvature axis."""
    seed_everything(0)
    def gate(r, shape):
        t = Trainer(_cfg(floor=0.1, decay=0.0, mid=0.0, scale=400.0, slew=1.0,
                         shape=shape), obs_dim=3, action_dim=2)
        t.observe_return(r); return t.rg_gate_now
    # endpoints agree; curvature differs in between
    assert abs(gate(-400.0, "cuberoot") - 1.0) < 1e-6 and abs(gate(-400.0, "quadratic") - 1.0) < 1e-6
    assert abs(gate(400.0, "cuberoot") - 0.1) < 1e-6 and abs(gate(400.0, "quadratic") - 0.1) < 1e-6
    # at mid and above, cuberoot has ALREADY relaxed λ far more than quadratic
    for r in (0.0, 100.0, 200.0):
        assert gate(r, "cuberoot") < gate(r, "quadratic") - 0.1
    # cuberoot bounded + monotone too
    prev = 1.01
    for r in (-400.0, -100.0, 0.0, 100.0, 400.0):
        g = gate(r, "cuberoot")
        assert 0.1 - 1e-9 <= g <= 1.0 + 1e-9 and g <= prev + 1e-9
        prev = g


def test_slew_limits_the_spike():
    """A collapse can't spike the gate: starting at 1.0, one big-positive eval can
    move the gate by at most `slew`, not jump straight to the floor."""
    seed_everything(0)
    t = Trainer(_cfg(floor=0.5, decay=0.0, slew=0.05), obs_dim=3, action_dim=2)
    assert t.rg_gate_now == 1.0
    t.observe_return(+1000.0)                   # sigmoid target ~0.5, but slew caps it
    assert abs(t.rg_gate_now - 0.95) < 1e-6     # 1.0 - slew, not 0.5
    for _ in range(20):
        t.observe_return(+1000.0)
    assert abs(t.rg_gate_now - 0.5) < 1e-3      # eventually eases to the floor


def test_gate_scales_logged_lambda():
    seed_everything(0)
    t = Trainer(_cfg(floor=0.5, decay=0.0, slew=1.0), obs_dim=3, action_dim=2)
    t.observe_return(+1000.0)                   # drive gate to floor 0.5
    assert abs(t.rg_gate_now - 0.5) < 1e-3
    m = t.model_update(_batch())
    assert m["penalty/return_gate"] == t.rg_gate_now
    assert abs(m["penalty/lambda"] - 0.5 * t.rg_gate_now) < 1e-6   # schedule(0.5)*gate


def test_resume_bitwise_with_return_gate(tmp_path):
    cfg = _cfg(enabled=True, decay=0.95, slew=0.1)
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=2)
    for i in range(3):
        t1.model_update(_batch(seed=i))
        t1.observe_return(-300.0 + 80 * i)
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    g_ref, e_ref = t1.rg_gate_now, t1.ret_ema

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=2)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    assert t2.ret_ema == e_ref and t2.rg_gate_now == g_ref
