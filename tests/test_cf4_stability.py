"""cf4 NaN-stabilization levers (PM 2026-06-14). Diagnosed on cf3-i0-s2: the twin's
UNREGULARIZED policy operator op_p (radius_p≈1.06>1) makes imagined p-rollouts grow
→ imagined returns → inf → actor grad → 46k → NaN, while loss/total stayed ~1e-3 (so
normalizing the total loss is the wrong target). Defence in depth, all default-off ⇒
legacy byte-exact:
  - model.dual_latent.radius_p — reinstate ONLY op_p's radius prior even when p is rough;
  - imagination.reward_clip / return_clip — cap imagined reward + λ-returns;
  - optim.value_clip — grad-clip the (previously unclipped) value optimizer;
  - optim.skip_nonfinite — skip the opt step on a non-finite grad (no weight poisoning).
"""
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(mode="twin", smooth_p=False, radius_p=0.0, reward_clip=0.0, return_clip=0.0,
         value_clip=0.0, skip_nonfinite=False, **op):
    opw = {f"w_{k}": op.get(f"w_{k}", 0.0) for k in ("normal", "smooth", "spread", "radius")}
    dl = {"enabled": True, "mode": mode, "d_dim": 0, "p_dim": 0, "couple_weight": 0.1,
          "p_consistency_weight": 1.0, "penalize_reward": False, "smooth_p": smooth_p,
          "radius_p": radius_p}
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "dynamics": "operator", "reward_heads": 1,
                  "operator": {"structure": "none", "rank": 0, **opw},
                  "dual_latent": dl},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "schedule": {"kind": "constant", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 4, "gamma": 0.99, "lambda_": 0.95,
                        "reward_clip": reward_clip, "return_clip": return_clip},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 16,
                  "value_clip": value_clip, "skip_nonfinite": skip_nonfinite},
    })


def _batch(n=16, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_defaults_off_no_new_behaviour():
    """Every lever off by default ⇒ the knobs read to their neutral values."""
    seed_everything(0)
    t = Trainer(_cfg(), obs_dim=3, action_dim=2)
    assert t.reward_clip == 0.0 and t.return_clip == 0.0 and t.value_clip == 0.0
    assert t.skip_nonfinite is False and t._nonfinite_skips == 0
    assert t.op_w_p["radius"] == 0.0          # radius_p off ⇒ op_p still fully rough


def test_radius_p_reinstates_only_op_p_radius_prior():
    """radius_p>0 bounds op_p's spectral radius WITHOUT smoothing it: radius weight
    on, normal/smooth/spread still 0 (p stays rough). The penalty is logged for op_p."""
    seed_everything(0)
    t = Trainer(_cfg(mode="twin", smooth_p=False, radius_p=0.1), obs_dim=3, action_dim=2)
    assert t.op_w_p["radius"] == 0.1
    assert t.op_w_p["smooth"] == 0.0 and t.op_w_p["normal"] == 0.0 and t.op_w_p["spread"] == 0.0
    m = t.model_update(_batch())
    assert "op/pen_radius_p" in m and math.isfinite(m["op/pen_radius_p"])


def test_reward_clip_bounds_imagined_reward():
    seed_everything(0)
    t = Trainer(_cfg(reward_clip=0.5), obs_dim=3, action_dim=2)
    z = t.encoder(_batch()[0]).detach()
    p = t.dual.p_of(z)
    a = torch.randn(z.shape[0], 2)
    r, _ = t._imagined_reward(p, a)
    assert r.abs().max().item() <= 0.5 + 1e-6


def test_return_clip_bounds_returns_and_trains():
    seed_everything(0)
    t = Trainer(_cfg(return_clip=2.0, reward_clip=10.0), obs_dim=3, action_dim=2)
    bm = t.behaviour_update(t.encoder(_batch()[0]).detach())
    assert math.isfinite(bm["loss/policy"]) and math.isfinite(bm["loss/value"])
    assert abs(bm["imagine/return_mean"]) <= 2.0 + 1e-6
    assert "stab/nonfinite_skips" in bm


def test_skip_nonfinite_protects_weights_from_a_diverged_rollout(monkeypatch):
    """A non-finite imagined reward (an op_p blowup) drives the actor/value grads
    non-finite; with skip_nonfinite the optimizer step is SKIPPED so θ_π and θ_v are
    left finite and unchanged, and the skip counter increments (the run survives)."""
    seed_everything(0)
    t = Trainer(_cfg(skip_nonfinite=True, value_clip=100.0), obs_dim=3, action_dim=2)
    z0 = t.encoder(_batch()[0]).detach()
    pi_before = [p.clone() for p in t.policy.parameters()]
    v_before = [p.clone() for p in t.value.parameters()]
    orig = t._imagined_reward
    def diverged(z, a, tau=None):
        r, d = orig(z, a, tau)
        return r * float("inf"), d        # the op_p-divergence symptom
    monkeypatch.setattr(t, "_imagined_reward", diverged)
    out = t.behaviour_update(z0)
    assert out["stab/nonfinite_skips"] >= 1
    # weights untouched (step skipped) and still finite — NOT poisoned by NaN
    for p, b in zip(t.policy.parameters(), pi_before):
        assert torch.equal(p, b) and torch.isfinite(p).all()
    for p, b in zip(t.value.parameters(), v_before):
        assert torch.equal(p, b) and torch.isfinite(p).all()


def test_skip_nonfinite_protects_model_weights(monkeypatch):
    """Same guard on the model update: a non-finite model loss must not step model_opt."""
    seed_everything(0)
    t = Trainer(_cfg(skip_nonfinite=True), obs_dim=3, action_dim=2)
    enc_before = [p.clone() for p in t.encoder.parameters()]
    orig = t.reward
    # poison the reward fit -> non-finite model loss
    def bad_reward(*a, **k):
        return orig(*a, **k) * float("inf")
    monkeypatch.setattr(t, "reward", bad_reward)
    t._model_update_dual(_batch())
    for p, b in zip(t.encoder.parameters(), enc_before):
        assert torch.equal(p, b) and torch.isfinite(p).all()
    assert t._nonfinite_skips >= 1


def test_resume_bitwise_with_stabilization(tmp_path):
    """skip-counter is checkpointed; full stabilized config resumes bitwise-exact."""
    cfg = _cfg(mode="twin", radius_p=0.1, reward_clip=10.0, return_clip=100.0,
               value_clip=100.0, skip_nonfinite=True)
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
    assert t2._nonfinite_skips == t1._nonfinite_skips
    seed_everything(0)
    a_res = t2.act(t2.encoder(_batch(seed=42)[0]).detach())
    assert torch.allclose(a_ref, a_res, atol=1e-6)
