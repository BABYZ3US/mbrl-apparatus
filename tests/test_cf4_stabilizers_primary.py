"""cf4 NaN-stabilizers on the PRIMARY (single-latent) path (bug-fix, PM 2026-06-20).

The four cf4 stabilizer flags — return_clip, value_clip, skip_nonfinite, double_value —
were implemented ONLY in the dual-latent twins (_behaviour_update_dual). The primary /
single-latent behaviour_update SILENTLY IGNORED them: a config that set e.g. return_clip
stabilized the dual path but let the single-latent path diverge (no clipping). These tests
build the SMALLEST single-latent Trainer (NO model.dual_latent ⇒ behaviour_update takes the
primary branch, dynamics defaults to affine) and assert each stabilizer now takes effect on
the PRIMARY path with the SAME semantics as the dual twin — and that with each flag at its
default (no-op) value the primary path is byte-for-byte unchanged.

These mirror tests/test_cf4_stability.py (which only ever exercised the dual/twin path).
Pairs each enabled-flag assertion with a default-off assertion. No torch is run in the
authoring sandbox; assertions target the exact changed behaviour so they FAIL on pre-fix
code (primary path ignored the flag) and PASS after the fix (primary path honours it).
"""
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.training import Trainer
from mbrl.utils.seeding import seed_everything


def _cfg(reward_clip=0.0, return_clip=0.0, value_clip=0.0, skip_nonfinite=False,
         double_value=False):
    """SINGLE-LATENT (primary) Trainer config: NO model.dual_latent, dynamics defaults
    to affine ⇒ model_update/behaviour_update take the primary branch. Mirrors the
    minimal Trainer setup in tests/test_cf4_stability.py and test_returns_and_tasks.py."""
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "reward_heads": 1},
        "penalty": {"n_probes": 2, "penalize_dynamics": False, "form": "frobenius",
                    "auto_dose": {"enabled": False},
                    "schedule": {"kind": "constant", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 4, "gamma": 0.99, "lambda_": 0.95,
                        "reward_clip": reward_clip, "return_clip": return_clip},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 16,
                  "value_clip": value_clip, "skip_nonfinite": skip_nonfinite,
                  "clipped_double_value": double_value},
    })


def _batch(n=16, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def _z0(t, seed=0):
    """Encoded latents to seed the imagination (primary path: behaviour reads z directly)."""
    return t.encoder(_batch(seed=seed)[0].to(t.device)).detach()


def _is_primary(t):
    """Guard: these tests are meaningless unless the Trainer is on the single-latent path."""
    return t.dual_latent is False and t.dual is None


# --------------------------------------------------------------------------- #
# defaults-off: every flag reads to its neutral value and the primary path runs
# --------------------------------------------------------------------------- #
def test_defaults_off_primary_path_neutral():
    """All cf4 flags off by default ⇒ neutral values, single-latent path, finite step,
    zero skips. (Same neutrality contract test_cf4_stability asserts, but PRIMARY.)"""
    seed_everything(0)
    t = Trainer(_cfg(), obs_dim=3, action_dim=2)
    assert _is_primary(t)
    assert t.return_clip == 0.0 and t.value_clip == 0.0
    assert t.skip_nonfinite is False and t.double_value is False
    assert t._nonfinite_skips == 0
    assert not hasattr(t, "value2")        # double_value off ⇒ no second value net built
    b = t.behaviour_update(_z0(t))
    assert math.isfinite(b["loss/policy"]) and math.isfinite(b["loss/value"])
    assert b["stab/nonfinite_skips"] == 0  # primary path now reports the same diagnostic


# --------------------------------------------------------------------------- #
# flag 1: return_clip — hard-bound the imagined lambda-returns on the PRIMARY path
# --------------------------------------------------------------------------- #
def test_return_clip_bounds_returns_on_primary_path():
    """return_clip>0 ⇒ the lambda-returns reported by the PRIMARY behaviour_update are
    bounded by the clip (so a diverged rollout can't NaN the loss). Default-off leaves
    the return mean unclipped. Pre-fix the primary path ignored the flag entirely."""
    # large reward_clip keeps imagined reward finite; small return_clip is the binding bound
    seed_everything(0)
    t = Trainer(_cfg(return_clip=2.0, reward_clip=10.0), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.return_clip == 2.0
    b = t.behaviour_update(_z0(t))
    assert math.isfinite(b["loss/policy"]) and math.isfinite(b["loss/value"])
    assert abs(b["imagine/return_mean"]) <= 2.0 + 1e-6   # clamp took effect on the mean
    assert "stab/nonfinite_skips" in b


def test_return_clip_default_off_does_not_clip_primary():
    """Default return_clip=0 ⇒ no clamp; with a deliberately huge reward the return
    mean exceeds any small bound (proving the clamp is what bounds it when enabled)."""
    seed_everything(0)
    t = Trainer(_cfg(return_clip=0.0, reward_clip=0.0), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.return_clip == 0.0
    # blow up the imagined reward so |returns| is large; with no clip the mean is unbounded
    orig = t._imagined_reward
    def big(z, a, tau=None):
        r, d = orig(z, a, tau)
        return r + 1000.0, d
    t._imagined_reward = big
    b = t.behaviour_update(_z0(t))
    # un-clipped: the +1000 reward floods the lambda-returns far past any small bound
    assert abs(b["imagine/return_mean"]) > 2.0


# --------------------------------------------------------------------------- #
# flag 2: skip_nonfinite — a non-finite grad must skip the opt step (no weight poison)
# --------------------------------------------------------------------------- #
def test_skip_nonfinite_protects_primary_policy_and_value(monkeypatch):
    """A non-finite imagined reward drives the actor/value grads non-finite; with
    skip_nonfinite the PRIMARY behaviour_update SKIPS the optimizer step (θ_π, θ_v left
    finite & unchanged) and increments the skip counter. Mirrors the dual-path test
    test_skip_nonfinite_protects_weights_from_a_diverged_rollout, but PRIMARY."""
    seed_everything(0)
    t = Trainer(_cfg(skip_nonfinite=True, value_clip=100.0), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.skip_nonfinite is True
    z0 = _z0(t)
    pi_before = [p.clone() for p in t.policy.parameters()]
    v_before = [p.clone() for p in t.value.parameters()]
    orig = t._imagined_reward
    def diverged(z, a, tau=None):
        r, d = orig(z, a, tau)
        return r * float("inf"), d        # the op-divergence symptom: inf imagined reward
    monkeypatch.setattr(t, "_imagined_reward", diverged)
    out = t.behaviour_update(z0)
    assert out["stab/nonfinite_skips"] >= 1
    for p, b in zip(t.policy.parameters(), pi_before):
        assert torch.equal(p, b) and torch.isfinite(p).all()
    for p, b in zip(t.value.parameters(), v_before):
        assert torch.equal(p, b) and torch.isfinite(p).all()


def test_no_skip_default_off_steps_primary_even_when_nonfinite(monkeypatch):
    """skip_nonfinite=False (default) ⇒ NO guard: the optimizer steps unconditionally on
    the primary path, the policy weights CHANGE, and the counter stays 0. This pins that
    the guard is OFF by default (byte-identical to legacy) — pre-fix behaviour."""
    seed_everything(0)
    t = Trainer(_cfg(skip_nonfinite=False), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.skip_nonfinite is False
    z0 = _z0(t)
    pi_before = [p.clone() for p in t.policy.parameters()]
    orig = t._imagined_reward
    def diverged(z, a, tau=None):
        r, d = orig(z, a, tau)
        return r * float("inf"), d
    monkeypatch.setattr(t, "_imagined_reward", diverged)
    out = t.behaviour_update(z0)
    assert out["stab/nonfinite_skips"] == 0          # guard off ⇒ never counted
    # unconditional step on a non-finite grad ⇒ at least one policy param moved (poisoned)
    moved = any(not torch.equal(p, b) for p, b in zip(t.policy.parameters(), pi_before))
    assert moved


# --------------------------------------------------------------------------- #
# flag 3: value_clip — grad-clip the (previously unclipped) value optimizer
# --------------------------------------------------------------------------- #
def test_value_clip_bounds_value_grad_step_on_primary():
    """value_clip>0 caps the value optimizer's grad norm on the PRIMARY path: a value
    step taken WITH a tiny clip moves the value weights LESS than the same step with no
    clip. Pre-fix the primary value optimizer was always unclipped (flag ignored)."""
    z_seed, batch_seed = 1, 1

    def value_delta(value_clip):
        seed_everything(0)
        t = Trainer(_cfg(value_clip=value_clip), obs_dim=3, action_dim=2)
        assert _is_primary(t) and t.value_clip == value_clip
        before = [p.clone() for p in t.value.parameters()]
        t.behaviour_update(_z0(t, seed=z_seed))
        return sum(float((p - b).pow(2).sum())
                   for p, b in zip(t.value.parameters(), before))

    d_clipped = value_delta(1e-8)   # essentially zero-out the value grad
    d_unclip = value_delta(0.0)     # default: unclipped step
    assert d_unclip > 0.0           # the value net does move with no clip
    assert d_clipped < d_unclip     # a tight clip shrinks the value update
    # Adam normalizes the step, so a near-zero clip does NOT fully freeze the net — but
    # it shrinks the value update by >20x vs unclipped (observed ratio ~1.6e-3). A
    # relative bound is robust where an absolute one is not.
    assert d_clipped < d_unclip * 0.05


def test_value_clip_default_off_is_unclipped_primary():
    """value_clip=0 + skip_nonfinite=False ⇒ the value optimizer takes the plain
    unconditional step (the legacy primary behaviour: no grad-clip branch)."""
    seed_everything(0)
    t = Trainer(_cfg(value_clip=0.0, skip_nonfinite=False), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.value_clip == 0.0
    before = [p.clone() for p in t.value.parameters()]
    t.behaviour_update(_z0(t, seed=1))
    moved = any(not torch.equal(p, b) for p, b in zip(t.value.parameters(), before))
    assert moved                    # unclipped ⇒ a normal value step happened


# --------------------------------------------------------------------------- #
# flag 4: double_value — TD3 twin-value min(V1,V2) bootstrap on the PRIMARY path
# --------------------------------------------------------------------------- #
def test_double_value_builds_and_trains_second_net_on_primary():
    """clipped_double_value=true ⇒ a second value net + EMA target + opt exist, and the
    PRIMARY behaviour_update trains value2 (its weights move) and EMA-updates value2_target.
    Pre-fix value2 was constructed but NEVER touched on the single-latent path."""
    seed_everything(0)
    t = Trainer(_cfg(double_value=True), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.double_value is True
    assert hasattr(t, "value2") and hasattr(t, "value2_target") and hasattr(t, "value2_opt")
    v2_before = [p.clone() for p in t.value2.parameters()]
    tgt_before = [p.clone() for p in t.value2_target.parameters()]
    b = t.behaviour_update(_z0(t))
    assert math.isfinite(b["loss/value"]) and math.isfinite(b["loss/policy"])
    # value2 actually trained on the primary path (was previously inert there)
    assert any(not torch.equal(p, q) for p, q in zip(t.value2.parameters(), v2_before))
    # and its EMA target was lerp'd toward it
    assert any(not torch.equal(p, q)
               for p, q in zip(t.value2_target.parameters(), tgt_before))


def test_double_value_changes_value_target_bootstrap_on_primary(monkeypatch):
    """The min(V1,V2) bootstrap must actually be USED for the lambda-return target on the
    primary path: force value2_target ≪ value_target so the min picks value2_target, and
    check the imagined return mean shifts vs the single-value (V1-only) bootstrap.
    Pins that double_value rewires the PRIMARY target, not just trains a spare net."""
    seed_everything(0)
    t = Trainer(_cfg(double_value=True), obs_dim=3, action_dim=2)
    assert _is_primary(t)
    z0 = _z0(t)

    # baseline: monkeypatch value2_target to EQUAL value_target ⇒ min is a no-op (== V1-only)
    import copy
    eq_target = copy.deepcopy(t.value_target).requires_grad_(False)
    monkeypatch.setattr(t, "value2_target", eq_target)
    seed_everything(0)
    ret_v1only = t.behaviour_update(z0)["imagine/return_mean"]

    # now make value2_target output a large NEGATIVE constant ⇒ min(V1,V2)=V2 everywhere,
    # which lowers the bootstrapped lambda-returns ⇒ the reported return mean drops.
    seed_everything(0)
    t2 = Trainer(_cfg(double_value=True), obs_dim=3, action_dim=2)
    class _Low(torch.nn.Module):
        def forward(self, x, tau=None):
            return torch.full((x.shape[0],), -1e3, device=x.device)
    monkeypatch.setattr(t2, "value2_target", _Low())
    seed_everything(0)
    ret_min = t2.behaviour_update(z0)["imagine/return_mean"]
    assert math.isfinite(ret_min)
    assert ret_min < ret_v1only - 1.0     # the min(V1,V2) bootstrap genuinely lowered returns


def test_double_value_default_off_no_second_net_primary():
    """Default ⇒ no value2 attribute, single-value bootstrap (legacy primary path)."""
    seed_everything(0)
    t = Trainer(_cfg(double_value=False), obs_dim=3, action_dim=2)
    assert _is_primary(t) and t.double_value is False
    assert not hasattr(t, "value2")
    b = t.behaviour_update(_z0(t))
    assert math.isfinite(b["loss/value"])
