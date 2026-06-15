"""Tests for the seven upgrades: vectorized collection (autoreset masking),
auto-dosed lambda, symlog reward, reward ensemble + pessimism, latent
LayerNorm, and the curvature-certified adaptive horizon.

No MuJoCo, no wandb: Pendulum (classic-control) only; video logging is
exercised nowhere here (it is try/except-guarded in scripts/train.py).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from mbrl.models import Encoder, EMAEncoder, RewardModel
from mbrl.models.reward import symlog, symexp
from mbrl.training import Trainer, ReplayBuffer, collect_vectorized
from mbrl.utils.checkpoint import CheckpointManager


def make_cfg(**over):
    base = {
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100,
                                 "floor": 0.0}},
        "smoothing": {"enabled": True, "sigma": 1.5},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(over))


def fake_batch(n=32, obs_dim=3, act_dim=1, seed=123, r_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            r_scale * torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


# ---------------- symlog ----------------
def test_symlog_roundtrip_and_shape():
    x = torch.linspace(-1e4, 1e4, 1001)
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4, atol=1e-4)
    assert symlog(torch.tensor(0.0)).item() == 0.0
    # compressive: |symlog| grows logarithmically
    assert symlog(torch.tensor(1e6)).item() < 15.0
    # odd function
    assert torch.allclose(symlog(-x), -symlog(x))


def test_reward_model_trains_in_symlog_space():
    cfg = make_cfg(model={"symlog_reward": True})
    torch.manual_seed(0)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    obs, a, r, obs_next = fake_batch(n=64, r_scale=1000.0)  # raw vs symlog differ hugely
    with torch.no_grad():
        expected_symlog = F.mse_loss(t.reward(t.encoder(obs), a), symlog(r)).item()
        expected_raw = F.mse_loss(t.reward(t.encoder(obs), a), r).item()
    m = t.model_update((obs, a, r, obs_next))
    assert m["loss/reward"] == pytest.approx(expected_symlog, rel=1e-5)
    assert m["loss/reward"] < 0.01 * expected_raw  # definitely not the raw-space loss
    # imagination consumes symexp of the model output
    g = torch.Generator().manual_seed(7)
    z = torch.randn(16, t.encoder.latent_dim, generator=g)
    a2 = torch.randn(16, 1, generator=g)
    with torch.no_grad():
        r_im, _ = t._imagined_reward(z, a2)
        assert torch.allclose(r_im, symexp(t.reward(z, a2)), atol=1e-6)


# ---------------- reward ensemble + pessimism ----------------
def test_reward_ensemble_shapes_and_disagreement():
    torch.manual_seed(0)
    rm = RewardModel(4, 2, hidden=32, depth=1, n_heads=3)
    z, a = torch.randn(16, 4), torch.randn(16, 2)
    heads = rm.all_heads(z, a)
    assert heads.shape == (3, 16)
    assert rm(z, a).shape == (16,)
    assert torch.allclose(rm(z, a), heads.mean(0), atol=1e-6)  # forward = head mean
    assert heads.std(0).mean().item() > 0  # independent head inits disagree
    # on_concat (penalty target) is also the head mean, in raw model space
    x = torch.cat([z, a], dim=-1)
    assert torch.allclose(rm.on_concat(x), heads.mean(0), atol=1e-6)
    # task-conditioned path keeps working
    rm_t = RewardModel(4, 2, hidden=32, depth=1, task_dim=1, n_heads=3)
    tau = torch.rand(16, 1)
    assert rm_t.all_heads(z, a, tau).shape == (3, 16)
    assert rm_t(z, a, tau).shape == (16,)


def test_pessimism_reduces_imagined_reward():
    cfg = make_cfg(model={"reward_heads": 3, "symlog_reward": True},
                   imagination={"pessimism": 0.5})
    torch.manual_seed(0)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, t.encoder.latent_dim, generator=g)
    a = torch.randn(64, 1, generator=g)
    with torch.no_grad():
        heads = symexp(t.reward.all_heads(z, a))
        mean = heads.mean(0)
        r_used, dis = t._imagined_reward(z, a)
    assert dis.item() > 0
    assert (r_used <= mean + 1e-7).all()           # pessimism never adds reward
    assert (mean - r_used).max().item() > 0        # ...and strictly subtracts somewhere
    assert torch.allclose(r_used, mean - 0.5 * heads.std(0), atol=1e-6)
    # behaviour_update logs positive disagreement
    b = t.behaviour_update(torch.randn(32, t.encoder.latent_dim, generator=g))
    assert b["model/reward_disagreement"] > 0
    assert math.isfinite(b["loss/policy"])


# ---------------- adaptive horizon ----------------
def test_adaptive_horizon_tracks_pen_ema_and_is_checkpointed(tmp_path):
    cfg = make_cfg(imagination={"adaptive_horizon": {
        "enabled": True, "h_min": 5, "h_max": 25, "decay": 0.99}})
    torch.manual_seed(0)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(32, t.encoder.latent_dim, generator=g)

    t.pen_ema, t.pen_peak = 1.0, 1.0      # curvature at its peak -> shortest
    assert t.behaviour_update(z0)["imagine/horizon"] == 5
    t.pen_ema = 0.5                        # halfway down -> mid horizon
    h_mid = t.behaviour_update(z0)["imagine/horizon"]
    assert 5 < h_mid < 25
    t.pen_ema = 0.0                        # certified smooth -> longest
    assert t.behaviour_update(z0)["imagine/horizon"] == 25

    # model_update maintains the EMA and the running peak
    t.pen_ema, t.pen_peak = None, 0.0
    t.model_update(fake_batch())
    assert t.pen_ema is not None and t.pen_peak >= t.pen_ema > 0

    # pen_ema / pen_peak survive a checkpoint round-trip
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t, env_steps=1, tag=f"step{t.step}")  # resume() finds ckpt_step<N>.pt
    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    assert cm.resume(t2) == 1
    assert t2.pen_ema == t.pen_ema and t2.pen_peak == t.pen_peak


def test_adaptive_horizon_disabled_uses_cfg_horizon():
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    t.pen_ema, t.pen_peak = 0.0, 1.0  # would mean h_max if enabled
    b = t.behaviour_update(torch.randn(16, t.encoder.latent_dim))
    assert b["imagine/horizon"] == 5  # the plain cfg.imagination.horizon


# ---------------- cf17 horizon ratchet (monotonic non-decreasing floor) ----------------
def _ratchet_cfg(**ah):
    base = {"enabled": True, "h_min": 5, "h_max": 25, "ratchet": True, "ratchet_base": 15}
    base.update(ah)
    return make_cfg(imagination={"adaptive_horizon": base})


def test_horizon_ratchet_locks_running_max():
    """Below ratchet_base the horizon passes through; once it reaches the base it locks a
    running-max floor and can rise but never fall below its peak (anti-collapse)."""
    t = Trainer(_ratchet_cfg(), obs_dim=3, action_dim=1)
    assert t._horizon_ratchet(5) == 5 and not t._ah_ratchet_on   # below base: free
    assert t._horizon_ratchet(12) == 12 and not t._ah_ratchet_on
    assert t._horizon_ratchet(15) == 15 and t._ah_ratchet_on     # reaches base: engages
    assert t._horizon_ratchet(20) == 20                          # climbs -> floor follows
    assert t._horizon_ratchet(25) == 25
    assert t._horizon_ratchet(8) == 25                           # spike down -> HELD at peak
    assert t._horizon_ratchet(15) == 25                          # still held
    assert t._ah_ratchet_floor == 25


def test_horizon_ratchet_off_can_fall():
    """ratchet=false is the original adaptive behaviour: the horizon may drop."""
    t = Trainer(_ratchet_cfg(ratchet=False), obs_dim=3, action_dim=1)
    assert t._horizon_ratchet(25) == 25
    assert t._horizon_ratchet(5) == 5                            # no floor: free to fall


def test_horizon_ratchet_state_checkpointed():
    """The ratchet floor/engaged flags survive a checkpoint round-trip (bitwise resume)."""
    t = Trainer(_ratchet_cfg(), obs_dim=3, action_dim=1)
    t._horizon_ratchet(20)                                       # engage, floor=20
    assert t._ah_ratchet_on and t._ah_ratchet_floor == 20
    t2 = Trainer(_ratchet_cfg(), obs_dim=3, action_dim=1)
    t2.load_state_dict(t.state_dict())
    assert t2._ah_ratchet_on and t2._ah_ratchet_floor == 20
    assert t2._horizon_ratchet(5) == 20                          # resumed: still holds the floor


# ---------------- reward-adaptive policy regularization (PM 2026-06-15) ----------------
def test_policy_init_scale_near_zero():
    from mbrl.models.policy import Policy
    torch.manual_seed(0)
    p = Policy(8, 2, hidden=32, depth=1, init_scale=0.001)
    mu, log_std = p(torch.randn(16, 8))
    assert mu.abs().mean() < 0.05 and log_std.abs().mean() < 0.05   # ~same near-zero map every seed


def _ra_cfg(**ra):
    base = {"mid": 0.0, "scale": 100.0}
    base.update(ra)
    return make_cfg(reward_adapt=base)


def test_reward_frac_maps_return():
    t = Trainer(_ra_cfg(entropy_anneal=True), obs_dim=3, action_dim=1)
    assert t._reward_frac() == 0.0                       # ret_ema None -> 0
    t.ret_ema = 0.0;   assert t._reward_frac() == 0.0    # at mid
    t.ret_ema = 50.0;  assert abs(t._reward_frac() - 0.5) < 1e-6
    t.ret_ema = 200.0; assert t._reward_frac() == 1.0    # clipped at 1


def test_policy_reg_three_knobs_track_reward():
    t = Trainer(_ra_cfg(entropy_anneal=True, entropy_floor={"enabled": True, "h_high": 1.0, "coef": 0.1},
                        actor_clip_adapt={"enabled": True, "min_frac": 0.1}), obs_dim=3, action_dim=1)
    H = torch.tensor(0.2)
    t.ret_ema = 0.0                                       # low return (rf=0): explore
    ec0, fp0, cl0 = t._policy_reg(H, 1.0)
    assert abs(ec0 - 1.0) < 1e-6 and fp0.item() > 0 and abs(cl0 - t.actor_clip) < 1e-6
    t.ret_ema = 200.0                                    # high return (rf=1): exploit
    ec1, fp1, cl1 = t._policy_reg(H, 1.0)
    assert ec1 < 1e-6 and fp1.item() == 0.0 and abs(cl1 - t.actor_clip * 0.1) < 1e-4


def test_reward_adapt_off_is_identity():
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)     # no reward_adapt block -> all off
    ec, fp, cl = t._policy_reg(torch.tensor(0.5), 3e-4)
    assert ec == 3e-4 and fp.item() == 0.0 and cl == t.actor_clip


def test_entropy_floor_sigmoid_catches_at_target():
    # sigmoid floor (PM 2026-06-15): bounded penalty, lift PEAKS at the target and vanishes
    # for a deep collapse — a catch-as-it-crosses barrier (vs relu's constant lift).
    coef, beta, h = 0.1, 4.0, 1.0
    t = Trainer(_ra_cfg(entropy_floor={"enabled": True, "h_high": h, "coef": coef,
                                       "shape": "sigmoid", "beta": beta}), obs_dim=3, action_dim=1)
    t.ret_ema = 0.0                                       # rf=0 -> target H* = h
    def probe(Hval):
        H = torch.tensor(float(Hval), requires_grad=True)
        _, fp, _ = t._policy_reg(H, 1.0)
        fp.backward()
        return fp.item(), -float(H.grad)                 # penalty, lift (push entropy up)
    p_at, l_at = probe(h)                                # at the target
    p_deep, l_deep = probe(h - 20.0)                     # deep collapse (far below)
    p_above, l_above = probe(h + 5.0)                    # above the floor
    assert p_deep <= coef + 1e-6 and p_above < 1e-3      # penalty bounded by coef; ~0 above
    assert abs(p_at - coef * 0.5) < 1e-6                 # at the target: coef/2
    assert l_at > l_deep and l_at > l_above              # lift PEAKS at the target...
    assert l_deep < 1e-3 and l_above < 1e-3              # ...and vanishes deep / above
    assert abs(l_at - coef * beta * 0.25) < 1e-6         # coef*beta/4 at the target


def test_entropy_floor_relu_constant_lift_deep():
    # contrast: relu floor has CONSTANT lift = coef even for a deep collapse (it reverses it)
    coef, h = 0.1, 1.0
    t = Trainer(_ra_cfg(entropy_floor={"enabled": True, "h_high": h, "coef": coef,
                                       "shape": "relu"}), obs_dim=3, action_dim=1)
    t.ret_ema = 0.0
    H = torch.tensor(h - 20.0, requires_grad=True)
    _, fp, _ = t._policy_reg(H, 1.0); fp.backward()
    assert abs(-float(H.grad) - coef) < 1e-6             # constant lift = coef, no matter how deep


def test_logstd_floor_reward_adaptive():
    # cf21: HARD log_std floor driven by rf — high (explore) at low return, lo (commit) at high
    hi, lo = -1.0, -4.0
    t = Trainer(_ra_cfg(logstd_floor={"enabled": True, "hi": hi, "lo": lo}), obs_dim=3, action_dim=2)
    assert abs(t.policy.log_std_min - hi) < 1e-6        # rf=0 (no eval yet) -> explore floor
    t.observe_return(200.0)                             # ret_ema=200, scale=100 -> rf=1 -> commit
    assert abs(t.policy.log_std_min - lo) < 1e-6
    t2 = Trainer(_ra_cfg(logstd_floor={"enabled": True, "hi": hi, "lo": lo}), obs_dim=3, action_dim=2)
    t2.observe_return(50.0)                             # rf=0.5 -> linear midpoint
    assert abs(t2.policy.log_std_min - (hi + (lo - hi) * 0.5)) < 1e-6


def test_logstd_floor_clamps_policy_output():
    # the floor is a hard clamp: a policy that WANTS log_std=-50 is held at the floor; σ can't
    # collapse. Driving log_std_min at runtime (as the Trainer does) is respected by forward.
    from mbrl.models.policy import Policy
    p = Policy(4, 2, hidden=16, depth=1, log_std_min=-1.5)
    with torch.no_grad():
        last = [m for m in p.net.modules() if isinstance(m, nn.Linear)][-1]
        last.weight.zero_(); last.bias[2:].fill_(-50.0)         # force the log_std half very negative
    z = torch.randn(8, 4)
    assert torch.allclose(p(z)[1], torch.full((8, 2), -1.5))    # clamped to the floor
    p.log_std_min = -3.0                                        # Trainer relaxes the floor at runtime
    assert torch.allclose(p(z)[1], torch.full((8, 2), -3.0))


def test_logstd_floor_off_is_legacy_clamp():
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)           # no reward_adapt -> lsf off
    assert abs(t.policy.log_std_min - (-5.0)) < 1e-6           # legacy -5 clamp, untouched


# ---------------- auto-dosed lambda ----------------
def test_auto_dose_computes_finite_lam0_and_resumes_bitwise(tmp_path):
    cfg = make_cfg(penalty={"auto_dose": {"enabled": True, "target_ratio": 0.1,
                                          "warmup_updates": 3}})
    torch.manual_seed(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    assert t1.model_update(fake_batch(seed=1))["penalty/lambda"] == 0.0  # warmup
    assert t1.model_update(fake_batch(seed=2))["penalty/lambda"] == 0.0
    assert t1.lam0_auto is None

    # save MID-warmup: the accumulators themselves must resume
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=2, tag=f"step{t1.step}")  # resume() finds ckpt_step<N>.pt

    m3 = t1.model_update(fake_batch(seed=3))  # warmup completes here
    assert t1.lam0_auto is not None
    assert math.isfinite(t1.lam0_auto) and t1.lam0_auto > 0
    assert m3["penalty/lam0_auto"] == pytest.approx(t1.lam0_auto)
    assert t1.lam.lam0 == pytest.approx(t1.lam0_auto)  # schedule was re-dosed
    m4 = t1.model_update(fake_batch(seed=4))
    assert m4["penalty/lambda"] > 0.0  # schedule active post-warmup

    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 2
    r3 = t2.model_update(fake_batch(seed=3))
    r4 = t2.model_update(fake_batch(seed=4))
    assert t2.lam0_auto == pytest.approx(t1.lam0_auto, rel=1e-12)
    for k in ("loss/dyn", "loss/reward", "loss/total", "penalty/lambda"):
        assert r3[k] == pytest.approx(m3[k], rel=1e-7), k
        assert r4[k] == pytest.approx(m4[k], rel=1e-7), k


def test_auto_dose_disabled_by_default():
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    m = t.model_update(fake_batch())
    assert m["penalty/lambda"] > 0.0  # schedule runs from step 0 as before
    assert "penalty/lam0_auto" not in m


# ---------------- LayerNorm on the latent ----------------
def test_layernorm_on_latent_and_in_ema_copy():
    enc = Encoder(obs_dim=6, latent_dim=4, hidden=32, depth=1)
    assert isinstance(enc.net[-1], nn.LayerNorm)
    assert tuple(enc.net[-1].normalized_shape) == (4,)
    assert enc.net[-1].elementwise_affine
    ema = EMAEncoder(enc, decay=0.99)
    assert isinstance(ema.ema.net[-1], nn.LayerNorm)  # deep-copied into EMA
    out = enc(torch.randn(8, 6))
    assert out.shape == (8, 4) and torch.isfinite(out).all()


# ---------------- vectorized collection ----------------
def test_vector_collection_masks_autoreset_boundaries():
    gym = pytest.importorskip("gymnasium")
    torch.manual_seed(0)
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    num_envs = 2
    env = gym.vector.AsyncVectorEnv(
        [lambda: gym.make("Pendulum-v1") for _ in range(num_envs)])
    buf = ReplayBuffer(5000, 3, 1)
    obs, _ = env.reset(seed=0)
    autoreset = np.zeros(num_envs, dtype=bool)
    obs, autoreset, taken = collect_vectorized(t, env, buf, obs, autoreset, 500)
    env.close()

    assert taken == 500
    # Pendulum truncates at 200 -> each sub-env hits exactly one next-step
    # autoreset boundary in 250 vector steps; that fake
    # (final_obs, ignored_action, reset_obs) pair must NOT be in the buffer.
    assert len(buf) == 500 - num_envs
    # every stored transition is a real one-step Pendulum transition:
    # |delta thetadot| per real step is bounded (~1.05); a reset jump is not.
    n = len(buf)
    dvel = np.abs(buf.obs_next[:n, 2] - buf.obs[:n, 2])
    assert float(dvel.max()) < 1.3
    assert np.isfinite(buf.obs[:n]).all() and np.isfinite(buf.rew[:n]).all()


def test_symexp_overflow_clamped_and_nan_hygiene():
    """Regression for the shiny-run crash: extrapolating reward heads in symlog
    space overflowed symexp (expm1(>89) = inf in fp32) -> inf imagined rewards
    -> NaN ret_scale/policy -> NaN pen_ema -> horizon controller ValueError.
    The clamp must keep imagined rewards finite even with absurd head outputs,
    and one NaN penalty must not poison pen_ema or crash the horizon."""
    import math
    import torch
    from omegaconf import OmegaConf
    from mbrl.training import Trainer

    cfg = OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 3, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "symlog_reward": True, "reward_heads": 3},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "constant", "lam0": 1e-3, "t0": 100,
                                 "floor": 0.0}},
        "smoothing": {"enabled": False, "sigma": 1.5},
        "imagination": {"horizon": 10, "gamma": 0.99, "lambda_": 0.95,
                        "entropy_coef": 3e-4, "value_target_decay": 0.98,
                        "pessimism": 0.5,
                        "adaptive_horizon": {"enabled": True, "h_min": 5,
                                             "h_max": 25, "decay": 0.99}},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    })
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    # force absurd symlog-space outputs (>> 89, the fp32 expm1 overflow point)
    with torch.no_grad():
        for head in t.reward.heads:
            head.bias.fill_(500.0)
    r_im, _ = t._imagined_reward(torch.randn(16, t.encoder.latent_dim),
                                 torch.randn(16, 1))
    assert torch.isfinite(r_im).all(), "clamp failed: symexp overflow"

    b = t.behaviour_update(torch.randn(32, t.encoder.latent_dim))
    assert math.isfinite(b["loss/policy"]) and math.isfinite(t.ret_scale)

    # poisoned pen_ema path: a NaN penalty must not stick, horizon stays valid
    t.pen_ema, t.pen_peak = 1.0, 2.0
    import numpy as np
    t.pen_ema = float("nan")
    assert t._imagination_horizon() == 5  # guard, not ValueError


def test_data_driven_symexp_clamp_and_checkpoint_roundtrip(tmp_path):
    """The symexp clamp is data-driven: bound = symexp_margin * running max
    |symlog(r)| over real batches (replaces the fixed +-20, which allowed
    symexp up to 4.8e8 and imagined-return variance up to 1e19 in low-lambda
    windows). With rewards in [-12, 0], symlog_bound ~ log(13) ~ 2.565; even
    absurd head outputs must symexp to at most ~ symexp(1.5 * 2.565) ~ 47."""
    cfg = make_cfg(model={"symlog_reward": True, "reward_heads": 3},
                   imagination={"pessimism": 0.5})
    torch.manual_seed(0)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    assert t.symlog_bound == 1.0  # conservative floor before any real data

    # real batches with rewards in [-12, 0]
    for seed in (1, 2):
        obs, a, r, obs_next = fake_batch(n=64, seed=seed)
        r = -12.0 * torch.rand(64, generator=torch.Generator().manual_seed(seed))
        r[0] = -12.0  # pin the extreme so the expected bound is deterministic
        t.model_update((obs, a, r, obs_next))
    expected = symlog(torch.tensor(-12.0)).abs().item()  # log(13) ~ 2.565
    assert t.symlog_bound == pytest.approx(expected, rel=1e-6)

    # force absurd symlog-space head outputs (the extrapolation failure mode)
    with torch.no_grad():
        for head in t.reward.heads:
            head.bias.fill_(500.0)
    g = torch.Generator().manual_seed(3)
    r_im, _ = t._imagined_reward(torch.randn(64, t.encoder.latent_dim, generator=g),
                                 torch.randn(64, 1, generator=g))
    assert torch.isfinite(r_im).all()
    assert r_im.abs().max().item() < 100.0  # ~47 = symexp(1.5 * 2.565), not 4.8e8
    assert r_im.abs().max().item() == pytest.approx(
        symexp(torch.tensor(1.5 * expected)).item(), rel=1e-4)

    # the bound survives a checkpoint round-trip (resume protocol)
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t, env_steps=1, tag=f"step{t.step}")
    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    assert t2.symlog_bound == 1.0
    assert cm.resume(t2) == 1
    assert t2.symlog_bound == pytest.approx(t.symlog_bound, rel=0)
    with torch.no_grad():
        for head in t2.reward.heads:
            head.bias.fill_(500.0)
    g = torch.Generator().manual_seed(3)
    r_im2, _ = t2._imagined_reward(torch.randn(64, t2.encoder.latent_dim, generator=g),
                                   torch.randn(64, 1, generator=g))
    assert r_im2.abs().max().item() < 100.0
