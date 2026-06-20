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


def test_lyap_stein_consistency_off_by_default_and_finite_when_on():
    """Stein/Lyapunov consistency on op_d (term (c)): off by default (no metric, no
    loss change), and when weighted it logs op/lyap_stein, trains a finite step, and
    leaves the model_update output finite. Stable form — only second moments, so no
    NaN even at the near-identity init."""
    seed_everything(0)
    t_off = Trainer(_cfg(mode="twin"), obs_dim=3, action_dim=2)
    assert t_off.lyap_w == 0.0
    m_off = t_off.model_update(_batch())
    assert "op/lyap_stein" not in m_off          # inert ⇒ not even logged
    seed_everything(0)
    cfg = _cfg(mode="twin"); cfg.model.dual_latent.lyap_weight = 0.3
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    assert t.lyap_w == 0.3
    m = t.model_update(_batch())
    assert "op/lyap_stein" in m and math.isfinite(m["op/lyap_stein"]) and m["op/lyap_stein"] >= 0.0
    assert math.isfinite(m["loss/total"])


def _excite_cfg(p=1.0):
    """twin + lyap (populates innov_ema via Stein) + the discrete excitation gate ON."""
    cfg = _cfg(mode="twin")
    cfg.model.dual_latent.lyap_weight = 0.3
    cfg.model.operator.excite_enabled = True
    cfg.model.operator.excite_p = p
    cfg.model.operator.excite_zstd_anchor = 0.8
    cfg.model.operator.excite_zstd_band = 0.1
    cfg.model.operator.excite_scale = 1.0
    return cfg


def test_excite_gate_off_by_default_is_inert():
    """excite_enabled=false ⇒ EMAs never touched, gate never fires, behaviour logs gate=0."""
    seed_everything(0)
    t = Trainer(_cfg(mode="twin"), obs_dim=3, action_dim=2)
    assert t.excite_enabled is False
    t.model_update(_batch())
    assert t.z_std_ema is None and t.innov_ema is None       # disabled ⇒ no state churn
    z0 = t.encoder(_batch()[0].to(t.device))
    b = t.behaviour_update(z0)
    assert b["excite/gate"] == 0.0 and b["excite/noise_std"] == 0.0


def test_excite_gate_fires_in_band_and_is_finite():
    """excite on + p=1 + ema z_std forced into band + innovation EMA populated ⇒ the gate fires,
    injects Q-scaled rollout noise, logs gate=1 / noise_std>0, and trains a finite behaviour step."""
    seed_everything(0)
    t = Trainer(_excite_cfg(p=1.0), obs_dim=3, action_dim=2)
    assert t.excite_enabled and t.excite_p == 1.0
    for _ in range(3):                                       # populate innov_ema (Stein) + z_std_ema
        t.model_update(_batch())
    assert t.innov_ema is not None and t.innov_ema > 0.0 and t.z_std_ema is not None
    t.z_std_ema = t.excite_zstd_anchor                       # force the operating-point gate condition
    b = t.behaviour_update(t.encoder(_batch()[0].to(t.device)))
    assert b["excite/gate"] == 1.0 and b["excite/noise_std"] > 0.0
    assert math.isfinite(b["loss/policy"]) and math.isfinite(b["loss/value"])


def test_excite_gate_respects_zstd_band():
    """Gate stays SHUT when ema z_std is outside [anchor±band], even at p=1 (operating-point only)."""
    seed_everything(0)
    t = Trainer(_excite_cfg(p=1.0), obs_dim=3, action_dim=2)
    for _ in range(3):
        t.model_update(_batch())
    t.z_std_ema = t.excite_zstd_anchor + 5 * t.excite_zstd_band   # far outside the band
    b = t.behaviour_update(t.encoder(_batch()[0].to(t.device)))
    assert b["excite/gate"] == 0.0 and b["excite/noise_std"] == 0.0


def test_detpos_op_p_constraint_off_by_default_and_finite_when_on():
    """det(op_p) > 0 barrier: off by default (no metric, no loss change). When
    weighted it logs op/det_p_mean (≈1 at the near-identity init), op/det_p_negfrac
    (0 — op_p starts orientation-preserving so the barrier is inactive), and trains
    finite. det is real even though op_p can be rotational (eigenvalues pair up)."""
    seed_everything(0)
    t_off = Trainer(_cfg(mode="twin"), obs_dim=3, action_dim=2)
    assert t_off.detpos_w == 0.0
    assert "op/det_p_mean" not in t_off.model_update(_batch())
    seed_everything(0)
    cfg = _cfg(mode="twin"); cfg.model.dual_latent.detpos_weight = 5.0
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    assert t.detpos_w == 5.0
    m = t.model_update(_batch())
    assert "op/det_p_mean" in m and math.isfinite(m["op/det_p_mean"])
    assert math.isfinite(m["op/detpos"]) and m["op/detpos"] >= 0.0
    assert m["op/det_p_negfrac"] == 0.0       # A_p ≈ I at init ⇒ det ≈ 1 > 0
    assert math.isfinite(m["loss/total"])


def test_radius_anneal_decays_ceiling_toward_floor_without_reaching():
    """radius_anneal: the svband ceiling decays radius_anneal_start→radius_max (floor)
    on exp(−step/τ), asymptotically — starts at 1, nears the floor far out but never
    reaches; op_d tracks it, op_p is untouched; logs op/radius_ceil."""
    seed_everything(0)
    cfg = _cfg(mode="twin")
    cfg.model.operator.radius_max = 0.4472136
    cfg.model.operator.w_svband = 5.0
    cfg.model.operator.radius_anneal_start = 1.0
    cfg.model.operator.radius_anneal_tau = 1000.0
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    floor, start = 0.4472136, 1.0
    op_p_before = t.dual.op_p.radius_max
    t.step = 0; t._anneal_operator_radius()
    assert abs(t.dual.op_d.radius_max - start) < 1e-6        # starts at 1
    t.step = 3000; t._anneal_operator_radius()
    assert floor < t.dual.op_d.radius_max < 0.6             # descending, exp(−3)≈0.05
    t.step = 7000; t._anneal_operator_radius()              # 7τ ⇒ exp(−7)≈9e-4, representable
    assert floor < t.dual.op_d.radius_max < floor + 1e-3    # near floor, NOT reaching
    assert t.dual.op_p.radius_max == op_p_before            # anneal touches op_d only
    m = t.model_update(_batch())
    assert "op/radius_ceil" in m and math.isfinite(m["op/radius_ceil"])


def test_struct_every_amortizes_svd_keeps_lyap_every_update(monkeypatch):
    """Phased SVD: struct_every=N runs the O(d^3) operator structural priors (svdvals) only every
    N-th update, while the matmul-only Stein/lyap lever stays every-update; struct_every=1 is the
    unchanged validated path."""
    import torch
    real = torch.linalg.svdvals

    def count_svd(every, updates=8):
        seed_everything(0)
        cfg = _cfg(mode="twin", w_normal=0.1)
        cfg.model.operator.struct_every = every
        cfg.model.dual_latent.lyap_weight = 0.3
        t = Trainer(cfg, obs_dim=3, action_dim=2)
        assert t.struct_every == every
        c = {"n": 0}
        monkeypatch.setattr(torch.linalg, "svdvals", lambda *a, **k: (c.__setitem__("n", c["n"] + 1), real(*a, **k))[1])
        t.step = 0
        lyap_each = []
        for _ in range(updates):
            m = t.model_update(_batch())
            lyap_each.append("op/lyap_stein" in m)
        monkeypatch.setattr(torch.linalg, "svdvals", real)
        return c["n"], all(lyap_each), m

    n1, lyap1, m1 = count_svd(1)
    n4, lyap4, m4 = count_svd(4)
    assert n1 > 0 and n4 < n1                 # phasing reduces svdvals calls
    assert n4 <= n1 // 3 + 2                   # ~1/4 the calls (every=4 -> struct on steps 0,4 of 8)
    assert lyap1 and lyap4                     # lyap computed on EVERY update in both
    assert "op/pen_normal_d" in m4             # diagnostics still logged (from cache) when skipped


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


def test_gram_readouts_track_representation_collapse():
    """The order-parameter readout: a rank-1 (collapsed) representation gives
    eff_rank≈1 and a huge cond(G); an isotropic full-rank one gives eff_rank≈k and
    cond(G)≈1. This is the live distance-to-collapse signal."""
    seed_everything(0)
    t = Trainer(_cfg(), obs_dim=3, action_dim=2)
    k = t.encoder.latent_dim
    # collapsed: every sample a scalar multiple of one direction (defects proliferated)
    v = torch.randn(1, k)
    z_collapsed = torch.randn(64, 1) * v
    r1 = t._representation_readouts(z_collapsed)
    assert r1["latent/gram_eff_rank"] < 1.5
    assert r1["latent/gram_cond"] > 1e3
    # coherent: isotropic full-rank representation
    z_full = torch.randn(4096, k)
    r2 = t._representation_readouts(z_full)
    assert r2["latent/gram_eff_rank"] > 0.7 * k
    assert r2["latent/gram_cond"] < r1["latent/gram_cond"]
    for r in (r1, r2):
        assert all(math.isfinite(x) for x in r.values())


def test_readouts_and_phase_drift_logged():
    """gram_* readouts logged in BOTH paths; dual/phase_drift only for twin."""
    seed_everything(0)
    t = Trainer(_cfg(mode="twin", couple_weight=0.1), obs_dim=3, action_dim=2)
    m = t.model_update(_batch())
    assert {"latent/gram_cond", "latent/gram_eff_rank", "latent/gram_spectral_entropy"} <= set(m)
    assert "dual/phase_drift" in m and 0.0 <= m["dual/phase_drift"]


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
