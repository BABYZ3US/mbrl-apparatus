"""Trainer-integrated spectral reward path (spectral.enabled=true).

Covers: (1) full spectral Trainer smoke (cache -> refit -> behaviour update),
(2) the data-driven symexp clamp on imagined spectral rewards, (3) poly_weights
hand values + per-degree schedule shifts, (4) bitwise save -> resume under a
spectral config, (5) policy gradients flow through the closed-form predict
(cos features are differentiable in (z, a)).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mbrl.models.spectral import SpectralReward, poly_weights
from mbrl.regularization.schedule import LambdaSchedule
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager


def make_cfg(poly=None, **spec_overrides):
    spec = {"enabled": True, "n_features": 64, "sigma_w": 1.0,
            "refit_every": 5, "cache_size": 256, "heads": 2,
            "poly": poly or {"degrees": [2], "coefs": [1.0], "shifts": [0]}}
    spec.update(spec_overrides)
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "symlog_reward": True, "reward_heads": 2},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3,
                                 "t0": 100, "floor": 0.0}},
        "spectral": spec,
        "smoothing": {"enabled": True, "sigma": 1.5},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    })


def fake_batch(n=32, obs_dim=3, act_dim=1, seed=123, r_scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            r_scale * torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_spectral_trainer_smoke():
    """Enough model updates to trigger a refit (batch 32, n_features 64 ->
    refit at update 2), then a behaviour update; everything finite, the exact
    spectral penalty positive after the refit."""
    torch.manual_seed(0)
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    metrics = []
    for i in range(8):
        metrics.append(t.model_update(fake_batch(seed=100 + i)))
    # before the first refit the heads predict zeros — logged, not fatal
    assert metrics[0]["spectral/fitted"] == 0.0
    assert metrics[-1]["spectral/refits"] >= 1
    assert metrics[-1]["spectral/cache_n"] == 8 * 32
    for m in metrics:
        for k, v in m.items():
            assert np.isfinite(v), (k, v)
    assert metrics[-1]["penalty/value"] > 0  # exact mean-over-heads H^2

    z0 = t.encoder(fake_batch(seed=999)[0]).detach()
    b = t.behaviour_update(z0)
    for k, v in b.items():
        assert np.isfinite(v), (k, v)


def test_imagined_rewards_respect_symexp_clamp():
    """Absurd targets / absurd coefficients: imagined spectral rewards must
    stay inside symexp(margin * symlog_bound)."""
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    # absurd real targets drive the data-driven bound up but keep it finite
    for i in range(3):
        m = t.model_update(fake_batch(seed=200 + i, r_scale=1e8))
    assert np.isfinite(t.symlog_bound) and t.symlog_bound > 1.0
    assert np.isfinite(m["loss/total"])
    # blow up the head coefficients: predictions are absurd pre-clamp
    for h in t.spec_heads:
        h.c = 1e6 * torch.randn(h.M, generator=torch.Generator().manual_seed(7))
    z = torch.randn(64, t.encoder(torch.zeros(1, 3)).shape[-1])
    a = torch.randn(64, 1)
    r_im, dis = t._imagined_reward(z, a)
    bound = math.expm1(t.symexp_margin * t.symlog_bound)
    assert torch.isfinite(r_im).all()
    # mean - pessimism*std of per-head clamped values cannot exceed the
    # per-head bound (pessimism=0 here -> plain mean)
    assert r_im.abs().max().item() <= bound * (1 + 1e-5)  # float32 slack at the bound
    assert np.isfinite(dis.item())


def test_poly_weights_hand_values_and_shifts():
    # pure quartic: |w| in {1, 2} -> |w|^4 in {1, 16}
    w = torch.tensor([1.0, 2.0])
    assert torch.allclose(poly_weights(w, [2], [1.0]), torch.tensor([1.0, 16.0]))
    # mixed degrees: 2|w|^2 + 3|w|^4 -> at |w|=1: 5; at |w|=2: 8 + 48 = 56
    assert torch.allclose(poly_weights(w, [1, 2], [2.0, 3.0]),
                          torch.tensor([5.0, 56.0]))
    with pytest.raises(ValueError):
        poly_weights(w, [1, 2], [1.0])

    # per-degree shifts phase-shift the lambda schedule per band
    poly = {"degrees": [1, 2], "coefs": [0.5, 1.0], "shifts": [0, 50]}
    t = Trainer(make_cfg(poly=poly), obs_dim=3, action_dim=1)
    head = t.spec_heads[0]
    sched = LambdaSchedule(kind="cuberoot", lam0=1e-3, t0=100, floor=0.0)
    for tt in (0, 200):
        expect = (0.5 * sched(tt) * head.w2
                  + 1.0 * sched(tt + 50) * head.w2.pow(2))
        got = t._spectral_band_weights(head, tt)
        assert torch.allclose(got, expect, rtol=1e-6), tt
    # weights move over t, and the shift changes them at fixed t
    w0, w200 = t._spectral_band_weights(head, 0), t._spectral_band_weights(head, 200)
    assert not torch.allclose(w0, w200)
    t_noshift = Trainer(make_cfg(poly={"degrees": [1, 2], "coefs": [0.5, 1.0],
                                       "shifts": [0, 0]}), obs_dim=3, action_dim=1)
    assert not torch.allclose(w0, t_noshift._spectral_band_weights(
        t_noshift.spec_heads[0], 0))


def test_spectral_resume_bitwise_identical(tmp_path):
    """save -> resume -> identical next model_update losses (spectral cfg)."""
    cfg = make_cfg()
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    for i in range(3):  # refit fires at update 2 -> heads carry fitted c
        t1.model_update(fake_batch(seed=300 + i))
    assert t1.spec_refits >= 1
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    m_ref = t1.model_update(fake_batch(seed=303))

    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    assert t2.spec_refits == t1.spec_refits  # m_ref's update didn't refit
    m_resumed = t2.model_update(fake_batch(seed=303))

    for k in ("loss/dyn", "loss/reward", "loss/total", "penalty/value"):
        assert m_resumed[k] == pytest.approx(m_ref[k], rel=1e-6), k


def test_spectral_snr_generator_and_ema_survive_resume(tmp_path):
    """The SNR split-half generator AND the per-head Wiener-weight EMA must round-trip
    through a checkpoint. They were missing from state_dict, so a resumed spectral run
    re-seeded the generator (drawing a different split-half permutation on the next
    refit) and restarted the EMA from None (a silent reward discontinuity in snr mode).
    The bitwise-resume test above doesn't catch it because its compared step doesn't
    refit, so it never re-consumes the generator."""
    cfg = make_cfg(weights_mode="snr", snr_bands=4, snr_ema=0.5,
                   sigma_w=[0.25, 0.5, 1.0, 2.0])
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    for i in range(8):  # advance past a refit: generator consumed, EMA populated
        t1.model_update(fake_batch(seed=300 + i))
    assert t1.spec_refits >= 1 and t1.spec_snr_ema[0] is not None
    gen_at_save = t1.spec_snr_gen.get_state().clone()
    ema_at_save = [x.clone() if x is not None else None for x in t1.spec_snr_ema]

    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=800, tag="step8")

    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 800
    # generator restored bit-for-bit (a fresh Trainer re-seeds to seed+777, which differs)
    assert torch.equal(t2.spec_snr_gen.get_state(), gen_at_save)
    # Wiener-weight EMA restored per head
    for got, want in zip(t2.spec_snr_ema, ema_at_save):
        assert (got is None) == (want is None)
        if want is not None:
            assert torch.equal(got.cpu(), want.cpu())


def test_policy_gradients_flow_through_spectral_reward():
    torch.manual_seed(0)
    t = Trainer(make_cfg(), obs_dim=3, action_dim=1)
    for i in range(4):  # ensure a refit happened (c != 0)
        t.model_update(fake_batch(seed=400 + i))
    assert t.spec_refits >= 1

    # direct: predict builds a torch graph back to z and a
    k = t.spec_heads[0].in_dim - 1  # latent dim (action_dim=1, task_dim=0)
    z = torch.randn(16, k, requires_grad=True)
    a = torch.randn(16, 1, requires_grad=True)
    r_im, _ = t._imagined_reward(z, a)
    assert r_im.grad_fn is not None
    r_im.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert a.grad is not None and a.grad.abs().sum() > 0

    # end to end: behaviour_update leaves nonzero policy gradients
    z0 = t.encoder(fake_batch(seed=555)[0]).detach()
    b = t.behaviour_update(z0)
    assert np.isfinite(b["loss/policy"])
    gsum = sum(p.grad.abs().sum().item() for p in t.policy.parameters()
               if p.grad is not None)
    assert gsum > 0


def test_sigma_ladder_trainer_smoke():
    """sigma_w as an OmegaConf list (the spectral_ladder preset's run-3 recipe)
    flows through Trainer construction, refit, and stays finite."""
    torch.manual_seed(0)
    cfg = make_cfg(poly={"degrees": [1, 3], "coefs": [0.1, 10.0], "shifts": [0, 0]},
                   sigma_w=[0.25, 0.5, 1.0, 2.0])
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    for h in t.spec_heads:  # ladder applied: bands genuinely separated
        assert (h.w2.sqrt().max() / h.w2.sqrt().min()) > 4.0
    for i in range(6):
        m = t.model_update(fake_batch(seed=200 + i))
        assert np.isfinite(m["loss/total"])
    assert t.spec_refits >= 1
    assert t._spectral_penalty_value() >= 0.0


def test_snr_weights_mode_trainer_smoke():
    """weights_mode=snr: refits use measured Wiener weights, EMA state exists,
    metrics expose the SNR diagnostics, everything stays finite."""
    torch.manual_seed(0)
    cfg = make_cfg(weights_mode="snr", snr_bands=4, snr_ema=0.5,
                   sigma_w=[0.25, 0.5, 1.0, 2.0])
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    last = {}
    for i in range(8):
        last = t.model_update(fake_batch(seed=300 + i))
        assert np.isfinite(last["loss/total"])
    assert t.spec_refits >= 1
    assert t.spec_snr_ema[0] is not None and torch.isfinite(t.spec_snr_ema[0]).all()
    assert "spectral/snr_min" in last and last["spectral/snr_min"] > 0


def test_sigma_auto_calibrates_and_resumes():
    """sigma_w='auto': heads absent until the first refit, calibration sets
    the ladder + sigma_star, the run proceeds finite, and a save -> fresh
    Trainer -> load round-trip restores the calibrated basis exactly."""
    torch.manual_seed(0)
    cfg = make_cfg(sigma_w="auto", n_features=64)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    assert t.spec_heads == [] and t.spec_sigma == "auto"
    last = {}
    for i in range(8):
        last = t.model_update(fake_batch(seed=400 + i))
        assert np.isfinite(last["loss/total"])
    assert t.spec_refits >= 1 and len(t.spec_heads) == 2
    assert t.spec_sigma_star is not None and t.spec_sigma_star > 0
    assert isinstance(t.spec_sigma, list) and len(t.spec_sigma) == 4
    assert last.get("spectral/sigma_star") == t.spec_sigma_star

    sd = t.state_dict()
    t2 = Trainer(make_cfg(sigma_w="auto", n_features=64), obs_dim=3, action_dim=1)
    assert t2.spec_heads == []      # fresh instance, uncalibrated
    t2.load_state_dict(sd)
    assert len(t2.spec_heads) == 2
    for h1, h2 in zip(t.spec_heads, t2.spec_heads):
        assert torch.equal(h1.W, h2.W) and torch.equal(h1.c, h2.c)
    assert t2.spec_sigma_star == t.spec_sigma_star


def test_learned_sigma_scales_move_and_resume():
    """sigma_w='learned': per-block log-scales get gradient steps from the
    reward fit error after the first refit, metrics expose them, and the
    save/load roundtrip restores scales + basis exactly."""
    torch.manual_seed(0)
    cfg = make_cfg(sigma_w="learned", n_features=64,
                   init_ladder=[0.25, 0.5, 1.0, 2.0], sigma_lr=1e-2)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    s0 = t.spec_heads[0].log_s.detach().clone()
    last = {}
    for i in range(10):
        last = t.model_update(fake_batch(seed=500 + i))
        assert np.isfinite(last["loss/total"])
    assert t.spec_refits >= 1
    assert not torch.allclose(t.spec_heads[0].log_s.detach(), s0)  # pipes moved
    assert "spectral/sigma_scale_0" in last

    sd = t.state_dict()
    t2 = Trainer(make_cfg(sigma_w="learned", n_features=64,
                          init_ladder=[0.25, 0.5, 1.0, 2.0], sigma_lr=1e-2),
                 obs_dim=3, action_dim=1)
    t2.load_state_dict(sd)
    assert torch.allclose(t2.spec_heads[0].log_s.detach(),
                          t.spec_heads[0].log_s.detach())
    assert torch.equal(t2.spec_heads[0].W_base, t.spec_heads[0].W_base)


def test_gaussian_dynamics_trainer_smoke():
    """model.dynamics=gaussian: NLL trains, imagination rolls out stochastic
    rsamples, everything finite; mean path stays affine in action."""
    torch.manual_seed(0)
    cfg = make_cfg()
    cfg.model.dynamics = "gaussian"
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    from mbrl.models import GaussianAffineDynamics
    assert isinstance(t.dynamics, GaussianAffineDynamics)
    for i in range(8):
        m = t.model_update(fake_batch(seed=600 + i))
        assert np.isfinite(m["loss/dyn"])
        # run-9 calibration telemetry present and finite on the gaussian path
        assert "dyn/calib_corr" in m and np.isfinite(m["dyn/calib_corr"])
        assert m["dyn/pred_std"] > 0
    # stochastic forward: two rollout steps from the same (z, a) differ
    z = torch.randn(5, t.encoder.latent_dim)
    a = torch.randn(5, 1)
    assert not torch.equal(t.dynamics(z, a), t.dynamics(z, a))
    # mean is deterministic and affine path is intact
    assert torch.equal(t.dynamics.mean(z, a), t.dynamics.mean(z, a))
    b = t.behaviour_update(torch.randn(16, t.encoder.latent_dim))
    assert np.isfinite(b["loss/policy"])


def test_full_mlp_dynamics_ablation_arm():
    """model.dynamics=mlp (run-9 R15 ablation): constructs with a loud
    warning, trains finite; deliberately NOT affine in action."""
    import warnings
    torch.manual_seed(0)
    cfg = make_cfg()
    cfg.model.dynamics = "mlp"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = Trainer(cfg, obs_dim=3, action_dim=1)
    assert any("R15" in str(x.message) for x in w)
    from mbrl.models import FullMLPDynamics
    assert isinstance(t.dynamics, FullMLPDynamics)
    for i in range(4):
        m = t.model_update(fake_batch(seed=700 + i))
        assert np.isfinite(m["loss/dyn"])


def test_encoder_aux_grounds_encoder_in_spectral_mode():
    """Spectral mode trains the encoder through the aux reward loss (collapse
    fix, 2026-06-08): with encoder_aux on, metrics expose aux_loss + z_std and
    the encoder moves MORE than in the aux-off ablation."""
    def run(aux):
        torch.manual_seed(0)
        cfg = make_cfg(encoder_aux=aux)
        t = Trainer(cfg, obs_dim=3, action_dim=1)
        w0 = [p.detach().clone() for p in t.encoder.parameters()]
        last = {}
        for i in range(8):
            last = t.model_update(fake_batch(seed=800 + i))
        delta = sum((p - q).abs().sum().item()
                    for p, q in zip(t.encoder.parameters(), w0))
        return last, delta
    on, d_on = run(True)
    off, d_off = run(False)
    assert "latent/z_std" in on and on["latent/z_std"] > 0
    assert "spectral/aux_loss" in on and np.isfinite(on["spectral/aux_loss"])
    assert "spectral/aux_loss" not in off
    assert d_on > d_off    # reward gradient reaches the encoder


def test_vae_encoder_run10_smoke():
    """encoder=vae: recon+KL train alongside the champion stack; EMA targets
    are deterministic (mu); metrics expose vae/recon + vae/kl; recon falls."""
    torch.manual_seed(0)
    cfg = make_cfg(sigma_w="auto", n_features=64)
    cfg.model.encoder = "vae"
    cfg.model.dynamics = "gaussian"
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    from mbrl.models import VAEEncoder
    assert isinstance(t.encoder, VAEEncoder)
    assert t.ema.ema.deterministic is True       # mu targets, no noise leak
    recons = []
    for i in range(10):
        m = t.model_update(fake_batch(seed=900 + i))
        assert np.isfinite(m["loss/total"])
        if "vae/recon" in m:
            recons.append(m["vae/recon"])
    assert len(recons) >= 8 and np.isfinite(recons).all()
    assert recons[-1] < recons[0]                # reconstruction is learning
    # checkpoint roundtrip carries the decoder
    sd = t.state_dict()
    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    t2.load_state_dict(sd)
    x = torch.randn(4, 3)
    mu1, lv1 = t.encoder.moments(x)
    mu2, lv2 = t2.encoder.moments(x)
    assert torch.allclose(mu1, mu2) and torch.allclose(lv1, lv2)
    assert torch.allclose(t.encoder.decoder(mu1), t2.encoder.decoder(mu2))
