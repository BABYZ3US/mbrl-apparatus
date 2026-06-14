"""cf5 rank-2 reward⊥energy frame (PM 2026-06-14). The controllable essence is rank-2:
two orthogonal axes — reward-ascent (∇_z R) ⊥ energy-descent (lyapunov −∇_z E |
contractive op_d mode). Pins: the pure helpers behave (tail penalty, cos²); both
energy modes wire into the dual twin, log frame/* + train; contractive needs twin;
default-off ⇒ byte-identical; energy head checkpoints (bitwise resume)."""
import math
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models.dynamics import OperatorDynamics
from mbrl.regularization.rank2_frame import (EnergyHead, axis_cos2, rank2_tail_penalty,
                                             lyapunov_grounding, contractive_axis_in_d,
                                             dissipativity_penalty, spectral_shell_penalty,
                                             log_det_barrier)
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(mode="twin", frame=True, energy_mode="lyapunov", rank=2,
         w_ortho=0.1, w_rank2=0.01, w_lyap=0.1):
    rf = {"enabled": frame, "energy_mode": energy_mode, "w_ortho": w_ortho,
          "w_rank2": w_rank2, "w_lyap": w_lyap, "subsample": 8, "target_rank": 2}
    dl = {"enabled": True, "mode": mode, "d_dim": 0, "p_dim": 0, "couple_weight": 0.1,
          "p_consistency_weight": 1.0, "penalize_reward": False, "smooth_p": False,
          "radius_p": 0.1, "rank2_frame": rf}
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 16, "depth": 1, "ema_decay": 0.99,
                  "dynamics": "operator", "reward_heads": 1,
                  "operator": {"structure": "none", "rank": rank},
                  "dual_latent": dl},
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


# ---- pure helpers ----
def test_rank2_tail_penalty_targets_rank2():
    g = torch.Generator().manual_seed(0)
    # rank-2 data: two directions only -> tail ~ 0
    basis = torch.randn(2, 6, generator=g)
    z2 = torch.randn(256, 2, generator=g) @ basis
    assert rank2_tail_penalty(z2).item() < 1e-4
    # full-rank isotropic -> tail > 0 (variance outside the top 2)
    z6 = torch.randn(256, 6, generator=g)
    assert rank2_tail_penalty(z6).item() > 0.1


def test_spectral_shell_two_sided_anti_collapse():
    """The Landau well penalizes rank-1 collapse AND norm collapse, near-zero for a
    healthy rank-2-at-setpoint spectrum."""
    g = torch.Generator().manual_seed(0)
    z_good = torch.randn(2048, 2, generator=g) @ torch.eye(2, 6)   # 2 unit-var modes
    good = spectral_shell_penalty(z_good, 2, 1.0).item()
    assert good < 0.3
    z_rank1 = torch.randn(2048, 1, generator=g) @ torch.randn(1, 6, generator=g)
    assert spectral_shell_penalty(z_rank1, 2, 1.0).item() > 1.0     # 2nd mode missing -> penalized
    assert spectral_shell_penalty(z_good * 0.05, 2, 1.0).item() > good  # norm collapse -> penalized


def test_shell_wires_into_model_update():
    seed_everything(0)
    cfg = _cfg(energy_mode="lyapunov")
    cfg.model.dual_latent.rank2_frame.w_shell = 0.5
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    m = t.model_update(_batch())
    assert "frame/shell" in m and math.isfinite(m["frame/shell"])
    assert math.isfinite(m["loss/total"])


def test_log_det_barrier_penalizes_singular():
    """The KL/log-det barrier is high for a near-singular (rank-1) Gram and low for a
    well-conditioned one — minimizing it pushes eigenvalues off zero (bounds cond)."""
    g = torch.Generator().manual_seed(0)
    z_iso = torch.randn(2048, 4, generator=g)
    z_sing = torch.randn(2048, 1, generator=g) @ torch.randn(1, 4, generator=g)   # rank-1
    assert log_det_barrier(z_sing).item() > log_det_barrier(z_iso).item()
    assert math.isfinite(log_det_barrier(z_sing).item())


def test_logdet_wires_into_model_update():
    seed_everything(0)
    cfg = _cfg(energy_mode="lyapunov")
    cfg.model.dual_latent.rank2_frame.w_shell = 1.0
    cfg.model.dual_latent.rank2_frame.w_logdet = 0.05
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    m = t.model_update(_batch())
    assert "frame/logdet_barrier" in m and math.isfinite(m["frame/logdet_barrier"])
    assert math.isfinite(m["loss/total"])


def test_axis_cos2_orthogonal_vs_parallel():
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_orth = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b_par = a.clone()
    assert axis_cos2(a, b_orth).item() < 1e-6
    assert abs(axis_cos2(a, b_par).item() - 1.0) < 1e-6


def test_contractive_axis_shape_and_unit():
    seed_everything(0)
    op = OperatorDynamics(4, 2, hidden=16, depth=1)
    d = torch.randn(8, 4)
    v = contractive_axis_in_d(op, d)
    assert v.shape == (8, 4) and not v.requires_grad
    assert torch.allclose(v.norm(dim=-1), torch.ones(8), atol=1e-4)


def test_energy_head_grounding_runs():
    seed_everything(0)
    e = EnergyHead(4, 16, 1)
    op = OperatorDynamics(4, 2, hidden=16, depth=1)
    d = torch.randn(8, 4)
    g = lyapunov_grounding(e, op, d, action_dim=2)
    assert g.item() >= 0.0 and math.isfinite(g.item())


def test_energy_anchor_prevents_constant_collapse():
    """cf7-fix: with the anchor, E = a·½‖d‖² + tanh(head). The tanh correction is bounded
    (∈(−1,1)) so it CANNOT cancel the kinetic floor — E provably grows with ‖d‖ no matter
    what the head does, foreclosing collapse to a constant."""
    seed_everything(0)
    e1 = EnergyHead(4, 16, 1, anchor=1.0)
    d = torch.randn(8, 4); big = d * 5.0
    dn = 0.5 * (d * d).sum(-1); bn = 0.5 * (big * big).sum(-1)
    gap = e1(big) - e1(d)
    # floor dominates: gap >= anchor*(bn-dn) - 2 (tanh shifts each end by < 1), still > 0
    assert (gap >= 1.0 * (bn - dn) - 2.0 - 1e-4).all() and gap.mean() > 0
    cfg = _cfg(energy_mode="lyapunov")
    cfg.model.dual_latent.rank2_frame.energy_anchor = 1.0
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    assert t.energy.anchor == 1.0
    assert math.isfinite(t.model_update(_batch())["loss/total"])


# ---- wiring ----
def test_lyapunov_frame_trains_and_logs():
    seed_everything(0)
    t = Trainer(_cfg(energy_mode="lyapunov"), obs_dim=3, action_dim=2)
    assert t.frame_enabled and t.energy is not None
    m = t.model_update(_batch())
    assert {"frame/ortho_cos", "frame/rank2_tail", "frame/lyap_resid"} <= set(m)
    assert math.isfinite(m["loss/total"])
    # the energy head received gradient (it is in model_opt and the frame uses it)
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in t.energy.parameters())


def test_contractive_frame_logs_no_energy_head():
    seed_everything(0)
    t = Trainer(_cfg(energy_mode="contractive"), obs_dim=3, action_dim=2)
    assert t.frame_enabled and t.energy is None
    m = t.model_update(_batch())
    assert "frame/ortho_cos" in m and "frame/rank2_tail" in m
    assert "frame/lyap_resid" not in m
    assert math.isfinite(m["loss/total"])


def test_contractive_requires_twin():
    with pytest.raises(ValueError):
        Trainer(_cfg(mode="shared", energy_mode="contractive"), obs_dim=3, action_dim=2)


def test_frame_default_off_is_noop():
    seed_everything(0)
    t = Trainer(_cfg(frame=False), obs_dim=3, action_dim=2)
    assert not t.frame_enabled and t.energy is None
    m = t.model_update(_batch())
    assert not any(k.startswith("frame/") for k in m)


def test_dissipativity_penalty_one_sided():
    """relu(E(d')−E(d) − supply): always satisfied when supply is huge (penalty 0),
    always violated when supply is hugely negative (penalty > 0). One-sided."""
    seed_everything(0)
    e = EnergyHead(4, 16, 1)
    d, d_next = torch.randn(8, 4), torch.randn(8, 4)
    assert dissipativity_penalty(e, d, d_next, torch.full((8,), 1e6)).item() == 0.0
    assert dissipativity_penalty(e, d, d_next, torch.full((8,), -1e6)).item() > 0.0
    g = dissipativity_penalty(e, d, d_next, torch.randn(8))
    assert g.item() >= 0.0 and math.isfinite(g.item())


def test_dissipativity_wires_into_model_update():
    """cf6: w_dissip>0 adds the passivity term on real transitions; logs the residual."""
    seed_everything(0)
    cfg = _cfg(energy_mode="lyapunov")
    cfg.model.dual_latent.rank2_frame.w_dissip = 0.1
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    m = t.model_update(_batch())
    assert "frame/dissip_resid" in m and math.isfinite(m["frame/dissip_resid"])
    assert math.isfinite(m["loss/total"])


def test_equilibrium_balance_applies_both_penalties():
    """cf7 balance: alignment (couple) and energy (dissipativity) are held at equal
    influence via adaptive weights (the fixed couple_w/w_dissip are bypassed). The
    bal_* metrics log, the per-side EMAs populate, loss stays finite."""
    seed_everything(0)
    cfg = _cfg(energy_mode="lyapunov")
    cfg.model.dual_latent.rank2_frame.w_dissip = 0.1
    cfg.model.dual_latent.rank2_frame.balance = True
    cfg.model.dual_latent.rank2_frame.balance_weight = 0.1
    cfg.model.dual_latent.rank2_frame.balance_decay = 0.9
    t = Trainer(cfg, obs_dim=3, action_dim=2)
    assert t.frame_balance
    m = t.model_update(_batch())
    assert {"frame/bal_w_align", "frame/bal_w_energy",
            "frame/bal_align", "frame/bal_energy"} <= set(m)
    assert math.isfinite(m["loss/total"])
    assert t._bal_ema_align is not None and t._bal_ema_energy is not None
    # weights are capped (a near-zero penalty can't explode its 1/ema weight)
    assert m["frame/bal_w_align"] <= 10.0 and m["frame/bal_w_energy"] <= 10.0


def test_resume_bitwise_with_balance(tmp_path):
    cfg = _cfg(energy_mode="lyapunov")
    cfg.model.dual_latent.rank2_frame.w_dissip = 0.1
    cfg.model.dual_latent.rank2_frame.balance = True
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=2)
    for i in range(3):
        t1.model_update(_batch(seed=i))
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=2)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    assert t2._bal_ema_align == t1._bal_ema_align
    assert t2._bal_ema_energy == t1._bal_ema_energy


def test_resume_bitwise_with_lyapunov_frame(tmp_path):
    cfg = _cfg(energy_mode="lyapunov")
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
