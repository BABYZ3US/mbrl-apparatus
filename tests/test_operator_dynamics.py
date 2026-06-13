"""OperatorDynamics — the dual-latent controlled-operator model's dynamics leg
(z' = A(z)z + B(z)a). Pins: R15 is PRESERVED (still affine in a, ∂²z'/∂a²=0, so
NO dynamics-curvature floor — this is what makes it a legal drop-in vs the run-9
FullMLPDynamics ablation); A = I+Â near-identity init; the structural priors are
well-formed and finite-grad even at the degenerate (near-identity) init; symmetric
⇒ normal-residual 0; rank=r ⇒ A−I has rank ≤ r; and the Trainer wires it as a
drop-in (op/* diagnostics logged, bitwise resume holds — no new RNG/state)."""
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models.dynamics import OperatorDynamics
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything


def _cfg(structure="none", rank=0, **w):
    op = {"structure": structure, "rank": rank,
          "w_normal": 0.0, "w_smooth": 0.0, "w_spread": 0.0, "w_radius": 0.0}
    op.update(w)
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "dynamics": "operator", "operator": op},
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


def test_shapes_and_drop_in_forward():
    seed_everything(0)
    d = OperatorDynamics(4, 2, hidden=16, depth=1)
    z, a = torch.randn(8, 4), torch.randn(8, 2)
    A, B = d.operators(z)
    assert A.shape == (8, 4, 4) and B.shape == (8, 4, 2)
    assert d(z, a).shape == (8, 4)                      # same interface as AffineDynamics


def test_r15_affine_in_action_no_curvature_floor():
    """The whole point: z' is LINEAR in a (B(z) action-independent), so
    ∂²z'/∂a² = 0 — the operator form reintroduces NO dynamics-curvature floor."""
    seed_everything(0)
    d = OperatorDynamics(4, 2, hidden=16, depth=1)
    z = torch.randn(8, 4)
    a = torch.randn(8, 2)
    z0 = d(z, torch.zeros_like(a))
    assert torch.allclose(d(z, 2 * a) - z0, 2 * (d(z, a) - z0), atol=1e-5)   # linearity in a
    # second derivative wrt a is identically zero (the R15 property)
    a_ = a.clone().requires_grad_(True)
    g = torch.autograd.grad(d(z, a_).sum(), a_, create_graph=True)[0]
    h = torch.autograd.grad(g.sum(), a_, retain_graph=True, allow_unused=True)[0]
    assert h is None or torch.allclose(h, torch.zeros_like(a_), atol=1e-6)


def test_structural_penalties_finite_grad_at_init():
    """Degenerate-init guard: A≈I means near-repeated singular values; the svd
    backward must still be finite on step 1 (else the spread/radius priors NaN)."""
    seed_everything(0)
    d = OperatorDynamics(4, 2, hidden=16, depth=1)
    z = torch.randn(16, 4)
    sp = d.structural_penalties(z)
    assert set(sp) == {"normal", "smooth", "spread", "radius"}
    assert all(torch.isfinite(v) for v in sp.values())
    (sp["normal"] + sp["smooth"] + sp["spread"] + sp["radius"]).backward()
    grads = [p.grad for p in d.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_symmetric_structure_is_normal():
    """structure=symmetric ⇒ A=Aᵀ ⇒ [A,Aᵀ]=0 ⇒ normal residual vanishes."""
    seed_everything(0)
    d = OperatorDynamics(4, 2, hidden=16, depth=1, structure="symmetric")
    z = torch.randn(16, 4)
    assert d.structural_penalties(z)["normal"].item() < 1e-8


def test_low_rank_operator_bounds_rank():
    """rank=r ⇒ A = U V ᵀ + I ⇒ A−I has at most r nonzero singular values."""
    seed_everything(0)
    d = OperatorDynamics(6, 2, hidden=16, depth=1, rank=2)
    z = torch.randn(5, 6)
    A, _ = d.operators(z)
    res = A - torch.eye(6)
    sv = torch.linalg.svdvals(res)                      # (5, 6)
    assert sv[:, 2:].abs().max().item() < 1e-5


def test_spectral_summary_keys_finite():
    seed_everything(0)
    d = OperatorDynamics(4, 2, hidden=16, depth=1)
    s = d.spectral_summary(torch.randn(16, 4))
    assert {"op/radius", "op/eff_rank", "op/normality_resid"} <= set(s)
    assert all(math.isfinite(v) for v in s.values())


def test_trainer_drop_in_logs_spectral_diagnostics():
    seed_everything(0)
    t = Trainer(_cfg(), obs_dim=3, action_dim=2)
    assert t.dyn_operator and isinstance(t.dynamics, OperatorDynamics)
    m = t.model_update(_batch())
    assert "op/radius" in m and "op/eff_rank" in m and "op/normality_resid" in m
    assert "op/pen_normal" not in m                     # weights all 0 -> no penalty term
    t.behaviour_update(t.encoder(_batch()[0]).detach())  # imagination rolls through A(z)z+B(z)a
    assert t.act(t.encoder(_batch()[0]).detach()).shape == (16, 2)


def test_trainer_structural_penalty_terms_when_weighted():
    seed_everything(0)
    t = Trainer(_cfg(w_normal=0.1, w_smooth=0.1, w_spread=0.01, w_radius=0.1),
                obs_dim=3, action_dim=2)
    m = t.model_update(_batch())
    for k in ("normal", "smooth", "spread", "radius"):
        assert f"op/pen_{k}" in m and math.isfinite(m[f"op/pen_{k}"])


def test_resume_bitwise_identical_with_operator(tmp_path):
    cfg = _cfg(w_normal=0.1, w_smooth=0.1)
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
