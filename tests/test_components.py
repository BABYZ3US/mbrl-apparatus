"""Battle-tested component units: TwinQ, SquashedGaussianPolicy,
EnsembleAffineDynamics (R15-safe), CEM planner. Analytic pins, not smoke."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.models.critics import TwinQ, SquashedGaussianPolicy
from mbrl.models.ensemble import EnsembleAffineDynamics
from mbrl.planning import cem_plan


# ---------------- TwinQ ----------------

def test_twin_q_shapes_and_min():
    torch.manual_seed(0)
    q = TwinQ(latent_dim=6, action_dim=2, hidden=32, depth=2)
    z, a = torch.randn(7, 6), torch.randn(7, 2)
    q1, q2 = q(z, a)
    assert q1.shape == (7,) and q2.shape == (7,)
    assert not torch.allclose(q1, q2)          # independent heads (different init)
    m = q.min_q(z, a)
    assert torch.all(m <= q1 + 1e-6) and torch.all(m <= q2 + 1e-6)


def test_twin_q_task_conditioning():
    q = TwinQ(latent_dim=4, action_dim=2, hidden=16, depth=1, task_dim=3)
    z, a, tau = torch.randn(5, 4), torch.randn(5, 2), torch.randn(5, 3)
    q1, _ = q(z, a, tau)
    assert q1.shape == (5,)


# ---------------- SquashedGaussianPolicy ----------------

def test_squashed_actions_are_bounded():
    torch.manual_seed(0)
    pi = SquashedGaussianPolicy(latent_dim=5, action_dim=3, hidden=32, action_scale=2.0)
    a, logp = pi.sample(torch.randn(64, 5))
    assert a.shape == (64, 3) and logp.shape == (64,)
    assert torch.all(a.abs() < 2.0)            # strictly inside (-scale, scale)
    d = pi.deterministic(torch.randn(8, 5))
    assert torch.all(d.abs() < 2.0)


def test_squashed_logprob_matches_change_of_variables():
    """Pin log pi against the analytic formula on a forced (mu, std, u)."""
    torch.manual_seed(0)
    pi = SquashedGaussianPolicy(latent_dim=2, action_dim=1, hidden=8, depth=1)
    z = torch.randn(1, 2)
    mu, log_std = pi(z)
    std = log_std.exp()
    torch.manual_seed(7)                       # freeze the rsample noise
    eps = torch.randn_like(mu)
    u = mu + std * eps
    a = torch.tanh(u)
    base = -0.5 * (eps ** 2 + 2 * log_std + math.log(2 * math.pi))
    want = (base - torch.log(1 - a ** 2 + 1e-6)).sum(-1)
    torch.manual_seed(7)                       # same noise inside sample()
    _, logp = pi.sample(z)
    assert torch.allclose(logp, want, atol=1e-5)


def test_squashed_rsample_carries_gradients():
    pi = SquashedGaussianPolicy(latent_dim=3, action_dim=2, hidden=16, depth=1)
    z = torch.randn(4, 3)
    a, _ = pi.sample(z)
    a.sum().backward()
    grads = [p.grad for p in pi.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


# ---------------- EnsembleAffineDynamics (R15-safe) ----------------

def test_ensemble_members_are_affine_in_action():
    """R15: d2(z')/da2 = 0 — check via the affinity identity
    f(z, a1) + f(z, a2) - f(z, 0) == f(z, a1 + a2) for every member."""
    torch.manual_seed(0)
    ens = EnsembleAffineDynamics(latent_dim=4, action_dim=2, n_members=3, hidden=16, depth=1)
    z = torch.randn(6, 4)
    a1, a2 = torch.randn(6, 2), torch.randn(6, 2)
    zero = torch.zeros(6, 2)
    for m in range(3):
        lhs = (ens.member_rollout_step(m, z, a1) + ens.member_rollout_step(m, z, a2)
               - ens.member_rollout_step(m, z, zero))
        rhs = ens.member_rollout_step(m, z, a1 + a2)
        assert torch.allclose(lhs, rhs, atol=1e-4)


def test_ensemble_disagreement_positive_and_mean_shape():
    torch.manual_seed(0)
    ens = EnsembleAffineDynamics(latent_dim=4, action_dim=2, n_members=4, hidden=16, depth=1)
    z, a = torch.randn(5, 4), torch.randn(5, 2)
    assert ens(z, a).shape == (5, 4)
    dis = ens.disagreement(z, a)
    assert dis.shape == (5,)
    assert torch.all(dis > 0)                  # independent inits really disagree


def test_ensemble_rejects_single_member():
    with pytest.raises(ValueError):
        EnsembleAffineDynamics(latent_dim=4, action_dim=2, n_members=1)


# ---------------- CEM planner ----------------

def test_cem_recovers_known_optimum():
    """Score = -|seq - target|^2 -> CEM must land on the target sequence."""
    target = torch.tensor([[0.3, -0.6], [0.8, 0.1], [-0.4, 0.5]])   # (H=3, A=2)

    def score(cand):                            # (pop, H, A) -> (pop,)
        return -((cand - target) ** 2).sum(dim=(1, 2))

    g = torch.Generator().manual_seed(42)
    best = cem_plan(score, horizon=3, action_dim=2, iters=8, pop=512,
                    elites=48, generator=g)
    assert torch.allclose(best, target, atol=0.05)


def test_cem_respects_bounds_and_determinism():
    def score(cand):
        return cand.sum(dim=(1, 2))             # push toward the upper bound

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    b1 = cem_plan(score, horizon=2, action_dim=1, iters=4, pop=64, elites=8, generator=g1)
    b2 = cem_plan(score, horizon=2, action_dim=1, iters=4, pop=64, elites=8, generator=g2)
    assert torch.equal(b1, b2)                  # same seed -> same plan (no global RNG)
    assert torch.all(b1 <= 1.0) and torch.all(b1 >= -1.0)
    assert torch.all(b1 > 0.8)                  # found the upper-bound optimum


def test_cem_rejects_bad_elites():
    with pytest.raises(ValueError):
        cem_plan(lambda c: c.sum(dim=(1, 2)), horizon=2, action_dim=1, pop=8, elites=9)