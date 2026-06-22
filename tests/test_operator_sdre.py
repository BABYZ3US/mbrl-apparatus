"""OperatorSDRE — the State-Dependent Riccati Equation regulator on the
OperatorDynamics local linearization (Çimen 2008; Anderson & Moore 1990; Kalman
1960b). Pins: the operator z'=A(z)z+B(z)a IS the linear plant, so the per-state
LQR is exact-per-state; the backward Riccati recursion is batched over N; gain
returns K:[N,m,k]; the closed loop A−B@K is STABLE (spectral radius < 1) on a
stable pair; and act() returns a clamped action a:[N,m] in [-1,1]. The operator
is a fake monkeypatched object with fixed (A, B) — no torch model, no RNG."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.planning.operator_sdre import OperatorSDRE
from mbrl.utils.seeding import seed_everything


class _FakeOperator:
    """Stand-in for OperatorDynamics: .operators(z) returns a FIXED stable pair
    A=[[0.9,0.1],[0,0.9]], B=[[0.0],[1.0]] (k=2, m=1), batched to z's N. No nn,
    no params — isolates the SDRE solve from any learned dynamics."""

    A0 = torch.tensor([[0.9, 0.1], [0.0, 0.9]])
    B0 = torch.tensor([[0.0], [1.0]])

    def operators(self, z):
        n = z.shape[0]
        A = self.A0.to(z).expand(n, 2, 2)
        B = self.B0.to(z).expand(n, 2, 1)
        return A, B


def _AB(n=8):
    op = _FakeOperator()
    z = torch.zeros(n, 2)
    return op.operators(z)


def test_gain_shape_is_N_m_k():
    seed_everything(0)
    A, B = _AB(8)                                   # A:[8,2,2], B:[8,2,1]
    K = OperatorSDRE(horizon=30).gain(A, B)
    assert K.shape == (8, 1, 2)                     # [N, m, k]
    assert torch.isfinite(K).all()


def test_closed_loop_is_stable():
    """The SDRE point: -K z drives the stable open-loop pair to a CLOSED loop
    A−B@K whose spectral radius is < 1 (max |eigenvalue| < 1)."""
    seed_everything(0)
    A, B = _AB(4)
    K = OperatorSDRE(horizon=30).gain(A, B)
    cl = A - B @ K                                  # [4,2,2] closed-loop A−BK
    rho = torch.linalg.eigvals(cl).abs().amax(dim=-1)   # spectral radius per sample
    assert (rho < 1.0).all()
    # open loop already had |λ|=0.9<1; the regulator must not DEstabilize it
    assert (rho <= 0.9 + 1e-5).all()


def test_act_returns_clamped_action():
    seed_everything(0)
    op = _FakeOperator()
    z = torch.randn(8, 2)
    a = OperatorSDRE(horizon=30).act(z, op)
    assert a.shape == (8, 1)                        # [N, m]
    assert torch.isfinite(a).all()
    assert a.abs().max().item() <= 1.0 + 1e-6       # clamped to [-1,1]


def test_act_zero_state_zero_action():
    """At z = z_ref the error is 0, so a = -K·0 = 0 (the regulator is at rest)."""
    seed_everything(0)
    op = _FakeOperator()
    a = OperatorSDRE(horizon=20).act(torch.zeros(5, 2), op)
    assert torch.allclose(a, torch.zeros(5, 1), atol=1e-7)


def test_z_ref_recenters_the_regulator():
    """a = -K (z - z_ref): at z == z_ref the action vanishes regardless of where
    z_ref sits, confirming the reference enters only through the error."""
    seed_everything(0)
    op = _FakeOperator()
    z_ref = torch.randn(6, 2)
    a = OperatorSDRE(horizon=20).act(z_ref.clone(), op, z_ref=z_ref)
    assert torch.allclose(a, torch.zeros(6, 1), atol=1e-6)


def test_gain_default_costs_match_explicit_identity():
    """Default Q/R (q_weight·I, r_weight·I) must equal passing those matrices."""
    seed_everything(0)
    A, B = _AB(4)
    s = OperatorSDRE(horizon=25, q_weight=1.0, r_weight=1.0)
    K_default = s.gain(A, B)
    K_explicit = s.gain(A, B, Q=torch.eye(2), R=torch.eye(1))
    assert torch.allclose(K_default, K_explicit, atol=1e-6)


def test_gain_is_deterministic():
    """No RNG / no global state: identical inputs give bitwise-equal gains."""
    seed_everything(0)
    A, B = _AB(4)
    s = OperatorSDRE(horizon=30)
    assert torch.equal(s.gain(A, B), s.gain(A, B))


def test_riccati_converges_to_fixed_point():
    """A long finite horizon reaches the DARE fixed point: a much longer solve
    yields the same steady gain (P stopped moving) — the recursion converged."""
    seed_everything(0)
    A, B = _AB(4)
    K_short = OperatorSDRE(horizon=40, riccati_iters=40).gain(A, B)
    K_long = OperatorSDRE(horizon=200, riccati_iters=200).gain(A, B)
    assert torch.allclose(K_short, K_long, atol=1e-6)


def test_higher_r_weight_softens_the_gain():
    """Larger control cost R penalizes actuation, so the feedback gain shrinks."""
    seed_everything(0)
    A, B = _AB(4)
    K_cheap = OperatorSDRE(horizon=40, r_weight=1.0).gain(A, B)
    K_pricey = OperatorSDRE(horizon=40, r_weight=100.0).gain(A, B)
    assert K_pricey.abs().sum().item() < K_cheap.abs().sum().item()
