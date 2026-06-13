"""Affine-in-action latent dynamics T = z + f(z) + G(z) a (framework 2.1, R15).

d^2 T / da^2 = 0 by construction: removes the dynamics-curvature floor and
empirically tightens imagined-return variance control ~2x.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

from .encoder import mlp


class AffineDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.k, self.m = latent_dim, action_dim
        self.f = mlp([latent_dim] + [hidden] * depth + [latent_dim])
        self.G = mlp([latent_dim] + [hidden] * depth + [latent_dim * action_dim])

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        G = self.G(z).view(*z.shape[:-1], self.k, self.m)
        return z + self.f(z) + (G @ a.unsqueeze(-1)).squeeze(-1)


class GaussianAffineDynamics(AffineDynamics):
    """STATE PROBABILITY TRANSITIONS p(z' | z, a) = N(mu, diag(sigma^2)) —
    not just the deterministic wave-forward (user, 2026-06-08).

    Design constraint honored: the MEAN stays affine in action
    (mu = z + f(z) + G(z) a, inherited), so d^2 mu / da^2 = 0 is preserved —
    the founding choice that removed the dynamics-curvature floor (R15).
    The variance head sigma(z) is a function of STATE ONLY (no action input),
    so the stochasticity adds no action curvature either. A full-MLP mean
    would silently reintroduce the curvature floor; deliberately not offered.

    forward() returns a reparameterized sample mu + sigma * eps — imagination
    becomes a stochastic rollout with gradients flowing through (rsample).
    Train with nll(); probe Hessians on mean() (deterministic)."""

    LOGVAR_RANGE = (-8.0, 2.0)

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__(latent_dim, action_dim, hidden, depth)
        self.logvar = mlp([latent_dim] + [hidden] * depth + [latent_dim])

    def moments(self, z: Tensor, a: Tensor) -> tuple:
        mu = AffineDynamics.forward(self, z, a)
        lv = self.logvar(z).clamp(*self.LOGVAR_RANGE)
        return mu, lv

    def mean(self, z: Tensor, a: Tensor) -> Tensor:
        return AffineDynamics.forward(self, z, a)

    def nll(self, z: Tensor, a: Tensor, target: Tensor) -> Tensor:
        """Gaussian negative log-likelihood per dim (constant dropped)."""
        mu, lv = self.moments(z, a)
        return 0.5 * ((target - mu).pow(2) * torch.exp(-lv) + lv).mean()

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        mu, lv = self.moments(z, a)
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


class OperatorDynamics(nn.Module):
    """Operator-field latent dynamics  z' = A(z) z + B(z) a  (the dual-latent
    controlled-operator model, PM 2026-06-13). A learned MATRIX FIELD over latent
    space: A(z) ∈ R^{k×k} is the local (Koopman-like) state operator, B(z) ∈
    R^{k×m} the control operator. This is the middle ground between a fixed
    Koopman K (z'=Kz+Ba — too rigid) and a generic world model (z'=f(z,a) — no
    spectral meaning): A(z) still has eigenvalues / eigenspaces / a spectral gap
    at every latent point, so spectral methods stay defined as fields over z.

    Strict generalization of AffineDynamics: A(z) is parameterized as I + Â(z),
    so at init (Â≈0) the map is a near-identity (stable) transition and a frozen
    Â=0 recovers the pure-control affine map z + B(z)a.

    R15 PRESERVED — affine in the action (∂²z'/∂a² = 0): B(z) does NOT depend on
    a, so the operator form reintroduces NO dynamics-curvature floor. This is why
    it is a legitimate drop-in for the affine dynamics rather than the run-9
    FullMLPDynamics ablation.

    Structural priors keep A a coherent OPERATOR BUNDLE rather than an
    unconstrained hypernetwork in disguise — exposed two ways:
      • shaped in-place by `structure` ∈ {none, symmetric} (hard) and `rank`>0
        (low-rank A = U(z)V(z)ᵀ + I);
      • as differentiable penalties via `structural_penalties(z)` — normal
        (‖AAᵀ−AᵀA‖²), smooth (‖A(zᵢ)−A(zⱼ)‖²/‖zᵢ−zⱼ‖²), spread (−Var σ, anti
        mode-collapse), radius (relu(σ_max−1)², soft stability bound). The Trainer
        weights these into the model loss; all default to 0 (pure A(z)z+B(z)a).
    `spectral_summary(z)` returns no-grad diagnostics (radius/eff_rank/normality)
    for logging the evolving spectrum.

    Singular values stand in for |eigenvalues|: they coincide as A→normal (the
    normal penalty's target) and have a numerically stable backward, unlike the
    complex eigvals of a general matrix. Pairing for the smoothness term uses a
    fixed roll(1) over the (already-shuffled) batch — no RNG draw, so resume
    stays bitwise-exact without touching checkpoint state.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, structure: str = "none", rank: int = 0):
        super().__init__()
        self.k, self.m = latent_dim, action_dim
        self.structure = str(structure)
        self.rank = int(rank)
        if self.rank > 0:
            self.U = mlp([latent_dim] + [hidden] * depth + [latent_dim * self.rank])
            self.V = mlp([latent_dim] + [hidden] * depth + [latent_dim * self.rank])
        else:
            self.A_net = mlp([latent_dim] + [hidden] * depth + [latent_dim * latent_dim])
        self.B = mlp([latent_dim] + [hidden] * depth + [latent_dim * action_dim])

    def _raw_A(self, z: Tensor) -> Tensor:
        if self.rank > 0:
            U = self.U(z).view(*z.shape[:-1], self.k, self.rank)
            V = self.V(z).view(*z.shape[:-1], self.k, self.rank)
            return U @ V.transpose(-1, -2)
        return self.A_net(z).view(*z.shape[:-1], self.k, self.k)

    def operators(self, z: Tensor) -> tuple:
        A = self._raw_A(z)
        if self.structure == "symmetric":
            A = 0.5 * (A + A.transpose(-1, -2))
        A = A + torch.eye(self.k, device=z.device, dtype=z.dtype)   # near-I init
        B = self.B(z).view(*z.shape[:-1], self.k, self.m)
        return A, B

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        A, B = self.operators(z)
        return (A @ z.unsqueeze(-1)).squeeze(-1) + (B @ a.unsqueeze(-1)).squeeze(-1)

    def structural_penalties(self, z: Tensor) -> dict:
        """Operator-bundle regularizers on a batch z (B,k). First-order only (no
        double-backward), so autocast is fine; the Trainer weights+sums them."""
        A, _ = self.operators(z)                         # (B,k,k), incl. the I shift
        At = A.transpose(-1, -2)
        comm = A @ At - At @ A                            # [A, Aᵀ]; 0 ⇔ A normal
        normal = comm.pow(2).flatten(-2).sum(-1).mean()
        A_roll, z_roll = A.roll(1, 0), z.roll(1, 0)       # deterministic pairing
        dA = (A - A_roll).pow(2).flatten(-2).sum(-1)
        dz = (z - z_roll).pow(2).sum(-1) + 1e-6
        smooth = (dA / dz).mean()
        sv = torch.linalg.svdvals(A.float())             # (B,k) ≥0, stable backward
        # spread = −(spectral entropy): minimizing it MAXIMIZES entropy ⇒ a
        # balanced spectrum / high effective rank ⇒ several distinct dynamical
        # modes. NOT −Var(σ): max-variance rewards ONE dominant mode (the very
        # collapse this prior is meant to prevent) — PM 2026-06-13.
        ps = sv / sv.sum(-1, keepdim=True).clamp_min(1e-9)
        spread = (ps * ps.clamp_min(1e-9).log()).sum(-1).mean()   # = −entropy
        radius = torch.relu(sv[..., 0] - 1.0).pow(2).mean()   # σ_max ≥ ρ(A): soft bound
        return {"normal": normal, "smooth": smooth, "spread": spread, "radius": radius}

    @torch.no_grad()
    def spectral_summary(self, z: Tensor) -> dict:
        A, _ = self.operators(z)
        sv = torch.linalg.svdvals(A.float())
        p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-9)
        eff_rank = torch.exp(-(p * p.clamp_min(1e-9).log()).sum(-1))   # entropy eff. rank
        At = A.transpose(-1, -2)
        comm = (A @ At - At @ A).pow(2).flatten(-2).sum(-1).sqrt()
        return {"op/radius": float(sv[..., 0].mean()),
                "op/eff_rank": float(eff_rank.mean()),
                "op/normality_resid": float(comm.mean())}


class FullMLPDynamics(nn.Module):
    """ABLATION-ONLY (bridge run 9): deterministic full-MLP transition
    z' = z + g([z; a]) — DELIBERATELY breaks R15's d^2 T/da^2 = 0. Exists to
    test whether the affine constraint actually binds (prediction: imagined-
    return variance blowup / worse return at matched dose). Never make this a
    default; if it matches affine, R15 needs requalification, not quiet
    adoption of this class."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.k, self.m = latent_dim, action_dim
        self.g = mlp([latent_dim + action_dim] + [hidden] * depth + [latent_dim])

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        return z + self.g(torch.cat([z, a], dim=-1))
