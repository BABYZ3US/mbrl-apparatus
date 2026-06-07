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
