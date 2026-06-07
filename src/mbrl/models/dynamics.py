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
