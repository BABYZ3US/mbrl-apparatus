"""Reward model R(z, a[, tau]) — the regularization target (R1).

With task conditioning (task_dim > 0) the Hessian penalty can extend over the
task coordinate too: curvature control along tau is the mechanism by which the
model is forced to *interpolate smoothly between tasks* — the multi-task
generalization hypothesis (smooth-in-task => zero-shot transfer).
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

from .encoder import mlp


class RewardModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, task_dim: int = 0):
        super().__init__()
        self.k, self.m, self.task_dim = latent_dim, action_dim, task_dim
        self.net = mlp([latent_dim + action_dim + task_dim] + [hidden] * depth + [1])

    def forward(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        parts = [z, a] if not self.task_dim else [z, a, tau]
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)

    def on_concat(self, x: Tensor) -> Tensor:
        """Scalar-per-sample on concatenated (z, a[, tau]) — the callable handed
        to hvp_penalty; Hessian taken in joint latent(-task) coords."""
        return self.net(x).squeeze(-1)
