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


def symlog(x: Tensor) -> Tensor:
    """symlog(x) = sign(x) * log1p(|x|) — squashes reward targets so the model
    fits in a bounded prediction space (Dreamer-V3 trick)."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(x.abs())


class RewardModel(nn.Module):
    """Shared trunk (the existing hidden layers) + n_heads separate final
    linear heads — a hobby ensemble: all heads regress the same targets, and
    disagreement comes purely from the independent N(0, 0.02) head inits."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, task_dim: int = 0, n_heads: int = 1):
        super().__init__()
        self.k, self.m, self.task_dim = latent_dim, action_dim, task_dim
        self.n_heads = n_heads
        self.net = mlp([latent_dim + action_dim + task_dim] + [hidden] * depth,
                       out_act=nn.SiLU)  # trunk
        heads = []
        for _ in range(n_heads):
            lin = nn.Linear(hidden, 1)
            nn.init.normal_(lin.weight, std=0.02)  # different draw per head
            nn.init.zeros_(lin.bias)
            heads.append(lin)
        self.heads = nn.ModuleList(heads)

    def _from_features(self, feats: Tensor) -> Tensor:
        """Trunk features -> per-head outputs, shape (n_heads, ...)."""
        return torch.stack([h(feats).squeeze(-1) for h in self.heads])

    def all_heads(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        """(n_heads, B) per-head predictions (symlog space if symlog training)."""
        parts = [z, a] if not self.task_dim else [z, a, tau]
        return self._from_features(self.net(torch.cat(parts, dim=-1)))

    def forward(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        """Head MEAN — the fit-loss output (all heads trained on the same targets)."""
        return self.all_heads(z, a, tau).mean(0)

    def on_concat(self, x: Tensor) -> Tensor:
        """Scalar-per-sample (head mean) on concatenated (z, a[, tau]) — the
        callable handed to hvp_penalty; Hessian taken in joint latent(-task)
        coords, on the model's raw (symlog-space) output: smoothness is
        enforced in prediction space, not after symexp."""
        return self._from_features(self.net(x)).mean(0)
