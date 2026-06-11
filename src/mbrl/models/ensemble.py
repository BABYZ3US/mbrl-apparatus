"""ensemble — an R15-SAFE probabilistic-ensemble dynamics (PETS-style, affine members).

The battle-tested MBRL uncertainty device (Chua et al. 2018 PETS; Janner et al.
2019 MBPO) is an ensemble of independently-initialized dynamics models whose
DISAGREEMENT estimates epistemic uncertainty. The standard PETS member is a full
MLP over (s, a) — which would break this apparatus's R15 invariant (dynamics
affine in the action: d2T/da2 = 0, founding-doc hard rule). So each member here
is an AFFINE-IN-ACTION map, the same model class as models/dynamics.py:

    z' = f_m(z[, tau]) + B_m(z[, tau]) @ a        (per member m)

Ensembling preserves R15 exactly (a sum of affine maps is affine), while still
buying the battle-tested goods: mean prediction, per-member rollouts (trajectory
sampling), and disagreement (std across members) for exploration bonuses or
model-error gating. Config-gated; the default trainer path is untouched.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoder import mlp


class AffineMember(nn.Module):
    """One ensemble member: z' = f(x) + B(x) @ a, x = [z, tau] — affine in a."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, task_dim: int = 0):
        super().__init__()
        in_dim = latent_dim + task_dim
        self.f = mlp([in_dim] + [hidden] * depth + [latent_dim])
        self.B = mlp([in_dim] + [hidden] * depth + [latent_dim * action_dim])
        self.latent_dim, self.action_dim, self.task_dim = latent_dim, action_dim, task_dim

    def forward(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        x = torch.cat([z, tau], dim=-1) if self.task_dim else z
        drift = self.f(x)
        Bmat = self.B(x).reshape(*x.shape[:-1], self.latent_dim, self.action_dim)
        return drift + (Bmat @ a.unsqueeze(-1)).squeeze(-1)


class EnsembleAffineDynamics(nn.Module):
    """N independently-initialized affine members + mean/disagreement readouts."""

    def __init__(self, latent_dim: int, action_dim: int, n_members: int = 5,
                 hidden: int = 256, depth: int = 2, task_dim: int = 0):
        super().__init__()
        if n_members < 2:
            raise ValueError("an ensemble needs n_members >= 2 (got %d)" % n_members)
        self.members = nn.ModuleList([
            AffineMember(latent_dim, action_dim, hidden, depth, task_dim)
            for _ in range(n_members)
        ])
        # dim attributes mirroring the single-dynamics classes (k, m) so the
        # ensemble stays a drop-in for duck-typed consumers
        self.k, self.m = latent_dim, action_dim

    def all_members(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        """(M, ..., latent_dim) — every member's prediction."""
        return torch.stack([m(z, a, tau) for m in self.members])

    def forward(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        """The ensemble MEAN prediction (a sum of affine maps — R15 holds)."""
        return self.all_members(z, a, tau).mean(dim=0)

    def disagreement(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        """Epistemic-uncertainty proxy: mean-over-dims std across members, (...)."""
        return self.all_members(z, a, tau).std(dim=0).mean(dim=-1)

    def member_rollout_step(self, m: int, z: Tensor, a: Tensor,
                            tau: Tensor | None = None) -> Tensor:
        """One step through member m — PETS trajectory sampling (TS-inf)."""
        return self.members[m](z, a, tau)
