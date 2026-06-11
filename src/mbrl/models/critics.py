"""critics — battle-tested actor-critic components (TwinQ, tanh-squashed Gaussian).

Two standard, heavily-validated building blocks from the SAC/TD3 family, packaged
as importable units (config-gated; nothing here changes the default trainer path):

* ``TwinQ`` — clipped double-Q (Fujimoto et al. 2018 TD3; Haarnoja et al. 2018
  SAC): two independent Q-MLPs over (z, a[, tau]); ``min_q`` takes the elementwise
  minimum, the standard overestimation-bias fix.
* ``SquashedGaussianPolicy`` — the SAC actor: a reparameterized Gaussian squashed
  through tanh, with the exact change-of-variables log-prob correction
  log pi(a) = log N(u) - sum log(1 - tanh(u)^2), bounded actions in
  (-action_scale, action_scale).

Conventions mirror models/policy.py: optional task conditioning via task_dim
(inputs become [z, tau]), the shared ``mlp`` builder, fp32 math.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoder import mlp

# tanh log-prob correction guard: keep 1 - tanh(u)^2 away from exact zero
_EPS = 1e-6


class TwinQ(nn.Module):
    """Clipped double-Q: two independent Q(z, a[, tau]) heads + min_q."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, task_dim: int = 0):
        super().__init__()
        in_dim = latent_dim + action_dim + task_dim
        self.q1 = mlp([in_dim] + [hidden] * depth + [1])
        self.q2 = mlp([in_dim] + [hidden] * depth + [1])
        self.task_dim = task_dim

    def _cat(self, z: Tensor, a: Tensor, tau: Tensor | None) -> Tensor:
        parts = [z, a] + ([tau] if self.task_dim else [])
        return torch.cat(parts, dim=-1)

    def forward(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self._cat(z, a, tau)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, z: Tensor, a: Tensor, tau: Tensor | None = None) -> Tensor:
        q1, q2 = self(z, a, tau)
        return torch.minimum(q1, q2)


class SquashedGaussianPolicy(nn.Module):
    """SAC actor: tanh(N(mu, sigma)) with the exact squash log-prob correction."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, action_scale: float = 1.0, task_dim: int = 0):
        super().__init__()
        self.net = mlp([latent_dim + task_dim] + [hidden] * depth + [2 * action_dim])
        self.action_dim, self.action_scale, self.task_dim = action_dim, action_scale, task_dim

    def forward(self, z: Tensor, tau: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = torch.cat([z, tau], dim=-1) if self.task_dim else z
        mu, log_std = self.net(x).chunk(2, dim=-1)
        return mu, log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def sample(self, z: Tensor, tau: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Reparameterized action in (-scale, scale) + its exact log-prob."""
        mu, log_std = self(z, tau)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)            # rsample: gradients flow
        a = torch.tanh(u)
        # ALGORITHM REVIEW 2026-06-11: mirror the WIRED policy (models/policy.py)
        # exactly — the numerically stable tanh log-det
        #   log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u))
        # (the eps-clamped log(1-a^2+eps) loses precision for |u| >~ 6), AND the
        # action_scale Jacobian (a_out = tanh(u)*s adds -sum_i log s) that the
        # old form omitted while claiming an exact log-prob.
        base = -0.5 * (((u - mu) / std) ** 2 + 2 * log_std + torch.log(
            torch.tensor(2.0 * torch.pi, device=u.device, dtype=u.dtype)))
        log_det = 2.0 * (torch.log(torch.tensor(2.0, device=u.device, dtype=u.dtype))
                         - u - torch.nn.functional.softplus(-2.0 * u))
        logp = (base - log_det).sum(dim=-1) \
            - mu.shape[-1] * torch.log(torch.tensor(self.action_scale + 1e-12,
                                                    device=u.device, dtype=u.dtype))
        return a * self.action_scale, logp

    def deterministic(self, z: Tensor, tau: Tensor | None = None) -> Tensor:
        """The mode (tanh of the mean) — evaluation-time action."""
        mu, _ = self(z, tau)
        return torch.tanh(mu) * self.action_scale
