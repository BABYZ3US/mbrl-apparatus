"""Policy and value heads trained on imagined latent rollouts (Dreamer-style).

The policy is NEVER curvature-penalized (R10): it needs curvature freedom.
Optional task conditioning (task_dim > 0): inputs become [z, tau].
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

from .encoder import mlp

_LOG2 = math.log(2.0)


class Policy(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, action_scale: float = 1.0, task_dim: int = 0,
                 init_scale: float = 1.0, log_std_min: float = -5.0):
        super().__init__()
        self.net = mlp([latent_dim + task_dim] + [hidden] * depth + [2 * action_dim])
        self.action_dim, self.action_scale, self.task_dim = action_dim, action_scale, task_dim
        # HARD variance bound (PM 2026-06-15): the lower clamp on log_std is a settable
        # attribute so the Trainer can drive it reward-adaptively (σ ≥ e^{log_std_min}). A
        # minimum σ keeps collection exploratory — the policy CANNOT collapse to a
        # deterministic point mass — the structural fix the soft entropy floors (cf19/cf20)
        # couldn't enforce. Default -5.0 = the legacy clamp (byte-identical when undriven).
        self.log_std_min = float(log_std_min)
        # near-zero init (seed robustness, PM 2026-06-15): shrink the final layer so the
        # initial policy is ~the same near-zero-action / mid-entropy map for EVERY seed,
        # cutting the across-seed spread in where training starts. init_scale<1 ⇒ mu≈0,
        # log_std≈0 (σ≈1). The reward-coupled entropy floor then keeps exploration alive.
        if init_scale != 1.0:
            last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.mul_(init_scale)
                last.bias.zero_()

    def forward(self, z: Tensor, tau: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = torch.cat([z, tau], dim=-1) if self.task_dim else z
        mu, log_std = self.net(x).chunk(2, dim=-1)
        return mu, log_std.clamp(self.log_std_min, 2.0)

    def sample(self, z: Tensor, tau: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Reparameterized tanh-Gaussian sample with exact log-prob.
        log|d tanh(u)/du| summed via the numerically stable
        log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u))."""
        mu, log_std = self(z, tau)
        eps = torch.randn_like(mu)
        pre = mu + eps * log_std.exp()
        a = torch.tanh(pre) * self.action_scale
        logp_gauss = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_det = (2.0 * (_LOG2 - pre - torch.nn.functional.softplus(-2.0 * pre))).sum(-1)
        logp = logp_gauss - log_det - mu.shape[-1] * math.log(self.action_scale + 1e-12)
        return a, logp

    def mean_action(self, z: Tensor, tau: Tensor | None = None) -> Tensor:
        """Deterministic action = the tanh-Gaussian MEAN (the standard eval/benchmark
        convention). No action noise; ignores log_std. Used for the det-eval metric."""
        mu, _ = self(z, tau)
        return torch.tanh(mu) * self.action_scale


class ValueFn(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 256, depth: int = 2,
                 task_dim: int = 0):
        super().__init__()
        self.net = mlp([latent_dim + task_dim] + [hidden] * depth + [1])
        self.task_dim = task_dim

    def forward(self, z: Tensor, tau: Tensor | None = None) -> Tensor:
        x = torch.cat([z, tau], dim=-1) if self.task_dim else z
        return self.net(x).squeeze(-1)
