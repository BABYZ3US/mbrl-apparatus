"""SequencePlanner — the Dreamer latent action-sequence planner (W: transformer
planning module, 2026-06-11).

Conditioned on the starting latent z0, a small CAUSAL transformer emits an
H-step action plan with per-step tanh-Gaussian log-probs. It is a drop-in actor
for behaviour_update: the affine dynamics T still predicts the latents
(z_{k+1}=T(z_k,a_k), R15 preserved), the planner only predicts the actions, and
the existing differentiable-imagination + λ-return objective trains it.
Execution is receding-horizon: take the first action, re-encode, replan.

House rules: the planner is the policy/actor, so it is NEVER curvature-penalized
(R10). The horizon is FIXED (the plan length); adaptive-horizon is incompatible
and is disabled when the planner is on. No dropout (bitwise-resume determinism).
The tanh log-prob is the numerically stable form shared with policy.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

_LOG2 = math.log(2.0)


class SequencePlanner(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, horizon: int,
                 d_model: int = 128, nhead: int = 4, layers: int = 2,
                 action_scale: float = 1.0, task_dim: int = 0):
        super().__init__()
        self.H, self.action_dim = int(horizon), action_dim
        self.action_scale, self.task_dim = action_scale, task_dim
        self.in_proj = nn.Linear(latent_dim + task_dim, d_model)
        # H learned action-slot queries (+ the z0 context token at position 0)
        self.query = nn.Parameter(torch.randn(self.H, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.mu_head = nn.Linear(d_model, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # causal mask over [ctx, q_1..q_H]: position i attends to j<=i (True=masked)
        mask = torch.triu(torch.ones(self.H + 1, self.H + 1), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _moments(self, z0: Tensor, tau: Tensor | None = None):
        """(mu (B,H,act), log_std (act,)) — the plan's action distribution."""
        x = torch.cat([z0, tau], dim=-1) if self.task_dim else z0
        ctx = self.in_proj(x).unsqueeze(1)                       # (B,1,d)
        q = self.query.unsqueeze(0).expand(z0.shape[0], -1, -1)  # (B,H,d)
        seq = torch.cat([ctx, q], dim=1)                         # (B,H+1,d)
        out = self.tf(seq, mask=self.causal_mask)                # (B,H+1,d)
        return self.mu_head(out[:, 1:]), self.log_std.clamp(-5.0, 2.0)

    def plan(self, z0: Tensor, tau: Tensor | None = None):
        """Reparameterized H-step plan. Returns (actions (H,B,act), logp (H,B))
        — time-major to drop straight into behaviour_update's rollout loop."""
        mu, log_std = self._moments(z0, tau)
        eps = torch.randn_like(mu)
        pre = mu + eps * log_std.exp()
        a = torch.tanh(pre) * self.action_scale
        logp_g = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_det = (2.0 * (_LOG2 - pre - F.softplus(-2.0 * pre))).sum(-1)
        logp = logp_g - log_det - self.action_dim * math.log(self.action_scale + 1e-12)
        return a.transpose(0, 1).contiguous(), logp.transpose(0, 1).contiguous()

    @torch.no_grad()
    def act(self, z0: Tensor, tau: Tensor | None = None) -> Tensor:
        """Receding-horizon execution: the deterministic first planned action."""
        mu, _ = self._moments(z0, tau)
        return torch.tanh(mu[:, 0]) * self.action_scale
