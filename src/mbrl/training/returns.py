"""Dreamer-style lambda-returns over imagined rollouts.

R^λ_t = r_t + γ [ (1−λ) v_{t+1} + λ R^λ_{t+1} ],   R^λ_H = v_H.

Cheap: one backward recursion over the horizon, no critic ensembles, no
per-step action-value maximization — the policy gradient flows directly
through the learned dynamics and reward (which the curvature penalty keeps
smooth — that is the synergy with R15).
"""
from __future__ import annotations

import torch
from torch import Tensor


def lambda_returns(rewards: Tensor, values: Tensor, gamma: float, lam: float) -> Tensor:
    """rewards: (H, B) imagined rewards r_0..r_{H-1};
    values: (H+1, B) v(z_0)..v(z_H) (last one is the bootstrap).
    Returns (H, B) lambda-returns aligned with rewards."""
    H = rewards.shape[0]
    out = torch.empty_like(rewards)
    last = values[-1]
    for t in reversed(range(H)):
        last = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * last)
        out[t] = last
    return out
