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


def gae_advantages(rewards: Tensor, values: Tensor, gamma: float,
                   lam: float) -> tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation (Schulman et al. 2016) — the battle-tested
    advantage estimator (PPO/A2C standard), here over imagined rollouts.

    A_t = sum_l (gamma*lam)^l * delta_{t+l},  delta_t = r_t + gamma v_{t+1} - v_t.

    rewards: (H, B); values: (H+1, B) with the bootstrap last.
    Returns (advantages, returns), both (H, B); returns = advantages + values[:-1]
    (the value-regression target). lam=1 recovers discounted Monte-Carlo minus
    baseline; lam=0 the one-step TD residual.
    """
    H = rewards.shape[0]
    adv = torch.empty_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(H)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values[:-1]
