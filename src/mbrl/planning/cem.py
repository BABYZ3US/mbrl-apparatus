"""cem — the Cross-Entropy Method planner (PETS/MPC standard).

The battle-tested sampling planner (Chua et al. 2018; standard MPC baseline):
iteratively (1) sample N action sequences from a Gaussian, (2) score them with a
caller-supplied rollout function, (3) refit the Gaussian to the top-K elites.
Pure torch + a passed-in generator — deterministic given a seed, no global RNG
(the resume-bitwise discipline). The rollout callable owns ALL model access, so
this module imports no dynamics/reward and stays a standalone unit.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


@torch.no_grad()
def cem_plan(score_fn: Callable[[Tensor], Tensor], horizon: int, action_dim: int,
             iters: int = 5, pop: int = 256, elites: int = 32,
             action_low: float = -1.0, action_high: float = 1.0,
             init_std: float = 0.5, min_std: float = 0.02,
             generator: torch.Generator | None = None,
             device: torch.device | str = "cpu") -> Tensor:
    """Optimize an action SEQUENCE (horizon, action_dim) against score_fn.

    score_fn: (pop, horizon, action_dim) -> (pop,) total score (higher better).
    Returns the elite-mean sequence after `iters` refits, clamped to the bounds.
    """
    if elites > pop:
        raise ValueError("elites (%d) must be <= pop (%d)" % (elites, pop))
    mid = 0.5 * (action_low + action_high)
    mean = torch.full((horizon, action_dim), mid, device=device)
    std = torch.full((horizon, action_dim), init_std, device=device)
    for _ in range(iters):
        noise = torch.randn((pop, horizon, action_dim), generator=generator, device=device)
        cand = (mean + std * noise).clamp(action_low, action_high)
        scores = score_fn(cand)                          # (pop,)
        top = torch.topk(scores, elites).indices
        elite = cand[top]                                # (elites, H, A)
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp_min(min_std)
    return mean.clamp(action_low, action_high)
