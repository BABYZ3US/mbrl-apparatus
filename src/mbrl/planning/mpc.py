"""mpc — CEM-MPC planner over the LEARNED latent model (PETS/Chua-style MPC).

Receding-horizon control: at each collection step, optimize an H-step action
SEQUENCE by rolling the learned latent dynamics forward and scoring the
discounted model return, then EXECUTE ONLY THE FIRST action (Todorov2012 MPC;
Janner2019 MBPO/short-rollout discipline; Chua2018 PETS = CEM + learned model).

This module is a thin wrapper: it owns the rollout/score construction and
defers all sampling/refit to the battle-tested `cem_plan` optimizer. It imports
no model — `step_fn` / `reward_fn` / `value_fn` are passed in, so any dynamics
(affine, operator, full-MLP) and any reward head plug in unchanged. Pure torch +
a passed-in generator → deterministic given a seed, no global RNG (the
resume-bitwise discipline).

Callable contracts (N = batch the callables receive; broadcasts over pop*horizon):
  step_fn(z[N, k], a[N, A]) -> z'[N, k]            learned latent dynamics
  reward_fn(z[N, k], a[N, A], tau) -> r[N]         learned reward (tau broadcast)
  value_fn(z[N, k], tau) -> v[N] | None            terminal value bootstrap (opt.)
These match `models.dynamics.*.forward(z, a)`, `models.reward.RewardModel.forward
(z, a, tau)`, and a critic value head reduced to a flat (N,) tensor.

Batched z0 approach: we LOOP over the B rows, running one independent `cem_plan`
per row, and stack the first actions. B = num_envs is small at collection, so
the loop is cheap; looping keeps each CEM's Gaussian fully independent (no
cross-row leakage) and reuses the single-z0 path verbatim — the simplest correct
option (the task explicitly sanctions it). A single fused [B*pop, H, A] call
would optimize a SHARED mean across rows under `cem_plan`'s current API, which is
wrong; per-row independence would require changes to the optimizer.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .cem import cem_plan

StepFn = Callable[[Tensor, Tensor], Tensor]
RewardFn = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]
ValueFn = Callable[[Tensor, Optional[Tensor]], Tensor]


class CEMPlanner:
    """CEM-MPC planner: collection-time actions from the learned latent model.

    One instance is reused across collection steps; `act` is stateless apart
    from the supplied `generator` (no warm-start carryover — each call runs a
    fresh receding-horizon optimization, the standard PETS setup).
    """

    def __init__(self, action_dim: int, horizon: int = 12, pop: int = 256,
                 iters: int = 5, elites: int = 32, gamma: float = 0.99,
                 device: torch.device | str = "cpu"):
        if elites > pop:
            raise ValueError("elites (%d) must be <= pop (%d)" % (elites, pop))
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.pop = int(pop)
        self.iters = int(iters)
        self.elites = int(elites)
        self.gamma = float(gamma)
        self.device = device

    @torch.no_grad()
    def act(self, z0: Tensor, step_fn: StepFn, reward_fn: RewardFn,
            value_fn: ValueFn | None = None, tau: Tensor | None = None,
            generator: torch.Generator | None = None) -> Tensor:
        """First action of the elite-mean plan from latent z0.

        z0: [k] single latent, or [B, k] batch (one independent CEM per row).
        Returns [A] for a single z0, or [B, A] for a batch. Actions live in
        [-1, 1] (the bounds `cem_plan` clamps to).
        """
        if z0.dim() == 1:
            return self._act_single(z0, step_fn, reward_fn, value_fn, tau, generator)
        if z0.dim() == 2:
            # Loop B independent CEM solves (B = num_envs is small at collection).
            acts = [self._act_single(z0[b], step_fn, reward_fn, value_fn, tau, generator)
                    for b in range(z0.shape[0])]
            return torch.stack(acts, dim=0)                       # [B, A]
        raise ValueError("z0 must be [k] or [B, k]; got shape %s" % (tuple(z0.shape),))

    def _act_single(self, z0: Tensor, step_fn: StepFn, reward_fn: RewardFn,
                    value_fn: ValueFn | None, tau: Tensor | None,
                    generator: torch.Generator | None) -> Tensor:
        """Single-latent path: build the discounted-model-return score_fn, run
        `cem_plan`, return the FIRST action of the elite-mean sequence."""
        z0 = z0.to(self.device)
        k = z0.shape[-1]

        def score_fn(cand: Tensor) -> Tensor:                    # cand: [pop, H, A]
            pop = cand.shape[0]
            z = z0.unsqueeze(0).expand(pop, k).contiguous()      # [pop, k]
            ret = torch.zeros(pop, device=cand.device, dtype=cand.dtype)
            disc = 1.0
            for t in range(self.horizon):
                a_t = cand[:, t, :]                              # [pop, A]
                ret = ret + disc * reward_fn(z, a_t, tau)        # Σ γ^t r_t
                z = step_fn(z, a_t)                              # z_{t+1}
                disc *= self.gamma
            if value_fn is not None:                             # + γ^H V(z_H)
                ret = ret + disc * value_fn(z, tau)
            return ret                                           # [pop], higher better

        plan = cem_plan(
            score_fn, horizon=self.horizon, action_dim=self.action_dim,
            iters=self.iters, pop=self.pop, elites=self.elites,
            action_low=-1.0, action_high=1.0,
            generator=generator, device=self.device,
        )                                                        # [H, A]
        return plan[0]                                           # first action [A]
