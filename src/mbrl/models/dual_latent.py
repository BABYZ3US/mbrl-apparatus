"""Dual-latent controlled-operator model (PM 2026-06-13).

A shared encoder backbone z = E(s) branches into a DYNAMICS latent d = D(z)
(coordinates organized for predictability) and a POLICY latent p = P(z)
(coordinates organized for control). This resolves the session-long tension of
forcing ONE representation to satisfy dynamics-linearization, Bellman, spectral
and policy-sufficiency at once — objectives that are not generally aligned.

Two modes give a clean A/B on "is a second operator worth it?":

  mode='shared' (option 1, ONE operator):
    a single operator rolls the backbone z directly (z' = A(z)z + B(z)a). d = D(z)
    and p = P(z) are instantaneous readouts: the dynamics fit is scored in d-space
    (only D(z) must be predictable — nuisance dims of z may stay free), control
    (reward/value/policy) reads p. One operator governs both branches.

  mode='twin' (option 3, TWO operators):
    d and p each get their OWN operator (op_d, op_p). The dynamics fit grounds
    op_d in d-space; the imagined CONTROL rollout is rolled by op_p in p-space;
    a weak coupling L_couple = ‖W_d d − W_p p‖² ties the two geometries so the
    shared encoder can't let them drift into unrelated features. The second
    operator lets the policy latent carry its own linearizable dynamics rather
    than inheriting the dynamics branch's through the P readout.

Both operators are OperatorDynamics (R15-safe: affine in a). The structural
priors (normal/smooth/spread/radius) apply to each operator independently.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

from .encoder import mlp
from .dynamics import OperatorDynamics


class DualLatent(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 depth: int = 2, mode: str = "shared", d_dim: int = 0,
                 p_dim: int = 0, op_structure: str = "none", op_rank: int = 0,
                 couple_dim: int = 0):
        super().__init__()
        self.mode = str(mode)
        self.k, self.m = latent_dim, action_dim
        self.d_dim = int(d_dim) or latent_dim
        self.p_dim = int(p_dim) or latent_dim
        # task-specific projections off the shared backbone z
        self.D = mlp([latent_dim] + [hidden] * depth + [self.d_dim])   # dynamics latent
        self.P = mlp([latent_dim] + [hidden] * depth + [self.p_dim])   # policy latent
        if self.mode == "shared":
            # one operator, on the backbone z (dim k); d,p are readouts of z
            self.op = OperatorDynamics(latent_dim, action_dim, hidden, depth,
                                       structure=op_structure, rank=op_rank)
        elif self.mode == "twin":
            self.op_d = OperatorDynamics(self.d_dim, action_dim, hidden, depth,
                                         structure=op_structure, rank=op_rank)
            self.op_p = OperatorDynamics(self.p_dim, action_dim, hidden, depth,
                                         structure=op_structure, rank=op_rank)
            cd = int(couple_dim) or min(self.d_dim, self.p_dim)
            self.Wd = nn.Linear(self.d_dim, cd, bias=False)
            self.Wp = nn.Linear(self.p_dim, cd, bias=False)
        else:
            raise ValueError(f"dual_latent.mode must be shared|twin, got {mode!r}")

    def d_of(self, z: Tensor) -> Tensor:
        return self.D(z)

    def p_of(self, z: Tensor) -> Tensor:
        return self.P(z)

    # --- operators (mode-dependent) --------------------------------------------
    def operators(self):
        """The OperatorDynamics module(s) whose structural priors get penalized."""
        return [self.op] if self.mode == "shared" else [self.op_d, self.op_p]

    def couple(self, d: Tensor, p: Tensor) -> Tensor:
        """L_couple = ‖W_d d − W_p p‖² (twin only); 0 for shared (d,p share z)."""
        if self.mode != "twin":
            return d.new_zeros(())
        return (self.Wd(d) - self.Wp(p)).pow(2).sum(-1).mean()
