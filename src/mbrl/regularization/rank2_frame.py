"""Rank-2 reward⊥energy frame (PM 2026-06-14).

The hypothesis: for this problem — and maybe most RL problems — the controllable
essence is RANK-2. The representation should organize into two orthogonal axes with
opposed senses:

  • p / reward axis   û_R = ∇_z R     (the maximal-reward / control direction, ASCEND)
  • d / energy axis   û_E = −∇_z E    (the minimal-energy / natural-flow direction, DESCEND)

with ⟨û_R, û_E⟩ = 0. In the two-band-superconductor reading this is a 2-component
order parameter: gram_eff_rank≈2 = the healthy coherent state, →1 = collapse to one
band, ≫2 = nuisance. For locomotion, rank-2 is the minimal rank that holds a limit
cycle (a running gait IS a 2D periodic orbit).

Two definitions of "energy" (cf5 arms):
  • lyapunov   — a learned scalar E(d), grounded by the autonomous dynamics drift
                 descending it (relu(E(d_auto) − E(d))). Energy = what the natural
                 dynamics relaxes; û_E = −∇_z E(D(z)).
  • contractive — the most-contracted direction of the dynamics operator op_d (its
                 smallest right-singular vector), pulled back to z. No head; energy
                 is implicit in the operator spectrum.

The terms below are first/second-order but config-gated (default off ⇒ no-op). The
axis-orthogonality is a double-backward (∇_z of reward/energy, then ∇_θ of the cos²)
in the same class as the Hutchinson curvature penalty — kept cheap by subsampling.
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

from ..models.encoder import mlp


class EnergyHead(nn.Module):
    """Learned Lyapunov-style scalar energy E(d) on the dynamics latent d. The
    minimal-energy direction is −∇E; it is grounded (see lyapunov_grounding) by
    requiring the autonomous dynamics drift to descend it, so it tracks the system's
    natural relaxation rather than an arbitrary scalar."""

    def __init__(self, d_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.net = mlp([d_dim] + [hidden] * depth + [1])

    def forward(self, d: Tensor) -> Tensor:
        return self.net(d).squeeze(-1)


def axis_cos2(g_r: Tensor, g_e: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean squared cosine between the per-sample reward and energy axes. 0 ⇔ the two
    gradient fields are everywhere orthogonal (the rank-2 frame is square)."""
    num = (g_r * g_e).sum(-1)
    den = g_r.norm(dim=-1) * g_e.norm(dim=-1) + eps
    cos = num / den
    return (cos * cos).mean()


def rank2_tail_penalty(z: Tensor, target_rank: int = 2) -> Tensor:
    """Variance of the representation OUTSIDE its top-`target_rank` eigendirections —
    the covariance eigenvalue tail. Minimizing it presses z into a rank-`target_rank`
    subspace (the live `latent/gram_eff_rank` readout turned into an objective)."""
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.transpose(-1, -2) @ zc) / max(zc.shape[0], 1)
    ev = torch.linalg.eigvalsh(cov.float())              # ascending eigenvalues
    if ev.shape[-1] <= target_rank:
        return z.new_zeros(())
    return ev[:-target_rank].clamp_min(0.0).sum()        # all but the top-`target_rank`


def lyapunov_grounding(energy: EnergyHead, op_d, d: Tensor, action_dim: int) -> Tensor:
    """Ground E as an energy of the dynamics: the AUTONOMOUS (zero-action) drift must
    not increase it. relu(E(d_auto) − E(d)) → 0, with d_auto detached so the grounding
    trains the energy head, not the operator."""
    zero_a = d.new_zeros(*d.shape[:-1], action_dim)
    d_auto = op_d(d, zero_a).detach()
    return torch.relu(energy(d_auto) - energy(d)).mean()


def contractive_axis_in_d(op_d, d_sub: Tensor) -> Tensor:
    """Smallest right-singular vector of op_d's state operator A_d at each sample — the
    most-contracted (minimal-energy) direction of the dynamics, in d-space. Detached:
    it is read off the current operator, not differentiated through (the orthogonality
    term moves the representation toward it, not it toward the representation)."""
    A, _ = op_d.operators(d_sub)
    # right singular vectors are the rows of Vh; S descending ⇒ last row = smallest
    _, _, Vh = torch.linalg.svd(A.float())
    return Vh[..., -1, :].detach().to(d_sub.dtype)
