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
    natural relaxation rather than an arbitrary scalar.

    `anchor` (cf7-fix, PM 2026-06-14) makes E non-collapsible:
    E(d) = anchor·½‖d‖² + tanh(head(d)) — a KINETIC-energy floor (latent 'temperature')
    plus a BOUNDED learned correction. Without it the head collapses to a CONSTANT — the
    trivial minimizer of the one-sided penalties relu(E(d')−E(d)−r) and relu(E(d_auto)−
    E(d)), which both go to exactly 0 and make the dissipativity VACUOUS (observed in
    cf6/cf7). The ½‖d‖² floor grows with the latent and the tanh correction (∈(−1,1))
    cannot cancel it, so E can never be constant; energy growth must be paid for by
    reward, which is the intended constraint."""

    def __init__(self, d_dim: int, hidden: int = 256, depth: int = 2, anchor: float = 0.0):
        super().__init__()
        self.net = mlp([d_dim] + [hidden] * depth + [1])
        self.anchor = float(anchor)

    def forward(self, d: Tensor) -> Tensor:
        if self.anchor > 0.0:
            # kinetic floor (grows with the latent) + a tanh-BOUNDED correction that
            # cannot cancel it — forecloses the collapse-to-constant failure mode
            return self.anchor * 0.5 * (d * d).sum(-1) + torch.tanh(self.net(d).squeeze(-1))
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


def spectral_shell_penalty(z: Tensor, target_rank: int = 2, target: float = 1.0) -> Tensor:
    """Two-sided rank-k energy shell (PM 2026-06-14) — the Ginzburg–Landau double-well
    for the representation's Gram spectrum:

        shell = Σ_{i≤k} (target − λ_i)²  +  Σ_{i>k} λ_i²

    Keep the top-`target_rank` covariance eigenvalues pinned at `target` (the active
    modes — penalized for collapsing to 0 AND for blowing up) and push the rest to 0
    (rank-k). Unlike the one-sided rank2_tail (which permitted rank-1 collapse — only
    pushed the tail down, never held the top up), this is two-sided in EVERY mode, so the
    spectrum can fall to neither rank-1 nor explode. Operates on the Gram directly — no
    learned energy head to go vacuous. For target_rank=1 it is the (target−‖ψ‖²)²
    double-well; here it is the rank-2 shell. eigvalsh has a stable backward."""
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.transpose(-1, -2) @ zc) / max(zc.shape[0], 1)
    ev = torch.linalg.eigvalsh(cov.float()).clamp_min(0.0)   # ascending eigenvalues
    k = min(int(target_rank), ev.shape[-1])
    shell = ((target - ev[-k:]) ** 2).sum()                  # top-k held at `target`
    if ev.shape[-1] > k:
        shell = shell + (ev[:-k] ** 2).sum()                 # rest pushed to 0
    return shell


def log_det_barrier(z: Tensor, eps: float = 1e-2) -> Tensor:
    """Log-determinant volume barrier on the representation's Gram (PM 2026-06-14 — the
    KL / Gaussian-prior spectrum term):  −mean ln(λ_i + eps).

    Minimizing it MAXIMIZES the log-volume Σ ln(λ_i+eps), pushing every eigenvalue AWAY
    from zero — the canonical anti-singularity / anti-collapse regularizer (the spectrum
    part of KL(N(0,Σ)‖N(0,I)); MCR2's 'total coding rate'). It bounds cond(G)=λ_max/λ_min:
    paired with the rank-k shell (which pushes the TAIL down), the tail settles at a small
    floor ~sqrt(w_logdet/(2·w_shell)) instead of going to 0, so cond stops blowing up,
    tunably via w_logdet. `eps` is the ridge that keeps the barrier soft (push ~1/eps at 0)."""
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.transpose(-1, -2) @ zc) / max(zc.shape[0], 1)
    ev = torch.linalg.eigvalsh(cov.float()).clamp_min(0.0)
    return -(ev + eps).log().mean()


def lyapunov_grounding(energy: EnergyHead, op_d, d: Tensor, action_dim: int) -> Tensor:
    """Ground E as an energy of the dynamics: the AUTONOMOUS (zero-action) drift must
    not increase it. relu(E(d_auto) − E(d)) → 0, with d_auto detached so the grounding
    trains the energy head, not the operator."""
    zero_a = d.new_zeros(*d.shape[:-1], action_dim)
    d_auto = op_d(d, zero_a).detach()
    return torch.relu(energy(d_auto) - energy(d)).mean()


def dissipativity_penalty(energy: EnergyHead, d: Tensor, d_next: Tensor,
                          supply: Tensor) -> Tensor:
    """Soft thermodynamic constraint (PM 2026-06-14) — the SOFTER alternative to hard
    ∇R⊥∇E orthogonality. The stored energy may increase over a transition only as much
    as the reward 'supply' earns it:

        E(d') − E(d) ≤ supply     ⇒     penalize  relu(E(d') − E(d) − supply).

    A passivity / dissipativity inequality (storage function E, supply rate = reward;
    COP ≥ 1). ONE-SIDED: it penalizes only INEFFICIENT energy spending, and explicitly
    LETS the policy climb energy when it buys reward — what a running gait must do (build
    kinetic energy for forward-velocity reward), and what rigid orthogonality / the
    radius clamp forbade (pinning the latent at minimal energy ≈ standing still). The
    autonomous supply=0 case recovers a pure Lyapunov descent E(d') ≤ E(d)."""
    return torch.relu(energy(d_next) - energy(d) - supply).mean()


def contractive_axis_in_d(op_d, d_sub: Tensor) -> Tensor:
    """Smallest right-singular vector of op_d's state operator A_d at each sample — the
    most-contracted (minimal-energy) direction of the dynamics, in d-space. Detached:
    it is read off the current operator, not differentiated through (the orthogonality
    term moves the representation toward it, not it toward the representation)."""
    A, _ = op_d.operators(d_sub)
    # right singular vectors are the rows of Vh; S descending ⇒ last row = smallest
    _, _, Vh = torch.linalg.svd(A.float())
    return Vh[..., -1, :].detach().to(d_sub.dtype)
