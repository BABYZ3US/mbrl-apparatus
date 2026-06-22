"""Maximal Coding Rate Reduction — the expansion term R(Z) as an anti-collapse
regularizer (Yu, Chan, You, Song, Ma, "Learning Diverse and Discriminative
Representations via the Principle of Maximal Coding Rate Reduction", NeurIPS 2020).

The CODING RATE of a batch Z ∈ ℝ^{N×d} (rows = samples) is the number of nats per
sample needed to encode Z up to mean squared distortion eps²:

    R(Z) = ½ · logdet( I_d + (d / (N · eps²)) · ZᵀZ )                      (nats)

It is the log-volume of the ε-ball-packed representation: the rate-distortion of a
Gaussian source with covariance Z̄ᵀZ̄/N quantized to eps. A near-singular (collapsed)
Z packs into ~0 nats; an isotropic Z spread across all d directions packs into the most.

UNSUPERVISED ANTI-COLLAPSE USE. MCR² in full is ΔR = R(Z) − Σ_k R(Z_k | Π_k): expand
the whole batch, compress each class. With no labels we keep ONLY the expansion term
R(Z) and MAXIMIZE it — push the latent to fill the coding ball (maximal volume / high
effective rank), which is exactly an anti-collapse pressure. Since the trainer MINIMIZES,
maximizing R is adding −w·R(Z):

    loss += w * mcr2_loss(Z)   with   mcr2_loss(Z) = −R(Z)            (w ≥ 0)

This is composable with `rank2_frame.spectral_band_penalty` (which walls the spectrum
into [floor, ceiling]) and is the differentiable, exactly-scaled sibling of
`rank2_frame.log_det_barrier` (−mean ln(λ_i+eps)): both are spectrum-spreading volume
terms, but R(Z) is the rate-distortion form with eps as the physical quantization scale
and the d/eps² gain inside the logdet, rather than an ad-hoc ridge eps and an external
mean over log-eigenvalues.

Numerics. We form cov = Z̄ᵀZ̄ / N (the [d,d] sample covariance of the CENTERED batch)
and take logdet of I_d + (d/eps²)·cov. That matrix equals I_d + (d/(N·eps²))·Z̄ᵀZ̄ exactly
(the 1/N is folded into cov), so the rate is identical to the displayed formula — but the
intermediate is better-scaled (entries are O(variance), not O(N·variance)). I + SPD is
SPD, so logdet is finite and slogdet returns sign = +1; we use torch.linalg.slogdet for a
stable, differentiable logabsdet (no eig, no .item(), gradient flows to Z).
"""
from __future__ import annotations

import torch
from torch import Tensor


def coding_rate(Z: Tensor, eps: float = 0.5) -> Tensor:
    """Coding rate R(Z) = ½·logdet( I_d + (d/(N·eps²))·ZᵀZ ) in nats, for Z ∈ [N, d]
    (rows = samples). Returns a differentiable scalar.

    Z is centered (mean over the N samples subtracted) before the rate — standard, so the
    rate measures spread about the batch mean, not distance from the origin. Requires N ≥ 2
    (a single sample has zero centered covariance ⇒ rate 0 and an ill-posed estimate); for
    N < 2 returns a differentiable scalar 0 tied to Z's dtype/device.

    Implementation: cov = Z̄ᵀZ̄ / N  (a [d, d] SPD matrix);  M = I_d + (d/eps²)·cov  (the
    1/N folded into cov makes M ≡ I + (d/(N·eps²))·Z̄ᵀZ̄, the displayed matrix). slogdet ->
    (sign, logabsdet); M is SPD so sign = +1; return ½·logabsdet.
    """
    if Z.dim() != 2:
        raise ValueError("coding_rate expects Z of shape [N, d]; got shape %r" % (tuple(Z.shape),))
    N, d = Z.shape[0], Z.shape[1]
    if N < 2 or d == 0:
        # no spread to measure (or no features): rate is 0, kept differentiable in Z
        return Z.sum() * 0.0
    Zc = Z - Z.mean(0, keepdim=True)                       # center over the N samples
    cov = (Zc.transpose(-1, -2) @ Zc) / N                  # [d, d] sample covariance, SPD
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    gain = d / (eps * eps)                                 # the d/eps² rate-distortion gain
    M = eye + gain * cov                                   # I_d + (d/(N·eps²))·ZᵀZ  (SPD)
    sign, logabsdet = torch.linalg.slogdet(M)              # M SPD ⇒ sign = +1, logabsdet finite
    return 0.5 * logabsdet                                 # signed-but-+ logdet, halved


def mcr2_loss(Z: Tensor, eps: float = 0.5) -> Tensor:
    """The MCR² expansion term as a quantity to ADD to a MINIMIZED loss: −R(Z).

    Minimizing −R(Z) maximizes the coding rate, spreading Z to fill the coding ball
    (anti-collapse / maximal effective rank). Compose as  loss += w * mcr2_loss(Z)  with
    w ≥ 0. Differentiable; eps is the quantization/distortion scale (default 0.5).
    """
    return -coding_rate(Z, eps)
