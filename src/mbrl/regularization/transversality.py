"""Transversality angle between reward and dynamics Hessians (R8 diagnostic).

Estimates the angle alpha between nabla^2 R and nabla^2 T in Frobenius space via
Hutchinson probes: <A, B>_F = E_v [ (A v) . (B v) ]. Logged each eval to track the
empirically observed 60-71 degree range and correlate with sample efficiency (R9).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .hutchinson import rademacher_like


def effective_dim(fn, inputs: Tensor, n_probes: int = 16,
                  generator: torch.Generator | None = None) -> float:
    """Effective dimension of fn's Hessian spectrum via the participation ratio
    of H^2:  d_eff = tr(H^2)^2 / tr(H^4) = (sum lam_i^2)^2 / sum lam_i^4.

    Sign-robust (uses lam^2), and both traces come from the Hutchinson machinery
    we already run: tr(H^2) = E||Hv||^2 (the penalty itself), tr(H^4) = E||H(Hv)||^2
    (one nested HVP). Uniform spectrum over k directions => d_eff = k; a spiked
    spectrum => d_eff ~ 1. The theory's prediction: the H^2 penalty's spectral
    filtering pushes d_eff DOWN on its own over training, which is what makes the
    multi-kernel benefit sqrt((d_eff-kappa)/d_eff) grow."""
    x = inputs.float().detach().requires_grad_(True)
    (g,) = torch.autograd.grad(fn(x).sum(), x, create_graph=True)
    B = x.shape[0]
    tr_h2 = x.new_zeros(B)
    tr_h4 = x.new_zeros(B)
    for _ in range(n_probes):
        v = rademacher_like(x, generator)
        (hv,) = torch.autograd.grad((g * v).sum(), x, retain_graph=True,
                                    create_graph=True)
        (h2v,) = torch.autograd.grad((g * hv.detach()).sum(), x, retain_graph=True)
        tr_h2 += hv.detach().pow(2).sum(dim=-1)   # per-sample ||Hv||^2
        tr_h4 += h2v.detach().pow(2).sum(dim=-1)  # per-sample ||H^2 v||^2
    # Per-sample ratio THEN aggregate (median, robust). Pooling the traces over
    # the batch first is wrong under heterogeneous curvature: samples with huge
    # local Hessians dominate tr(H^4) quadratically and drive the pooled ratio
    # below 1, which is impossible for a true participation ratio.
    keep = tr_h2 > 1e-10 * tr_h2.max().clamp_min(1e-30)  # drop flat samples
    if not keep.any():
        return 0.0
    d = tr_h2[keep].pow(2) / (n_probes * tr_h4[keep] + 1e-30)
    return d.median().item()


def transversality_angle(fn_r, fn_t, inputs: Tensor, n_probes: int = 8,
                         generator: torch.Generator | None = None) -> float:
    """Angle (degrees) between Hessians of fn_r and fn_t at `inputs` (batch-averaged).

    fn_t should map x -> scalar per sample (e.g. a fixed random projection of the
    dynamics output, or its sum — choose once and keep fixed across training).
    """
    x = inputs.float().detach().requires_grad_(True)
    (gr,) = torch.autograd.grad(fn_r(x).sum(), x, create_graph=True)
    (gt,) = torch.autograd.grad(fn_t(x).sum(), x, create_graph=True)

    dot = rr = tt = 0.0
    for _ in range(n_probes):
        v = rademacher_like(x, generator)
        (hrv,) = torch.autograd.grad((gr * v).sum(), x, retain_graph=True)
        (htv,) = torch.autograd.grad((gt * v).sum(), x, retain_graph=True)
        dot += (hrv * htv).sum(dim=-1).mean().item()
        rr += hrv.pow(2).sum(dim=-1).mean().item()
        tt += htv.pow(2).sum(dim=-1).mean().item()
    cos = dot / (math.sqrt(rr * tt) + 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
