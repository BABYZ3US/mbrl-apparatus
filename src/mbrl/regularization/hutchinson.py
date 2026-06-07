"""Hutchinson-estimated isotropic Hessian penalty (R4, R5, R16).

E_v ||H v||^2 with Rademacher v unbiasedly estimates ||H||_F^2; each H v is one
Hessian-vector product via double backward. 2 probes is the validated default.
The penalty is isotropic by construction (R16): never weight eigendirections.

Computed in fp32 even under bf16 autocast — second derivatives are noise-sensitive.
"""
from __future__ import annotations

import torch
from torch import Tensor


def rademacher_like(x: Tensor, generator: torch.Generator | None = None) -> Tensor:
    # Sample on the generator's device (e.g. CPU fallback generator under MPS),
    # then move — keeps probe streams reproducible across backends.
    dev = generator.device if generator is not None else x.device
    v = torch.randint(0, 2, x.shape, generator=generator, device=dev,
                      dtype=x.dtype).mul_(2).sub_(1)
    return v.to(x.device) if dev != x.device else v


def hvp_penalty(
    fn,
    inputs: Tensor,
    n_probes: int = 2,
    generator: torch.Generator | None = None,
    create_graph: bool = True,
) -> Tensor:
    """Unbiased estimate of mean_batch ||nabla^2 fn||_F^2 at `inputs`.

    fn: callable mapping inputs (B, d) -> scalar-per-sample (B,). For a reward
        model, pass lambda x: R(x[..., :k], x[..., k:]).squeeze(-1) with x = cat(z, a).
    inputs: (B, d) leaf tensor; penalty taken w.r.t. these coordinates
        (latent coords, per experiment 2.3's caveat).
    n_probes: Hutchinson probes; 2 validated (R4). 1 probe is the *biased*
        underperformer only if you square a single trace estimate — squaring
        ||Hv||^2 per-probe stays unbiased at any N; variance shrinks with N.
    create_graph: keep True during training so the penalty is differentiable.
    """
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        x = inputs.float().detach().requires_grad_(True)
        out = fn(x)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)

        pen = x.new_zeros(())
        for _ in range(n_probes):
            v = rademacher_like(x, generator)
            (hv,) = torch.autograd.grad(
                (grad * v).sum(), x, create_graph=create_graph, retain_graph=True
            )
            pen = pen + hv.pow(2).sum(dim=-1).mean()
        return pen / n_probes


def laplacian_trace_penalty(
    fn,
    inputs: Tensor,
    n_probes: int = 2,
    generator: torch.Generator | None = None,
    create_graph: bool = True,
    clamp: bool = True,
) -> Tensor:
    """(Delta R)^2 estimator via two *independent* probe estimates of the trace
    multiplied together (R5: same Euler-Lagrange as Frobenius).

    clamp=True (default) applies the thermodynamic-consistency clamp from the
    original findings (report section 3): the product (v1' H v1)(v2' H v2) is
    sign-indefinite, and the original experiments found NON-NEGATIVITY is the
    operative property — unclamped product −79, clamped +41 (beating Frobenius
    −40 in that run set). Clamping max(est, 0) per sample trades a little bias
    for the sign constraint; the user's narrowed-down recipe ('clamped decaying
    trace') is this estimator under a decaying schedule."""
    if n_probes < 2:
        raise ValueError("(Delta R)^2 product estimator needs >= 2 independent probes (R5).")
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        x = inputs.float().detach().requires_grad_(True)
        out = fn(x)
        (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)

        traces = []
        for _ in range(n_probes):
            v = rademacher_like(x, generator)
            (hv,) = torch.autograd.grad(
                (grad * v).sum(), x, create_graph=create_graph, retain_graph=True
            )
            traces.append((v * hv).sum(dim=-1))  # v^T H v, unbiased for tr(H)
        # product of independent estimates -> unbiased for tr(H)^2 (before clamp)
        t1, t2 = traces[0], traces[1]
        est = t1 * t2
        if clamp:  # per-sample non-negativity (report section 3)
            est = est.clamp_min(0)
        for i in range(2, n_probes - 1, 2):
            pair = traces[i] * traces[i + 1]
            est = est + (pair.clamp_min(0) if clamp else pair)
        return est.mean() / max(1, (n_probes // 2))
