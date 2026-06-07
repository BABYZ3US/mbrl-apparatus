"""Reward-surface slices: R-hat along random 2-planes in (z, a) space.

The 'spikiness picture' — visualizes what the curvature penalty suppresses (R1).
Render before/after regularization side by side.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def surface_slice(reward_model, z0: torch.Tensor, a0: torch.Tensor,
                  extent: float = 2.0, n: int = 81, seed: int = 0):
    """Evaluate R-hat on a random 2-plane through (z0, a0). Returns (U, V, R)."""
    g = torch.Generator().manual_seed(seed)
    d = z0.shape[-1] + a0.shape[-1]
    u = torch.randn(d, generator=g); u /= u.norm()
    v = torch.randn(d, generator=g); v -= (v @ u) * u; v /= v.norm()
    ts = torch.linspace(-extent, extent, n)
    U, V = torch.meshgrid(ts, ts, indexing="ij")
    x0 = torch.cat([z0, a0], dim=-1)
    X = x0 + U[..., None] * u + V[..., None] * v          # (n, n, d)
    R = reward_model.on_concat(X.reshape(-1, d)).reshape(n, n)
    return U.numpy(), V.numpy(), R.numpy()


def plot_surface_pair(before, after, titles=("unregularized", "curvature-regularized")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), subplot_kw={"projection": "3d"})
    for ax, (U, V, R), t in zip(axes, (before, after), titles):
        ax.plot_surface(U, V, R, cmap="viridis", linewidth=0)
        ax.set_title(t)
    fig.suptitle(r"$\hat R$ on a random latent 2-plane")
    return fig
