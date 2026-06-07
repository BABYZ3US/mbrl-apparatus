"""Episode-level reward smoothing (DreamSmooth; recipe ingredient, framework 2.5).

Both ingredients contribute to the headline: Hessian-only -89, DreamSmooth-only
-137, full recipe +98 (from -165 baseline).
"""
from __future__ import annotations

import torch
from torch import Tensor


def smooth_rewards(rs: Tensor, cfg) -> Tensor:
    """rs: (H, B) imagined rewards. Gaussian smoothing along the horizon axis."""
    if not cfg.enabled or rs.shape[0] < 3:
        return rs
    sigma = cfg.sigma
    radius = max(1, int(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=rs.device, dtype=rs.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    x = rs.T.unsqueeze(1)  # (B, 1, H)
    sm = torch.nn.functional.conv1d(x, kernel, padding=radius)
    return sm.squeeze(1).T
