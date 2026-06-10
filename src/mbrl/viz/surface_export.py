"""surface_export — structured reward-surface + Hessian-spectrum data for the Studio.

The Godot `viz-reward` panel renders a reward height-field colored by curvature
against a Sobolev-ball budget plane (godot_studio/scripts/viz/reward_surface.gd —
fed today by a SYNTHETIC field). This module produces the REAL thing from a trained
reward head: a 2-plane slice of R-hat, its curvature field, the budget line, and
(separately) the reward-Hessian eigenvalue spectrum — and writes them as a JSON
artifact that the stdlib `mbrl.studio.surface_index` serves over pull.surface. The
heavy torch work happens HERE (during/after training), never in the boundary server
(docs/remote_execution.md §1).

Curvature mirrors core_ml.gd::curvature_field EXACTLY — |5-point Laplacian| with
edge-replicated boundaries, unit index spacing — so a live surface and the synthetic
one look identical in the panel. Budget = frac × peak curvature
(frac = 0.16 = RewardSurface.CURV_BUDGET_FRAC).

Torch + numpy (NOT stdlib) — lives in viz/, never imported by the bridge server.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

# x: (..., d) -> (...,) scalar per row. RewardModel.on_concat fits this directly.
RewardFn = Callable[[Tensor], Tensor]

CURV_BUDGET_FRAC = 0.16  # mirror RewardSurface.CURV_BUDGET_FRAC (godot)


def coordinate_plane_basis(d: int, i: int, j: int) -> tuple[Tensor, Tensor]:
    """Unit basis vectors e_i, e_j spanning the (i, j) coordinate plane in R^d."""
    if not (0 <= i < d and 0 <= j < d and i != j):
        raise ValueError(f"plane axes (i={i}, j={j}) invalid for d={d}")
    u = torch.zeros(d)
    v = torch.zeros(d)
    u[i] = 1.0
    v[j] = 1.0
    return u, v


@torch.no_grad()
def reward_grid(f: RewardFn, center: Tensor, u: Tensor, v: Tensor,
                extent: float = 2.0, n: int = 81) -> np.ndarray:
    """Evaluate f on the n×n plane center + s·u + t·v, s,t ∈ [-extent, extent]."""
    ts = torch.linspace(-extent, extent, n)
    S, T = torch.meshgrid(ts, ts, indexing="ij")
    d = center.shape[-1]
    X = center + S[..., None] * u + T[..., None] * v       # (n, n, d)
    R = f(X.reshape(-1, d)).reshape(n, n)
    return R.detach().cpu().numpy().astype(np.float64)


def curvature_field(R: np.ndarray) -> np.ndarray:
    """|5-point discrete Laplacian| with edge-replicated boundaries.

    The core_ml.gd::curvature_field convention exactly (unit index spacing, clamped
    edge neighbors, absolute value) so live == synthetic in the panel.
    """
    P = np.pad(R, 1, mode="edge")
    lap = P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] - 4.0 * R
    return np.abs(lap)


def curvature_energy(R: np.ndarray) -> float:
    """Mean |Laplacian| — the H² seminorm proxy (core_ml.gd::curvature_energy)."""
    curv = curvature_field(R)
    return float(curv.mean()) if curv.size else 0.0


def surface_budget(curv: np.ndarray, frac: float = CURV_BUDGET_FRAC) -> float:
    """Sobolev-ball budget line = frac × peak curvature."""
    return frac * (float(curv.max()) if curv.size else 0.0)


def hessian_spectrum(f: RewardFn, x0: Tensor) -> np.ndarray:
    """Eigenvalues (DESCENDING) of the Hessian of scalar f at x0.

    For f(x) = xᵀA x with symmetric A the Hessian is 2A — the analytic check the
    test pins. Symmetrized before eigvalsh to shed numerical asymmetry.
    """
    x0 = x0.detach().clone()  # operate in x0's dtype — match the model's params

    def scalar(x: Tensor) -> Tensor:  # f wants a batch dim; give one, take it back
        return f(x.unsqueeze(0)).squeeze(0)

    H = torch.autograd.functional.hessian(scalar, x0)
    H = 0.5 * (H + H.transpose(-1, -2))
    eig = torch.linalg.eigvalsh(H)               # ascending, real (symmetric)
    return eig.detach().cpu().numpy()[::-1].copy()  # descending


def export_surface(f: RewardFn, center: Tensor, *, plane=(0, 1),
                   extent: float = 2.0, n: int = 81, frac: float = CURV_BUDGET_FRAC,
                   path: list | None = None, step: int | None = None,
                   run: str | None = None) -> dict:
    """The pull.surface payload: {z, curv, budget, path, plane, ...}.

    `plane` is a pair of coordinate-axis indices (i, j) in the (z, a[, τ]) space.
    `path` (optional) is the optimizer trajectory as [[u, v, step], ...] in plane
    coords — passed through (collected during training), not computed here.
    """
    d = center.shape[-1]
    i, j = int(plane[0]), int(plane[1])
    u, v = coordinate_plane_basis(d, i, j)
    R = reward_grid(f, center, u, v, extent=extent, n=n)
    curv = curvature_field(R)
    return {
        "z": R.tolist(),
        "curv": curv.tolist(),
        "budget": surface_budget(curv, frac),
        "path": list(path) if path else [],
        "plane": {"u": i, "v": j},
        "extent": float(extent),
        "n": int(n),
        "step": step,
        "run": run,
    }


def write_surface_json(payload: dict, results_root, run: str, step: int) -> Path:
    """Write a surface artifact to results/runs/<run>/surfaces/surface_s<step>.json."""
    out_dir = Path(results_root) / "runs" / run / "surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"surface_s{int(step)}.json"
    out.write_text(json.dumps(payload))
    return out
