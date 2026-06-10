"""Tests for mbrl.viz.surface_export — reward-surface slice, curvature, Hessian.

Checked against analytic ground truth (a quadratic), the way the apparatus tests
its math (PLAN.md §7). Needs torch + numpy.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from mbrl.viz import surface_export as sx


def test_curvature_of_quadratic_is_four_in_interior():
    # R[i,j] = i² + j² (unit spacing) -> discrete 5-point Laplacian = 4 everywhere interior
    n = 7
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    R = (ii**2 + jj**2).astype(float)
    curv = sx.curvature_field(R)
    assert np.allclose(curv[1:-1, 1:-1], 4.0)


def test_curvature_of_constant_and_linear_is_zero_interior():
    n = 6
    assert np.allclose(sx.curvature_field(np.full((n, n), 3.0)), 0.0)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lin = 2.0 * ii + 5.0 * jj
    assert np.allclose(sx.curvature_field(lin)[1:-1, 1:-1], 0.0)


def test_budget_is_frac_of_peak():
    curv = np.array([[0.0, 1.0], [2.0, 10.0]])
    assert sx.surface_budget(curv, frac=0.16) == 0.16 * 10.0


def test_reward_grid_matches_the_function():
    # f(x) = x0 + x1 on the (0,1) plane through the origin -> R = S + T
    f = lambda x: x[..., 0] + x[..., 1]
    u, v = sx.coordinate_plane_basis(2, 0, 1)
    R = sx.reward_grid(f, torch.zeros(2), u, v, extent=1.0, n=3)
    assert R.shape == (3, 3)
    assert np.isclose(R[1, 1], 0.0)           # center
    assert np.isclose(R[0, 0], -2.0)          # s=-1, t=-1
    assert np.isclose(R[2, 2], 2.0)           # s=+1, t=+1


def test_hessian_spectrum_of_quadratic():
    # f(x) = 3 x0² + 1 x1²  -> Hessian = diag(6, 2) -> eigenvalues [6, 2] (descending)
    a = torch.tensor([3.0, 1.0], dtype=torch.float64)
    f = lambda x: (a * x**2).sum(-1)
    eig = sx.hessian_spectrum(f, torch.tensor([0.5, -0.3], dtype=torch.float64))
    assert np.allclose(eig, [6.0, 2.0], atol=1e-6)


def test_hessian_spectrum_is_descending():
    a = torch.tensor([1.0, 5.0, 2.0], dtype=torch.float64)
    f = lambda x: (a * x**2).sum(-1)
    eig = sx.hessian_spectrum(f, torch.zeros(3, dtype=torch.float64))
    assert list(eig) == sorted(eig, reverse=True)
    assert np.allclose(sorted(eig), [2.0, 4.0, 10.0])  # 2*[1,2,5]


def test_export_surface_shape_and_payload():
    a = torch.tensor([2.0, 0.5, 1.0])
    f = lambda x: (a * x**2).sum(-1)
    payload = sx.export_surface(f, torch.zeros(3), plane=(0, 1), extent=2.0, n=9,
                                path=[[0.0, 0.0, 0], [0.1, 0.2, 100]],
                                step=1000, run="champ-Pendulum-v1-s0")
    assert len(payload["z"]) == 9 and len(payload["z"][0]) == 9
    assert len(payload["curv"]) == 9
    assert payload["plane"] == {"u": 0, "v": 1}
    assert payload["budget"] >= 0.0
    assert payload["path"][1] == [0.1, 0.2, 100]
    assert payload["step"] == 1000
    # round-trips through JSON (it's a wire payload)
    assert json.loads(json.dumps(payload))["n"] == 9


def test_write_surface_json_roundtrip(tmp_path):
    f = lambda x: (x**2).sum(-1)
    payload = sx.export_surface(f, torch.zeros(2), plane=(0, 1), n=5, step=2000,
                                run="r")
    p = sx.write_surface_json(payload, tmp_path, "r", 2000)
    assert p.name == "surface_s2000.json"
    assert json.loads(p.read_text())["n"] == 5
