"""Tests for mbrl.studio.surface_index — the pull.surface reader (v0.1 M4).

Pure stdlib + pytest tmp_path. Mirrors the artifact layout written by
mbrl.viz.surface_export.write_surface_json.
"""
from __future__ import annotations

import json
from pathlib import Path

from mbrl.studio.surface_index import SurfaceIndex


def _write_surface(root: Path, run: str, step: int, payload: dict) -> None:
    d = root / "runs" / run / "surfaces"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"surface_s{step}.json").write_text(json.dumps(payload))


def test_list_surfaces_sorted_by_step(tmp_path):
    _write_surface(tmp_path, "r", 2000, {"n": 5, "step": 2000})
    _write_surface(tmp_path, "r", 1000, {"n": 5, "step": 1000})
    steps = [s["step"] for s in SurfaceIndex(tmp_path).list_surfaces("r")]
    assert steps == [1000, 2000]


def test_get_surface_latest_by_default(tmp_path):
    _write_surface(tmp_path, "r", 1000, {"step": 1000, "budget": 0.1})
    _write_surface(tmp_path, "r", 3000, {"step": 3000, "budget": 0.3})
    assert SurfaceIndex(tmp_path).get_surface("r")["step"] == 3000


def test_get_surface_specific_step(tmp_path):
    _write_surface(tmp_path, "r", 1000, {"step": 1000})
    _write_surface(tmp_path, "r", 2000, {"step": 2000})
    assert SurfaceIndex(tmp_path).get_surface("r", step=1000)["step"] == 1000


def test_unknown_run_or_step_is_empty(tmp_path):
    _write_surface(tmp_path, "r", 1000, {"step": 1000})
    idx = SurfaceIndex(tmp_path)
    assert idx.get_surface("ghost") == {}
    assert idx.get_surface("r", step=9999) == {}
    assert idx.list_surfaces("ghost") == []


def test_torn_surface_file_returns_empty(tmp_path):
    d = tmp_path / "runs" / "r" / "surfaces"
    d.mkdir(parents=True)
    (d / "surface_s1000.json").write_text("{ not valid json")
    assert SurfaceIndex(tmp_path).get_surface("r", step=1000) == {}
