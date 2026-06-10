"""surface_index — stdlib reader for reward-surface artifacts (backs pull.surface).

Reward surfaces are written by mbrl.viz.surface_export.write_surface_json to
results/runs/<run>/surfaces/surface_s<step>.json (the heavy torch work happens
there, out of the boundary). This reader is stdlib-only so studio_bridge_server.py
can serve pull.surface from inside the seal — the same pattern run_index uses for
pull.metric (docs/remote_execution.md §1).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_STEP_RE = re.compile(r"^surface_s(\d+)\.json$")


class SurfaceIndex:
    """Read-only index over the per-run reward-surface artifacts.

    >>> idx = SurfaceIndex("results")
    >>> idx.list_surfaces("champ-Pendulum-v1-s0")     # [{step, path}, ...]
    >>> idx.get_surface("champ-Pendulum-v1-s0")       # latest -> pull.surface payload
    >>> idx.get_surface("champ-Pendulum-v1-s0", step=2000)
    """

    def __init__(self, results_root):
        self.runs_dir = Path(results_root) / "runs"

    def _dir(self, run: str) -> Path:
        return self.runs_dir / run / "surfaces"

    def list_surfaces(self, run: str) -> list[dict]:
        """Surface artifacts for a run, ascending by step ([] if none)."""
        d = self._dir(run)
        out: list[dict] = []
        if d.is_dir():
            for p in d.glob("surface_s*.json"):
                m = _STEP_RE.match(p.name)
                if m:
                    out.append({"step": int(m.group(1)), "path": str(p)})
        return sorted(out, key=lambda r: r["step"])

    def get_surface(self, run: str, step: int | None = None) -> dict:
        """The {z, curv, budget, path, ...} payload for one surface.

        step=None -> the latest; an unknown run/step (or a torn file) -> {} rather
        than raising (a missing surface is a normal query — the panel shows "none").
        """
        items = self.list_surfaces(run)
        if not items:
            return {}
        if step is None:
            chosen = items[-1]
        else:
            chosen = next((it for it in items if it["step"] == step), None)
            if chosen is None:
                return {}
        try:
            obj = json.loads(Path(chosen["path"]).read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}
