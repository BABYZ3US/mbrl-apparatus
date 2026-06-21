"""tensor_index — stdlib reader for named-tensor artifacts (backs pull.tensor / pull.salience).

Named tensors (the latent Gram G = E[zzᵀ] served as "gram", the twin operator
matrices op_d/op_p, and the reward input/feature salience |∂reward/∂obs|) are
written by the GATED snapshot hooks in mbrl.training.loop to
results/runs/<run>/tensors/<name>_<step>.json (the heavy torch work happens there,
out of the boundary). This reader is stdlib-only so studio_bridge_server.py can
serve pull.tensor / pull.salience from inside the seal — the EXACT pattern
surface_index uses for pull.surface (docs/remote_execution.md §1).

The on-disk layout mirrors surfaces/ one level over: surface_index reads
<root>/runs/<run>/surfaces/surface_s<step>.json; this reads
<root>/runs/<run>/tensors/<name>_<step>.json. TensorIndex is constructed the SAME
way SurfaceIndex is — with the results ROOT (the bridge passes
self.results_dir.parent, since results_dir is <root>/runs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# A tensor artifact is "<base name>_<step>.json"; the step is the trailing integer.
# The base name may itself contain underscores (e.g. reward_input_salience), so the
# split is on the LAST underscore-then-digits-then-.json.
_NAME_STEP_RE = re.compile(r"^(?P<name>.+)_(?P<step>\d+)\.json$")


class TensorIndex:
    """Read-only index over the per-run named-tensor artifacts.

    Constructed the SAME way SurfaceIndex is (with the results ROOT). The runs dir
    is <root>/runs and each run's tensors live under <root>/runs/<run>/tensors/.

    >>> idx = TensorIndex("results")
    >>> idx.list_tensors("champ-Pendulum-v1-s0")        # {"items": ["gram", ...]}
    >>> idx.get_tensor("champ-Pendulum-v1-s0", "gram")  # latest step -> payload dict
    >>> idx.get_tensor("champ-Pendulum-v1-s0", "gram", step=2000)
    """

    def __init__(self, results_root):
        self.runs_dir = Path(results_root) / "runs"

    def _dir(self, run: str) -> Path:
        return self.runs_dir / run / "tensors"

    def _steps_for(self, run: str, name: str) -> list[int]:
        """The steps (ascending) at which `<name>` was snapshotted ([] if none)."""
        d = self._dir(run)
        steps: list[int] = []
        if d.is_dir():
            for p in d.glob(f"{name}_*.json"):
                m = _NAME_STEP_RE.match(p.name)
                if m and m.group("name") == name:
                    steps.append(int(m.group("step")))
        return sorted(steps)

    def list_tensors(self, run: str) -> dict:
        """Catalog of distinct base names under the run's tensors/ dir.

        Strips the `_<step>.json` suffix to recover each base name; returns
        {"items": [name, ...]} (sorted, deduplicated; empty list if the run has no
        tensors dir) — never raises (a missing dir is a normal query).
        """
        d = self._dir(run)
        names: set[str] = set()
        if d.is_dir():
            for p in d.glob("*.json"):
                m = _NAME_STEP_RE.match(p.name)
                if m:
                    names.add(m.group("name"))
        return {"items": sorted(names)}

    def get_tensor(self, run: str, name: str, step: int | None = None) -> dict:
        """The parsed JSON payload for one named tensor.

        Reads <run>/tensors/<name>_<step>.json (the latest step when `step` is None
        — the max snapshotted step). Returns the parsed dict (e.g.
        {"run","name","step","matrix"|"vector"|"salience","eig"?}), or {} if the
        run/name/step is absent or the file is torn — mirrors SurfaceIndex (a
        missing tensor is a normal query; the panel shows "none", it never raises).
        """
        if not name:
            return {}
        steps = self._steps_for(run, name)
        if not steps:
            return {}
        if step is None:
            chosen = steps[-1]
        elif step in steps:
            chosen = step
        else:
            return {}
        path = self._dir(run) / f"{name}_{chosen}.json"
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}
