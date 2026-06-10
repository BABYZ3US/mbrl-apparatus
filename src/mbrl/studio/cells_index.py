"""cells_index — the apparatus-side reader behind the ``pull.sweep`` verb.

Sweep OUTCOME grids live in ``results/*_cells.jsonl``, written by the synthetic /
benchmark scripts (scripts/bridge_experiment.py, scripts/spectral_benchmark.py, …).
Unlike ``runs/<run>/metrics.jsonl`` these are NOT time-series: each line is one
experimental CELL — scalar context fields (n, seed, noise_sigma, …) plus one nested
dict per ARM mapping metric → value::

    {"n": 512, "seed": 0,
     "frobenius_diag": {"lam": 0.01, "test_mse": 0.22, "wall_s": 0.007},
     "lap2_positive":  {"lam": 100.0, "test_mse": 141.2, ...}, ...}

``read_cells`` flattens that into per-(line, arm) ROWS the Studio's ablation panel
can bin directly::

    {"arm": "frobenius_diag", "n": 512, "seed": 0, "lam": 0.01, "test_mse": 0.22, ...}

List-valued fields are dropped (not binnable); strings are kept (categorical labels,
e.g. a recipe's "choice"); the names of numeric row fields are collected into
``fields`` so the panel can offer them as metric candidates. A line with NO nested
arm dicts degrades to a single row with ``arm: ""`` (flat-format tolerance).

Pure stdlib and torn-line tolerant, like run_index — safe to import inside the seal
(docs/remote_execution.md §1).
"""
from __future__ import annotations

import json
from pathlib import Path

_SUFFIX = "_cells.jsonl"


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, str, bool))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class CellsIndex:
    """Catalog + flatten the ``*_cells.jsonl`` sweep grids under a results root."""

    def __init__(self, results_root: Path | str) -> None:
        self.root = Path(results_root)

    def list_sweeps(self) -> list[dict]:
        """``[{name, file, n_lines}]`` for every ``*_cells.jsonl`` under the root.

        ``name`` is the filename minus the ``_cells.jsonl`` suffix — the handle
        ``read_cells``/``pull.sweep`` take. ``n_lines`` counts non-blank lines (a
        cheap size hint; parsing happens only on read). Missing root → ``[]``.
        """
        out: list[dict] = []
        if not self.root.exists():
            return out
        for p in sorted(self.root.glob(f"*{_SUFFIX}")):
            try:
                n = sum(1 for line in p.read_text().splitlines() if line.strip())
            except OSError:
                continue  # unreadable file: skip, don't fail the catalog
            out.append({"name": p.name[: -len(_SUFFIX)], "file": p.name, "n_lines": n})
        return out

    def read_cells(self, sweep: str) -> dict:
        """``{sweep, found, rows, arms, fields}`` — the flattened arm-rows.

        ``rows`` is one dict per (line, arm): the arm name under ``"arm"``, the
        line's scalar context fields, then the arm's scalar metrics (an arm metric
        shadows a same-named context field — the metric is the more specific value).
        ``arms`` is the sorted unique arm names; ``fields`` the sorted names of
        numeric row fields (candidate metrics/axes). Missing/unreadable file →
        ``found: False`` with empty lists; torn lines are skipped, never fatal.
        """
        path = self.root / f"{sweep}{_SUFFIX}"
        empty = {"sweep": sweep, "found": False, "rows": [], "arms": [], "fields": []}
        if not path.exists():
            return empty
        try:
            text = path.read_text()
        except OSError:
            return empty

        rows: list[dict] = []
        arms: set[str] = set()
        fields: set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cell = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write from a killed script; skip the line
            if not isinstance(cell, dict):
                continue
            context = {k: v for k, v in cell.items() if _is_scalar(v)}
            arm_dicts = {k: v for k, v in cell.items() if isinstance(v, dict)}
            if not arm_dicts:
                # flat-format line: the context itself is the one row
                arm_dicts = {"": {}}
            for arm in sorted(arm_dicts):
                row: dict = {"arm": arm}
                row.update(context)
                for k, v in arm_dicts[arm].items():
                    if _is_scalar(v):
                        row[k] = v
                rows.append(row)
                if arm:
                    arms.add(arm)
                for k, v in row.items():
                    if _is_number(v):
                        fields.add(k)
        return {"sweep": sweep, "found": True, "rows": rows,
                "arms": sorted(arms), "fields": sorted(fields)}
