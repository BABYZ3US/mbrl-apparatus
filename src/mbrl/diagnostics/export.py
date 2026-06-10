"""export — write a diagnostics artifact the bridge can serve (pull.diagnostics).

Mirrors viz/surface_export.write_surface_json: one JSON file per named report,
under results/diagnostics/<name>.json. Stdlib only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def write_diagnostics_json(payload: dict, results_root, name: str) -> Path:
    """Write results/diagnostics/<name>.json (creating dirs); returns the path."""
    out_dir = Path(results_root) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = dict(payload)
    doc.setdefault("name", name)
    doc.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(doc, indent=2))
    return out
