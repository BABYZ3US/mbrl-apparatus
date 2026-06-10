"""diagnostics_index — the apparatus-side reader behind ``pull.diagnostics``.

Diagnostics artifacts (PCA scree + cross-validation reports; written by
scripts/diagnose.py via mbrl.diagnostics.export) live as one JSON per report at
``results/diagnostics/<name>.json``. Same catalog-or-named contract as
CellsIndex: no name -> the catalog; a name -> that report. Pure stdlib,
torn-file tolerant — safe inside the seal.
"""
from __future__ import annotations

import json
from pathlib import Path


class DiagnosticsIndex:
    def __init__(self, results_root: Path | str) -> None:
        self.root = Path(results_root) / "diagnostics"

    def list_reports(self) -> list[dict]:
        """``[{name, file, created?}]`` for every readable report, sorted by name."""
        out: list[dict] = []
        if not self.root.exists():
            return out
        for p in sorted(self.root.glob("*.json")):
            row = {"name": p.stem, "file": p.name}
            try:
                doc = json.loads(p.read_text())
                if isinstance(doc, dict) and "created" in doc:
                    row["created"] = str(doc["created"])
            except (OSError, json.JSONDecodeError):
                continue  # torn/unreadable: skip, don't fail the catalog
            out.append(row)
        return out

    def get_report(self, name: str) -> dict:
        """The full report payload, or ``{name, found: False}``."""
        p = self.root / f"{name}.json"
        if not p.exists():
            return {"name": name, "found": False}
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {"name": name, "found": False}
        if not isinstance(doc, dict):
            return {"name": name, "found": False}
        doc.setdefault("name", name)
        doc["found"] = True
        return doc
