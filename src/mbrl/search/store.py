"""SearchStore — persisted state for an ongoing random search (W9).

One JSON file per search under ``<results_root>/searches/<name>.json``,
written atomically (tmp + replace) and read torn-tolerantly ({} on damage) —
the same file discipline as the artifact manifests. The state embeds each
arm's full spec so a tick can launch queued arms without re-deriving them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class SearchStore:
    def __init__(self, results_root) -> None:
        self.dir = Path(results_root) / "searches"

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def list_names(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def load(self, name: str) -> dict:
        try:
            obj = json.loads(self.path(name).read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}

    def save(self, name: str, state: dict) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path(name).with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str))
        os.replace(tmp, self.path(name))
        return self.path(name)
