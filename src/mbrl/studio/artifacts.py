"""artifacts — the boundary-readable manifest of a run's W&B / local artifacts.

The training side logs W&B artifacts (checkpoints via CheckpointManager, replay shards
via collect.py) — but the one-boundary server CANNOT call the W&B API (it would drag
wandb across the seal, docs/remote_execution.md §1). So the training side ALSO writes a
tiny stdlib JSON manifest per run, and the boundary reads THAT to answer pull.artifacts.

    <results_root>/runs/<run>/artifacts.json   # a JSON list of artifact entries

An entry is an open dict, conventionally::

    {"name": "model-<wandb_id>", "type": "checkpoint", "tag": "step2000",
     "env_steps": 200000, "cfg_hash": "abc123", "path": "<local>", "uri": "wandb://..."}

``record_artifact`` UPSERTS by ``name`` — a checkpoint that logs many versions under one
artifact name stays ONE manifest entry (latest state). Pure stdlib — safe on both sides.
"""
from __future__ import annotations

import json
from pathlib import Path


def _manifest_path(results_root, run: str) -> Path:
    return Path(results_root) / "runs" / str(run) / "artifacts.json"


def list_artifacts(results_root, run: str) -> list[dict]:
    """The run's artifact entries ([] if the manifest is absent or torn)."""
    try:
        obj = json.loads(_manifest_path(results_root, run).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [e for e in obj if isinstance(e, dict)] if isinstance(obj, list) else []


def record_artifact(results_root, run: str, entry: dict) -> Path:
    """Upsert an artifact entry (by ``entry['name']``) into the run's manifest.

    Best-effort + atomic (tmp + replace). Upsert-by-name keeps a many-version artifact
    (a checkpoint logged every N steps) as a single, latest entry rather than a flood.
    """
    path = _manifest_path(results_root, run)
    name = entry.get("name")
    items = [e for e in list_artifacts(results_root, run) if e.get("name") != name]
    items.append(dict(entry))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=1, default=str))
    tmp.replace(path)
    return path
