"""Local JSONL metrics mirror — makes every run figure-ready without network.

Each run writes results/runs/<run_name>/meta.json + metrics.jsonl (one JSON
object per log call). W&B remains the cloud mirror of the same keys; figures
can be regenerated from either source (scripts/make_figures.py --source).
"""
from __future__ import annotations

import json
from pathlib import Path


class MetricsLogger:
    def __init__(self, root: str | Path, run_name: str, meta: dict | None = None):
        self.dir = Path(root) / "runs" / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        if meta:
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        self._fh = open(self.dir / "metrics.jsonl", "a", buffering=1)  # line-buffered

    def log(self, metrics: dict):
        self._fh.write(json.dumps(
            {k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else v)
             for k, v in metrics.items()}) + "\n")

    def close(self):
        self._fh.close()


def load_runs(root: str | Path, group: str | None = None) -> dict[str, dict]:
    """Read all local runs -> {run_name: {"meta": dict, "history": list[dict]}}.
    Filter by meta["group"] if group given. Tolerates partial/crashed runs."""
    out = {}
    runs_dir = Path(root) / "runs"
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir()):
        mf, hf = d / "meta.json", d / "metrics.jsonl"
        if not hf.exists():
            continue
        meta = json.loads(mf.read_text()) if mf.exists() else {}
        if group and meta.get("group") != group:
            continue
        history = []
        for line in hf.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn write from a killed run; skip
        out[d.name] = {"meta": meta, "history": history}
    return out
