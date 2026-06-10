"""Local JSONL metrics mirror — makes every run figure-ready without network.

Each run writes results/runs/<run_name>/meta.json + metrics.jsonl (one JSON
object per log call). W&B remains the cloud mirror of the same keys; figures
can be regenerated from either source (scripts/make_figures.py --source).
"""
from __future__ import annotations

import json
from pathlib import Path

from mbrl.utils.metric_store import MetricStore


class MetricsLogger:
    def __init__(self, root: str | Path, run_name: str, meta: dict | None = None,
                 config: dict | None = None):
        self.dir = Path(root) / "runs" / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        if meta:
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        if config:
            # the run's RESOLVED config (W8: pull.artifacts serves it — spec
            # diffing + resume need the exact dict the run trained with)
            (self.dir / "config.json").write_text(
                json.dumps(config, indent=1, default=str))
        self._fh = open(self.dir / "metrics.jsonl", "a", buffering=1)  # line-buffered
        # Buffered SQLite mirror (metrics.db). The studio bridge reads it back lock-free
        # under WAL. Dual-write: JSONL stays the canonical source, the db is additive.
        # Any store failure disables the mirror but never breaks training.
        self._store: MetricStore | None = None
        try:
            self._store = MetricStore(self.dir / "metrics.db")
        except Exception:
            self._store = None

    def log(self, metrics: dict):
        self._fh.write(json.dumps(
            {k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else v)
             for k, v in metrics.items()}) + "\n")
        if self._store is not None:
            try:
                self._store.append(metrics.get("env_steps", metrics.get("step")), metrics)
            except Exception:
                self._store = None  # disable on first failure; never break the run

    def close(self):
        self._fh.close()
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass


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
