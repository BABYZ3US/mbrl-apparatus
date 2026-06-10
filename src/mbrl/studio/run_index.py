"""run_index — the apparatus-side query layer behind the Studio's pull.* reads.

The Studio's data panel asks the one-boundary server (scripts/studio_bridge_server.py)
for run lists, metric curves, and dataset/checkpoint catalogs via the Bridge verbs
``pull.runs`` / ``pull.metric`` / ``pull.datasets``. This module turns the on-disk
artifacts those runs leave behind into the JSON payloads those verbs return.

On-disk format (owned by ``mbrl.utils.metrics_logger`` — mirrored here, deliberately
NOT imported, because ``mbrl.utils.__init__`` pulls torch/wandb and would break the
studio seal, docs/remote_execution.md section 1)::

    <results_root>/runs/<run_name>/meta.json       # {"group": ..., ...} (optional)
    <results_root>/runs/<run_name>/metrics.jsonl   # one JSON object per log() call

Checkpoints (owned by ``mbrl.utils.checkpoint.CheckpointManager`` — mirrored here)::

    <ckpt_root>/<cfg_hash>/ckpt_<tag>.pt           # tag in {step<N>, best, milestone<N>}

Pure stdlib. No torch, no wandb, no yaml — safe to import inside the boundary.
Tolerant of partial/torn writes from killed runs (skips unparseable lines/files).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# Candidate x-axis keys, best first. PLAN.md section 3: always log REAL env steps as
# the sample-efficiency x-axis. We try these per history entry; if none is present we
# fall back to the entry's ordinal index so a curve still renders.
_STEP_KEYS = ("env_steps", "total_env_steps", "global_step", "step", "_step")

# Group fallback when a run has no meta.json["group"]: strip the trailing seed suffix
# from the run name (matches the legacy server scan_runs + scripts/status.py rule, so
# old metric-less runs group the same way they always did).
_SEED_SUFFIX = re.compile(r"-s\d+$")


def _read_jsonl(path: Path) -> list[dict]:
    """Read a metrics.jsonl, skipping blank/torn lines (killed-run tolerant)."""
    rows: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn write from a killed run
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_meta(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _is_number(v) -> bool:
    # bool is a subclass of int — exclude it; a flag is not a metric value.
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _step_of(entry: dict, ordinal: int) -> float:
    for k in _STEP_KEYS:
        if _is_number(entry.get(k)):
            return float(entry[k])
    return float(ordinal)


def _tag_step(tag: str) -> int | None:
    """``step1234`` -> 1234, ``milestone100000`` -> 100000, ``best`` -> None."""
    for prefix in ("step", "milestone"):
        if tag.startswith(prefix) and tag[len(prefix):].isdigit():
            return int(tag[len(prefix):])
    return None


@dataclass(frozen=True)
class RunInfo:
    name: str
    group: str | None
    last_step: float
    n_points: int
    keys: list[str]  # metric keys seen (step keys excluded)

    def as_dict(self) -> dict:
        return asdict(self)


class RunIndex:
    """Read-only index over a results root (and optional checkpoint root).

    >>> idx = RunIndex("results")                       # results/runs/<name>/...
    >>> idx.list_runs(group="champion")                 # -> pull.runs payload
    >>> idx.get_metric("champion-Pendulum-v1-s0", "episode_return")  # -> pull.metric
    >>> idx.list_datasets(kind="checkpoint")            # -> pull.datasets payload
    """

    def __init__(self, results_root, ckpt_root=None, shards_root=None):
        self.results_root = Path(results_root)
        self.runs_dir = self.results_root / "runs"
        # CheckpointManager's default lives at <repo>/checkpoints (PLAN.md section 8);
        # the caller points us at it. None => no checkpoint scanning.
        self.ckpt_root = Path(ckpt_root) if ckpt_root is not None else None
        # Replay-buffer shards land in <results_root>/shards by default (collect.py
        # --out). None disables buffer cataloguing.
        self.shards_root = (Path(shards_root) if shards_root is not None
                            else self.results_root / "shards")

    # ---------------- runs (pull.runs) ----------------
    def _run_dirs(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            return []
        return [d for d in sorted(self.runs_dir.iterdir())
                if d.is_dir() and (d / "metrics.jsonl").exists()]

    def run_info(self, name: str) -> RunInfo | None:
        d = self.runs_dir / name
        if not (d / "metrics.jsonl").exists():
            return None
        history = _read_jsonl(d / "metrics.jsonl")
        meta = _read_meta(d / "meta.json")
        last_step = 0.0
        keys: set[str] = set()
        for i, e in enumerate(history):
            last_step = _step_of(e, i)
            keys.update(k for k in e if k not in _STEP_KEYS)
        group = meta.get("group") or _SEED_SUFFIX.sub("", name)
        return RunInfo(name=name, group=group,
                       last_step=last_step, n_points=len(history),
                       keys=sorted(keys))

    def list_runs(self, group: str | None = None,
                  include_checkpoints: bool = False) -> list[dict]:
        """pull.runs payload: one dict per run (name, group, last_step, n_points, keys).

        group: filter by the run's group (meta.json["group"], else the seed-stripped
        name). include_checkpoints: ALSO union runs that have a checkpoint dir but no
        metrics yet (last_step=None, n_points=0) — the legacy scan_runs behavior, so a
        run appears the moment it first checkpoints.
        """
        seen: set[str] = set()
        out: list[dict] = []
        for d in self._run_dirs():
            info = self.run_info(d.name)
            if info is None:
                continue
            if group is not None and info.group != group:
                continue
            seen.add(d.name)
            out.append(info.as_dict())
        if include_checkpoints and self.ckpt_root and self.ckpt_root.is_dir():
            for d in sorted(p for p in self.ckpt_root.iterdir() if p.is_dir()):
                if d.name in seen:
                    continue
                grp = _SEED_SUFFIX.sub("", d.name)
                if group is not None and grp != group:
                    continue
                seen.add(d.name)
                out.append({"name": d.name, "group": grp, "last_step": None,
                            "n_points": 0, "keys": []})
        return out

    # ---------------- metric (pull.metric) ----------------
    def get_metric(self, run: str, key: str) -> dict:
        """``{"steps": [...], "values": [...]}`` for one metric key.

        Pairs each value with the step recorded in the SAME log entry (env_steps
        preferred); entries lacking the key are skipped. Unknown run/key returns
        empty arrays rather than raising — the panel renders "no data".
        """
        history = _read_jsonl(self.runs_dir / run / "metrics.jsonl")
        steps: list[float] = []
        values: list[float] = []
        for i, e in enumerate(history):
            if key in e and _is_number(e[key]):
                steps.append(_step_of(e, i))
                values.append(float(e[key]))
        return {"steps": steps, "values": values}

    def metric_keys(self, run: str) -> list[str]:
        info = self.run_info(run)
        return info.keys if info else []

    # ---------------- datasets / checkpoints (pull.datasets) ----------------
    def list_datasets(self, kind: str | None = None) -> list[dict]:
        """Catalog of on-disk datasets: checkpoints + replay-buffer shards. `kind`
        filters ("checkpoint" / "buffer"); None returns all. (Minari has no local
        index yet -> nothing.)"""
        items: list[dict] = []
        if kind in (None, "checkpoint") and self.ckpt_root and self.ckpt_root.is_dir():
            items.extend(self._scan_checkpoints())
        if kind in (None, "buffer") and self.shards_root and self.shards_root.is_dir():
            items.extend(self._scan_shards())
        return items

    def _scan_shards(self) -> list[dict]:
        # Replay shards (ReplayBuffer.export_shard): any .pt under <shards_root>
        # (collect.py writes shard_w<id>.pt there). Catalogued as kind="buffer".
        out: list[dict] = []
        for sh in sorted(self.shards_root.rglob("*.pt")):
            try:
                size = sh.stat().st_size
            except OSError:
                size = 0
            out.append({"id": str(sh.relative_to(self.shards_root)), "kind": "buffer",
                        "name": sh.stem, "size": size, "path": str(sh)})
        return out

    # ---------------- artifact manifest (pull.artifacts) ----------------
    def get_config(self, run: str) -> dict:
        """The run's RESOLVED training config (results/runs/<run>/config.json,
        written by MetricsLogger at startup). {} when absent/torn — old runs
        predate the dump; the UI says so instead of inventing one."""
        path = self.results_root / "runs" / str(run) / "config.json"
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}

    def list_artifacts(self, run: str) -> list[dict]:
        """A run's artifact manifest (checkpoints + any logged W&B artifacts), read
        from results/runs/<run>/artifacts.json — written by the training side
        (mbrl.studio.artifacts) since the boundary can't call the W&B API (the seal)."""
        from .artifacts import list_artifacts as _list
        return _list(self.results_root, run)

    def _scan_checkpoints(self) -> list[dict]:
        # Real layout (train.py + CheckpointManager): <ckpt_root>/<run>/<cfg_hash>/
        # ckpt_<tag>.pt. Also tolerates a flat <ckpt_root>/<cfg_hash>/ckpt_*.pt. We
        # rglob the leaf files and derive run/hash from the path depth (the old code
        # only handled the flat layout, so pull.datasets found nothing in practice).
        out: list[dict] = []
        for ck in sorted(self.ckpt_root.rglob("ckpt_*.pt")):
            rel = ck.relative_to(self.ckpt_root).parts
            if len(rel) >= 3:           # <run>/<hash>/ckpt_*.pt
                run, cfg_hash = rel[0], rel[-2]
            elif len(rel) == 2:         # <hash>/ckpt_*.pt (flat)
                run, cfg_hash = None, rel[0]
            else:                        # ckpt_*.pt directly under root (unusual)
                run, cfg_hash = None, ""
            tag = ck.stem[len("ckpt_"):]
            try:
                size = ck.stat().st_size
            except OSError:
                size = 0
            out.append({
                "id": str(ck.relative_to(self.ckpt_root)),
                "kind": "checkpoint",
                "run": run,
                "cfg_hash": cfg_hash,
                "tag": tag,
                "step": _tag_step(tag),
                "size": size,
                "path": str(ck),
            })
        return out
