"""SQLite metric store — a buffered, WAL-backed mirror of the JSONL metric log.

The trainer appends metrics here in batches (one SQLite transaction per buffer
flush); the studio bridge reads them back with `SELECT ... WHERE key=? ORDER BY
env_steps` (lock-free under WAL while the trainer keeps writing). This is the
database seam: SQLite now (stdlib, zero-dep, file-local), swappable for a remote
backend behind the same append/read interface later.

Stdlib only (sqlite3) — safe inside the sealed training runtime. Schema is the wire
contract shared with mbrl.studio.metric_db on the reader side:

    metrics(env_steps REAL, key TEXT, value REAL)
    index   idx_metrics_key_step (key, env_steps)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def _as_float(v) -> float | None:
    """A metric value -> float, or None if it is not numeric (skip strings, dicts,
    wandb.Video, etc.). bool is treated as non-numeric on purpose."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "item"):  # torch/numpy scalar
        try:
            return float(v.item())
        except Exception:
            return None
    return None


class MetricStore:
    """Buffered writer over one run's metrics.db. append() collects rows; flush()
    commits them in a single transaction; close() flushes and closes."""

    _STEP_KEYS = ("env_steps", "step")

    def __init__(self, db_path: str | Path, buffer_size: int = 200):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = max(1, int(buffer_size))
        self._buf: list[tuple[float, str, float]] = []
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)  # we manage txns
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics("
            "env_steps REAL NOT NULL, key TEXT NOT NULL, value REAL NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_key_step ON metrics(key, env_steps)"
        )

    def append(self, env_steps, metrics: dict) -> None:
        """Buffer one row per numeric (key, value) in `metrics`, stamped with env_steps.
        The step key itself is never stored as a metric; non-numeric values are skipped."""
        if env_steps is None:
            return
        es = _as_float(env_steps)
        if es is None:
            return
        for k, v in metrics.items():
            if k in self._STEP_KEYS:
                continue
            fv = _as_float(v)
            if fv is None:
                continue
            self._buf.append((es, k, fv))
        if len(self._buf) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Commit the buffered rows in one transaction."""
        if not self._buf:
            return
        self._conn.execute("BEGIN")
        self._conn.executemany(
            "INSERT INTO metrics(env_steps, key, value) VALUES (?, ?, ?)", self._buf
        )
        self._conn.execute("COMMIT")
        self._buf.clear()

    def read(self, key: str) -> tuple[list[float], list[float]]:
        """(steps, values) for `key`, ascending by env_steps. Reads committed rows."""
        cur = self._conn.execute(
            "SELECT env_steps, value FROM metrics WHERE key=? ORDER BY env_steps", (key,)
        )
        rows = cur.fetchall()
        return [float(r[0]) for r in rows], [float(r[1]) for r in rows]

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._conn.close()
