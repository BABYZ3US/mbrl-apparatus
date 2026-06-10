"""metric_db — stdlib sqlite3 READER for the per-run metrics.db curve store.

The training side writes a small SQLite database alongside the JSONL mirror so the
Studio can pull metric curves cheaply and, crucially, *incrementally* (pull only the
rows past a cursor). This module is the read half: it opens that db read-only and
returns the same ``{run, key, steps, values}`` payload the JSONL path produces, so the
one-boundary server (scripts/studio_bridge_server.py) can prefer it transparently.

Pure stdlib (sqlite3 only). NOTHING from torch / mbrl.training — safe inside the studio
seal (docs/remote_execution.md §1), same as run_index / surface_index.

SQLite contract (writer and reader agree on this EXACTLY):

    <results_dir>/runs/<run_name>/metrics.db
    PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL.
    CREATE TABLE IF NOT EXISTS metrics(
        env_steps REAL NOT NULL, key TEXT NOT NULL, value REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_metrics_key_step ON metrics(key, env_steps);

One row per (env_steps, key, value) numeric pair. The step key in a metrics dict is
``env_steps`` (fallback ``step``); the step key itself is never stored as a metric row,
and non-numeric values are skipped. Tolerant of a missing db (-> empty / False).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def _db_path(results_dir, run: str) -> Path:
    """``<results_dir>/runs/<run>/metrics.db`` — the one contract path."""
    return Path(results_dir) / "runs" / str(run) / "metrics.db"


def has_db(results_dir, run: str) -> bool:
    """True iff the run's metrics.db file exists (cheap stat, no open)."""
    return _db_path(results_dir, run).is_file()


def _connect_ro(path: Path) -> sqlite3.Connection | None:
    """Open the db read-only (uri ?mode=ro). None if the file is absent.

    mode=ro never creates the file and never takes a write lock, so a live writer
    (WAL) is undisturbed. Returns None rather than raising on a missing/locked db.
    """
    if not path.is_file():
        return None
    try:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _query(results_dir, run: str, key: str, since: float | None) -> dict:
    """Shared SELECT -> {run, key, steps, values}, ascending by env_steps.

    `since` None => the whole curve; otherwise only rows with env_steps > since.
    Missing db / table / key all return empty arrays — a normal "no data" query.
    """
    steps: list[float] = []
    values: list[float] = []
    conn = _connect_ro(_db_path(results_dir, run))
    if conn is not None:
        try:
            if since is None:
                cur = conn.execute(
                    "SELECT env_steps, value FROM metrics "
                    "WHERE key=? ORDER BY env_steps",
                    (key,),
                )
            else:
                cur = conn.execute(
                    "SELECT env_steps, value FROM metrics "
                    "WHERE key=? AND env_steps > ? ORDER BY env_steps",
                    (key, float(since)),
                )
            for env_steps, value in cur.fetchall():
                steps.append(float(env_steps))
                values.append(float(value))
        except sqlite3.Error:
            # No metrics table yet (fresh/torn db), or any read hiccup — treat as
            # "no data" rather than crashing the boundary server's dispatch.
            steps, values = [], []
        finally:
            conn.close()
    return {"run": str(run), "key": str(key), "steps": steps, "values": values}


def read_metric_db(results_dir, run: str, key: str) -> dict:
    """Full curve for `key`: {run, key, steps, values} ascending by env_steps."""
    return _query(results_dir, run, key, since=None)


def read_metric_since(results_dir, run: str, key: str, since: float) -> dict:
    """Incremental curve: only rows with env_steps > `since`, ascending.

    Same shape as read_metric_db. The Studio passes the last env_steps it has seen
    as the cursor to pull just the new tail.
    """
    return _query(results_dir, run, key, since=float(since))
