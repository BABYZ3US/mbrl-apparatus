"""Studio metric_db — stdlib sqlite3 READER for the per-run metrics.db curve store.

Writes a tiny metrics.db by hand using the SHARED SQLite contract (the same schema
the training-side writer uses) under a tmp results dir, then asserts the reader returns
the full curve ascending by env_steps and that read_metric_since filters by the cursor.
Offline, stdlib only — exercises the seal-safe reader behind pull.metric / pull.metric_since.
"""
import sqlite3
import sys
from pathlib import Path

# src/ on path for the pure reader module (no conftest; mirror the bridge test header).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from mbrl.studio import metric_db


# ---------------- the SQLite contract (writer + reader agree EXACTLY) ----------
def _make_db(results_dir: Path, run: str, rows: list[tuple]) -> Path:
    """Create <results_dir>/runs/<run>/metrics.db with the contract schema.

    `rows` is a list of (env_steps, key, value) numeric triples — one metrics row each.
    """
    run_dir = results_dir / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    db = run_dir / "metrics.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics("
            "env_steps REAL NOT NULL, key TEXT NOT NULL, value REAL NOT NULL)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_key_step "
            "ON metrics(key, env_steps)")
        conn.executemany(
            "INSERT INTO metrics(env_steps, key, value) VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    return db


def test_has_db_true_only_when_file_present(tmp_path):
    assert metric_db.has_db(tmp_path, "ghost") is False
    _make_db(tmp_path, "champ-Pendulum-v1-s0",
             [(500.0, "eval/return", -900.0)])
    assert metric_db.has_db(tmp_path, "champ-Pendulum-v1-s0") is True


def test_read_metric_db_returns_full_curve_ascending(tmp_path):
    run = "champ-Pendulum-v1-s0"
    # Insert out of order + an unrelated key interleaved; reader must sort by env_steps
    # and select ONLY the requested key.
    _make_db(tmp_path, run, [
        (1000.0, "eval/return", -200.0),
        (500.0, "eval/return", -900.0),
        (500.0, "loss", 3.0),
        (1500.0, "eval/return", -50.0),
        (1000.0, "loss", 1.0),
    ])
    out = metric_db.read_metric_db(tmp_path, run, "eval/return")
    assert out["run"] == run and out["key"] == "eval/return"
    assert out["steps"] == [500.0, 1000.0, 1500.0]
    assert out["values"] == [-900.0, -200.0, -50.0]


def test_read_metric_since_filters_by_cursor_exclusive(tmp_path):
    run = "champ-Pendulum-v1-s0"
    _make_db(tmp_path, run, [
        (500.0, "eval/return", -900.0),
        (1000.0, "eval/return", -200.0),
        (1500.0, "eval/return", -50.0),
    ])
    # since=1000 -> strictly greater than (exclusive): only the 1500 row.
    out = metric_db.read_metric_since(tmp_path, run, "eval/return", 1000.0)
    assert out["steps"] == [1500.0]
    assert out["values"] == [-50.0]
    # since below everything -> the whole curve.
    out_all = metric_db.read_metric_since(tmp_path, run, "eval/return", 0.0)
    assert out_all["steps"] == [500.0, 1000.0, 1500.0]


def test_missing_db_is_empty_not_crash(tmp_path):
    out = metric_db.read_metric_db(tmp_path, "nope", "eval/return")
    assert out == {"run": "nope", "key": "eval/return", "steps": [], "values": []}
    out_since = metric_db.read_metric_since(tmp_path, "nope", "eval/return", 0.0)
    assert out_since["steps"] == [] and out_since["values"] == []


def test_unknown_key_is_empty(tmp_path):
    run = "champ-Pendulum-v1-s0"
    _make_db(tmp_path, run, [(500.0, "eval/return", -900.0)])
    out = metric_db.read_metric_db(tmp_path, run, "no/such/key")
    assert out["steps"] == [] and out["values"] == []
