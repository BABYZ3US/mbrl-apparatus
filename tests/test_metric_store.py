"""MetricStore — the buffered, WAL-backed SQLite metric writer.

Round-trips a metric curve, batches via the buffer, skips non-numeric + the step key,
appends across re-opens, and lets a second connection read under WAL while the writer
is open. Offline, stdlib only. The schema is the contract shared with metric_db (reader).
"""
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from mbrl.utils.metric_store import MetricStore


def test_append_flush_read_roundtrip(tmp_path):
    s = MetricStore(tmp_path / "metrics.db")
    s.append(1000, {"loss/total": 0.5, "env_steps": 1000})
    s.append(2000, {"loss/total": 0.25, "env_steps": 2000})
    s.flush()
    steps, values = s.read("loss/total")
    assert steps == [1000.0, 2000.0]
    assert values == [0.5, 0.25]
    s.close()


def test_buffer_batches_then_flushes(tmp_path):
    s = MetricStore(tmp_path / "metrics.db", buffer_size=4)
    # 3 appends of one numeric key each -> 3 buffered rows, below the batch size of 4
    for i in range(3):
        s.append(i, {"m": float(i)})
    steps, _ = s.read("m")
    assert steps == []  # nothing committed yet
    s.append(3, {"m": 3.0})  # 4th row trips the buffer -> auto-flush
    steps, values = s.read("m")
    assert steps == [0.0, 1.0, 2.0, 3.0]
    assert values == [0.0, 1.0, 2.0, 3.0]
    s.close()


def test_two_keys_are_independent(tmp_path):
    s = MetricStore(tmp_path / "metrics.db")
    s.append(10, {"a": 1.0, "b": 9.0})
    s.append(20, {"a": 2.0})
    s.flush()
    assert s.read("a") == ([10.0, 20.0], [1.0, 2.0])
    assert s.read("b") == ([10.0], [9.0])
    s.close()


def test_non_numeric_and_step_keys_skipped(tmp_path):
    s = MetricStore(tmp_path / "metrics.db")
    s.append(5, {"good": 1.5, "label": "x", "nested": {"k": 1},
                 "flag": True, "env_steps": 5, "step": 5})
    s.flush()
    # only the numeric, non-step key landed
    assert s.read("good") == ([5.0], [1.5])
    assert s.read("label") == ([], [])
    assert s.read("env_steps") == ([], [])
    assert s.read("flag") == ([], [])  # bool is treated as non-numeric
    s.close()


def test_reopen_appends_not_clobbers(tmp_path):
    db = tmp_path / "metrics.db"
    s1 = MetricStore(db)
    s1.append(1, {"m": 1.0})
    s1.close()
    s2 = MetricStore(db)
    s2.append(2, {"m": 2.0})
    s2.close()
    s3 = MetricStore(db)
    assert s3.read("m") == ([1.0, 2.0], [1.0, 2.0])
    s3.close()


def test_wal_allows_concurrent_reader(tmp_path):
    db = tmp_path / "metrics.db"
    s = MetricStore(db)
    s.append(100, {"m": 0.7})
    s.flush()  # committed, but the writer connection stays OPEN
    # a separate read-only connection sees the committed row under WAL
    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = ro.execute("SELECT env_steps, value FROM metrics WHERE key=?", ("m",)).fetchall()
    ro.close()
    assert rows == [(100.0, 0.7)]
    s.close()
