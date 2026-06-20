"""Tests for mbrl.utils.resultio — atomic + namespaced result-envelope writer."""

import json

import pytest

from mbrl.utils import resultio
from mbrl.utils.resultio import STATUS, ResultEnvelope


def test_make_autofills_id_and_ts():
    env = resultio.make("demo-task", "pass")
    assert isinstance(env, ResultEnvelope)
    assert env.task == "demo-task"
    assert env.status == "pass"
    assert env.id  # auto-filled, non-empty
    assert env.ts and env.ts.endswith("Z")  # ISO-8601 UTC


def test_make_passthrough_fields():
    env = resultio.make(
        "t",
        "partial",
        seed=7,
        params={"lr": 0.1},
        value=42,
        evidence={"k": "v"},
        blockers=["b1"],
        method="bisection",
        checked="2026-06-18T00:00:00Z",
    )
    assert env.seed == 7
    assert env.params == {"lr": 0.1}
    assert env.value == 42
    assert env.evidence == {"k": "v"}
    assert env.blockers == ["b1"]
    assert env.method == "bisection"
    assert env.checked == "2026-06-18T00:00:00Z"


def test_make_rejects_invalid_status():
    with pytest.raises(ValueError) as exc:
        resultio.make("t", "bogus-status")
    # Error message should enumerate the allowed statuses.
    for s in STATUS:
        assert s in str(exc.value)


def test_make_accepts_every_status():
    for s in STATUS:
        env = resultio.make("t", s)
        assert env.status == s


def test_round_trip_write_read(tmp_path):
    env = resultio.make("round-trip", "pass", seed=1, params={"a": 1}, value=[1, 2, 3])
    path = resultio.write(env, tmp_path)
    assert path.exists()
    loaded = resultio.read(path)
    assert loaded == env  # dataclass equality over all fields
    assert loaded.to_dict() == env.to_dict()


def test_write_accepts_plain_dict(tmp_path):
    env = resultio.make("dict-task", "skip")
    path = resultio.write(env.to_dict(), tmp_path)
    loaded = resultio.read(path)
    assert loaded == env


def test_namespacing_same_task_different_ids(tmp_path):
    # Two results, identical task, distinct ids -> two distinct files.
    a = resultio.make("same-task", "pass")
    b = resultio.make("same-task", "fail")
    assert a.id != b.id
    pa = resultio.write(a, tmp_path)
    pb = resultio.write(b, tmp_path)
    assert pa != pb
    assert pa.exists() and pb.exists()
    jsons = sorted(p.name for p in tmp_path.glob("*.json"))
    assert len(jsons) == 2
    assert pa.name in jsons and pb.name in jsons


def test_no_leftover_temp_files(tmp_path):
    env = resultio.make("clean", "pass")
    path = resultio.write(env, tmp_path)
    # Only the final .json should remain — no .tmp-* temp files.
    entries = list(tmp_path.iterdir())
    assert entries == [path]
    assert all(p.suffix == ".json" for p in entries)
    assert not any(p.name.startswith(".tmp-") for p in entries)


def test_result_filename_is_namespaced_and_safe():
    env = resultio.make("weird/task name!", "pass")
    name = resultio.result_filename(env)
    assert name.endswith(".json")
    assert env.id in name
    # No path separators or unsafe chars survive sanitization.
    assert "/" not in name and " " not in name and "!" not in name


def test_result_filename_distinct_for_distinct_ids():
    a = resultio.make("task", "pass")
    b = resultio.make("task", "pass")
    assert resultio.result_filename(a) != resultio.result_filename(b)


def test_new_id_uniqueness_many_calls():
    ids = {resultio.new_id() for _ in range(5000)}
    assert len(ids) == 5000  # all unique


def test_new_id_prefix_and_sortable():
    a = resultio.new_id("job-")
    assert a.startswith("job-")
    # Time-sortable: an id minted later sorts >= an earlier one (ms timestamp).
    first = resultio.new_id()
    second = resultio.new_id()
    assert first <= second or first.split("-")[0] <= second.split("-")[0]


def test_write_atomic_final_content_valid_json(tmp_path):
    env = resultio.make("atomic", "open", value={"nested": [1, 2]})
    path = resultio.write(env, tmp_path)
    # The on-disk file must be complete, valid JSON (never partial).
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["task"] == "atomic"
    assert data["status"] == "open"
    assert data["value"] == {"nested": [1, 2]}


def test_explicit_filename_override(tmp_path):
    env = resultio.make("t", "pass")
    path = resultio.write(env, tmp_path, filename="custom.json")
    assert path.name == "custom.json"
    assert resultio.read(path) == env
