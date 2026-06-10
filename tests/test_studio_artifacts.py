"""Tests for mbrl.studio.artifacts — the per-run artifact manifest (backs pull.artifacts)."""
from mbrl.studio.artifacts import list_artifacts, record_artifact


def test_record_then_list(tmp_path):
    record_artifact(tmp_path, "r", {"name": "model-abc", "type": "checkpoint", "env_steps": 1000})
    items = list_artifacts(tmp_path, "r")
    assert len(items) == 1 and items[0]["name"] == "model-abc"
    assert items[0]["env_steps"] == 1000


def test_upsert_by_name_keeps_one_entry(tmp_path):
    record_artifact(tmp_path, "r", {"name": "model-abc", "type": "checkpoint", "env_steps": 1000})
    record_artifact(tmp_path, "r", {"name": "model-abc", "type": "checkpoint", "env_steps": 5000})
    items = list_artifacts(tmp_path, "r")
    assert len(items) == 1                       # same name -> upserted, not appended
    assert items[0]["env_steps"] == 5000         # latest state wins
    record_artifact(tmp_path, "r", {"name": "replay-Pendulum-v1", "type": "replay"})
    assert {e["name"] for e in list_artifacts(tmp_path, "r")} == {"model-abc", "replay-Pendulum-v1"}


def test_missing_manifest_is_empty(tmp_path):
    assert list_artifacts(tmp_path, "ghost") == []


def test_torn_manifest_is_empty(tmp_path):
    d = tmp_path / "runs" / "r"
    d.mkdir(parents=True)
    (d / "artifacts.json").write_text("{ not valid json")
    assert list_artifacts(tmp_path, "r") == []
