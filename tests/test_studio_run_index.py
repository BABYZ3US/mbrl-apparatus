"""Tests for mbrl.studio.run_index — the pull.* query backend (v0.1 M1).

Pure stdlib + pytest tmp_path; no torch/wandb/yaml. Mirrors the on-disk format
written by mbrl.utils.metrics_logger and mbrl.utils.checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

from mbrl.studio.run_index import RunIndex


def _write_run(root: Path, name: str, group: str | None, rows: list[dict]) -> None:
    d = root / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    if group is not None:
        (d / "meta.json").write_text(json.dumps({"group": group}))
    with open(d / "metrics.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_list_runs_and_group_filter(tmp_path):
    _write_run(tmp_path, "champ-Pendulum-v1-s0", "champion",
               [{"env_steps": 100, "episode_return": -5.0},
                {"env_steps": 200, "episode_return": -3.0}])
    _write_run(tmp_path, "base-Pendulum-v1-s0", "baseline",
               [{"env_steps": 50, "episode_return": -9.0}])

    idx = RunIndex(tmp_path)
    assert {r["name"] for r in idx.list_runs()} == {
        "champ-Pendulum-v1-s0", "base-Pendulum-v1-s0"}

    champ = idx.list_runs(group="champion")
    assert [r["name"] for r in champ] == ["champ-Pendulum-v1-s0"]
    assert champ[0]["last_step"] == 200.0
    assert champ[0]["n_points"] == 2
    assert "episode_return" in champ[0]["keys"]
    assert "env_steps" not in champ[0]["keys"]  # step keys excluded from metric keys


def test_get_metric_pairs_value_with_env_steps(tmp_path):
    _write_run(tmp_path, "r", None,
               [{"env_steps": 100, "episode_return": -5.0, "penalty": 1.0},
                {"env_steps": 200, "penalty": 0.5},            # no return in this entry
                {"env_steps": 300, "episode_return": -1.0}])
    m = RunIndex(tmp_path).get_metric("r", "episode_return")
    assert m["steps"] == [100.0, 300.0]   # middle entry skipped (key absent)
    assert m["values"] == [-5.0, -1.0]


def test_get_metric_unknown_is_empty_not_error(tmp_path):
    _write_run(tmp_path, "r", None, [{"env_steps": 1, "a": 1.0}])
    idx = RunIndex(tmp_path)
    assert idx.get_metric("r", "nope") == {"steps": [], "values": []}
    assert idx.get_metric("ghost", "a") == {"steps": [], "values": []}


def test_bool_is_not_treated_as_a_metric_value(tmp_path):
    _write_run(tmp_path, "r", None, [{"env_steps": 1, "done": True, "ret": 2.0}])
    idx = RunIndex(tmp_path)
    assert idx.get_metric("r", "done") == {"steps": [], "values": []}
    assert idx.get_metric("r", "ret")["values"] == [2.0]


def test_torn_jsonl_line_is_skipped(tmp_path):
    d = tmp_path / "runs" / "crashed"
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text(
        json.dumps({"env_steps": 1, "x": 1.0}) + "\n{ this is a torn write")
    idx = RunIndex(tmp_path)
    assert idx.run_info("crashed").n_points == 1
    assert idx.get_metric("crashed", "x")["values"] == [1.0]


def test_fallback_step_is_ordinal_when_no_step_key(tmp_path):
    _write_run(tmp_path, "r", None, [{"loss": 9.0}, {"loss": 8.0}])
    m = RunIndex(tmp_path).get_metric("r", "loss")
    assert m["steps"] == [0.0, 1.0]


def test_list_datasets_scans_checkpoints(tmp_path):
    (tmp_path / "results" / "runs").mkdir(parents=True)
    ckpt = tmp_path / "checkpoints" / "abc123def456"
    ckpt.mkdir(parents=True)
    (ckpt / "ckpt_step2000.pt").write_bytes(b"x" * 10)
    (ckpt / "ckpt_best.pt").write_bytes(b"y" * 5)

    idx = RunIndex(tmp_path / "results", ckpt_root=tmp_path / "checkpoints")
    by_tag = {d["tag"]: d for d in idx.list_datasets(kind="checkpoint")}
    assert by_tag["step2000"]["step"] == 2000
    assert by_tag["step2000"]["cfg_hash"] == "abc123def456"
    assert by_tag["best"]["step"] is None
    assert idx.list_datasets(kind="minari") == []  # unknown/absent kind -> empty


def test_no_results_dir_is_empty_not_error(tmp_path):
    idx = RunIndex(tmp_path / "does_not_exist")
    assert idx.list_runs() == []
    assert idx.run_info("whatever") is None


def test_group_prefers_meta_else_strips_seed_suffix(tmp_path):
    _write_run(tmp_path, "champ-Pendulum-v1-s0", "champion", [{"env_steps": 1, "r": 1.0}])
    _write_run(tmp_path, "base-Pendulum-v1-s2", None, [{"env_steps": 1, "r": 1.0}])  # no meta
    by_name = {r["name"]: r for r in RunIndex(tmp_path).list_runs()}
    assert by_name["champ-Pendulum-v1-s0"]["group"] == "champion"        # meta wins
    assert by_name["base-Pendulum-v1-s2"]["group"] == "base-Pendulum-v1"  # regex fallback


def test_list_runs_unions_checkpoint_only_runs(tmp_path):
    results = tmp_path / "results"
    _write_run(results, "champ-Pendulum-v1-s0", "champion", [{"env_steps": 9, "r": 1.0}])
    ck = tmp_path / "checkpoints" / "champ-Pendulum-v1-s1" / "deadbeef"  # checkpoint-only run
    ck.mkdir(parents=True)
    (ck / "ckpt_step100.pt").write_bytes(b"x")
    runs = RunIndex(results, ckpt_root=tmp_path / "checkpoints").list_runs(include_checkpoints=True)
    by_name = {r["name"]: r for r in runs}
    assert by_name["champ-Pendulum-v1-s0"]["last_step"] == 9.0
    assert by_name["champ-Pendulum-v1-s1"]["last_step"] is None       # not trained yet
    assert by_name["champ-Pendulum-v1-s1"]["n_points"] == 0


def test_scan_checkpoints_real_nested_layout(tmp_path):
    # the train.py layout the old code missed: checkpoints/<run>/<cfg_hash>/ckpt_<tag>.pt
    ck = tmp_path / "checkpoints" / "champ-Pendulum-v1-s0" / "abc123"
    ck.mkdir(parents=True)
    (ck / "ckpt_step5000.pt").write_bytes(b"x" * 4)
    ds = RunIndex(tmp_path / "results", ckpt_root=tmp_path / "checkpoints").list_datasets(
        kind="checkpoint")
    assert len(ds) == 1
    assert ds[0]["run"] == "champ-Pendulum-v1-s0"
    assert ds[0]["cfg_hash"] == "abc123" and ds[0]["step"] == 5000


def test_list_datasets_includes_buffer_shards(tmp_path):
    results = tmp_path / "results"
    (results / "shards").mkdir(parents=True)
    (results / "shards" / "shard_w0.pt").write_bytes(b"x" * 6)
    (results / "shards" / "shard_w1.pt").write_bytes(b"y" * 3)
    buffers = RunIndex(results).list_datasets(kind="buffer")
    assert {b["name"] for b in buffers} == {"shard_w0", "shard_w1"}
    assert all(b["kind"] == "buffer" for b in buffers)


def test_list_artifacts_reads_manifest(tmp_path):
    from mbrl.studio.artifacts import record_artifact
    record_artifact(tmp_path, "champ-Pendulum-v1-s0",
                    {"name": "model-xyz", "type": "checkpoint", "env_steps": 9})
    idx = RunIndex(tmp_path)
    arts = idx.list_artifacts("champ-Pendulum-v1-s0")
    assert len(arts) == 1 and arts[0]["name"] == "model-xyz"
    assert idx.list_artifacts("ghost") == []
