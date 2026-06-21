"""Studio bridge SERVER — author + launch over one socket, byte-exact framing.

Covers the runner that authors AND launches training runs (submit.spec) plus the
viz pull path, and the pure spec->Hydra mapping. STDLIB + the spec_to_config /
server modules ONLY — never imports torch or scripts/train.py (the seal:
docs/remote_execution.md §1). submit.spec is exercised in DRY-RUN so no training
is spawned; we assert the exact `python scripts/train.py ...` command instead.

The framing test is load-bearing: the 4-byte length prefix MUST be little-endian
to match Godot's PackedByteArray.encode_u32 (godot_studio/addons/mbrl_bridge/
protocol.gd). A big-endian prefix would desync the whole channel.
"""
import json
import socket
import sqlite3
import struct
import sys
import time
from pathlib import Path

import yaml

# scripts/ on path for the server module; src/ for the pure mapping module.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import studio_bridge_server as sb
from mbrl.studio import spec_to_config as s2c


# ---------------- framing: byte-exact with protocol.gd ----------------
def test_frame_is_little_endian_and_roundtrips():
    msg = {"type": "submit.spec", "id": 7, "data": {"model_spec": {"a": 1}}}
    blob = sb.frame(msg)
    (n_le,) = struct.unpack("<I", blob[:4])
    (n_be,) = struct.unpack(">I", blob[:4])
    assert n_le == len(blob) - 4            # little-endian length (encode_u32)
    assert n_be != n_le or n_le < 256       # for a small payload the two differ
    assert json.loads(blob[4:].decode("utf-8")) == msg


def test_frame_decoder_extracts_whole_frames_from_split_chunks():
    """The recv loop must reassemble frames across arbitrary chunk boundaries."""
    a = sb.frame(sb.make("hello", {"version": 1}, 1))
    b = sb.frame(sb.make("pull.runs", {}, 2))
    stream = a + b
    dec = sb.FrameDecoder()
    got = []
    # feed one byte at a time — the worst-case fragmentation
    for i in range(len(stream)):
        got.extend(dec.feed(stream[i:i + 1]))
    assert [m["id"] for m in got] == [1, 2]
    assert got[0]["data"]["version"] == 1


# ---------------- spec -> Hydra overrides (pure) ----------------
def test_spec_to_overrides_nested_dict_list_bool():
    spec = {
        "model": {"dynamics": "gaussian", "latent_cap_mult": 1},
        "penalty": {"clamp_trace": True, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "floor": 1.0e-5}},
        "spectral": {"enabled": True,
                     "poly": {"degrees": [1, 3], "coefs": [0.1, 10.0]}},
    }
    ov = s2c.spec_to_overrides(spec)
    assert "model.dynamics=gaussian" in ov
    assert "model.latent_cap_mult=1" in ov
    assert "penalty.clamp_trace=true" in ov          # bool lower-cased
    assert "penalty.penalize_dynamics=false" in ov
    assert "penalty.schedule.kind=cuberoot" in ov
    assert "spectral.enabled=true" in ov
    assert "spectral.poly.degrees=[1,3]" in ov        # list -> bracketed, no spaces
    assert "spectral.poly.coefs=[0.1,10.0]" in ov


def test_spec_to_overrides_experiment_and_env_group_forms():
    spec = {
        "experiment": {"name": "champion"},
        "env": {"name": "Walker2d-v5"},
        "seed": 3,
        "model": {"dynamics": "gaussian"},
    }
    ov = s2c.spec_to_overrides(spec)
    # group selectors come first, in the +experiment / env= forms train.py wants
    assert ov[0] == "+experiment=champion"
    assert ov[1] == "env=walker2d"                    # display name -> group file
    assert "experiment.name=champion" not in ov       # NOT also a field override
    assert "seed=3" in ov
    assert "model.dynamics=gaussian" in ov


def test_spec_to_overrides_unknown_env_name_slug_fallback():
    assert s2c.env_group("HalfCheetah-v5") == "halfcheetah"
    assert s2c.env_group("SomeNewEnv-v9") == "somenewenv"


def test_spec_to_overrides_rejects_non_dict():
    import pytest
    with pytest.raises(TypeError):
        s2c.spec_to_overrides(["not", "a", "dict"])


# ---------------- write_experiment_yaml: parseable @package _global_ ----------
def test_write_experiment_yaml_is_parseable_global_package(tmp_path):
    spec = {"model": {"dynamics": "gaussian"},
            "spectral": {"enabled": True, "poly": {"degrees": [1, 3]}}}
    path = s2c.write_experiment_yaml(spec, tmp_path, "champ")
    assert path == tmp_path / "experiment" / "champ.yaml"
    text = path.read_text()
    assert text.splitlines()[0] == "# @package _global_"   # Hydra root-splice directive
    loaded = yaml.safe_load(text)                            # body must parse
    assert loaded["model"]["dynamics"] == "gaussian"
    assert loaded["spectral"]["poly"]["degrees"] == [1, 3]
    assert loaded["experiment"]["name"] == "champ"           # name stamped in


def test_run_name_matches_train_convention():
    spec = {"experiment": {"name": "champion"}, "env": {"name": "HalfCheetah-v5"}}
    assert s2c.run_name_for_spec(spec, seed=2) == "champion-HalfCheetah-v5-s2"
    # defaults mirror configs/base.yaml when the spec omits them
    assert s2c.run_name_for_spec({}) == "dev-Pendulum-v1-s0"


# ---------------- submit.spec in dry-run: accepted + the train command ----------
def test_submit_spec_dryrun_returns_accepted_and_command(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    spec = {
        "experiment": {"name": "champion"},
        "env": {"name": "Walker2d-v5"},
        "seed": 3,
        "model": {"dynamics": "gaussian", "latent_cap_mult": 1},
        "spectral": {"enabled": True, "poly": {"degrees": [1, 3]}},
    }
    reply = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": spec}, 11))
    assert reply["type"] == sb.SUBMIT_SPEC and reply["id"] == 11
    d = reply["data"]
    assert d["accepted"] is True
    assert d["dry_run"] is True
    assert d["run_name"] == "champion-Walker2d-v5-s3"
    cmd = d["command"]
    # `python scripts/train.py ...` — sys.executable then the Hydra entry point
    assert cmd[0] == sys.executable
    assert cmd[1] == "scripts/train.py"
    assert "+experiment=champion" in cmd
    assert "env=walker2d" in cmd
    assert "model.dynamics=gaussian" in cmd
    assert "spectral.enabled=true" in cmd
    assert "spectral.poly.degrees=[1,3]" in cmd
    assert "seed=3" in cmd
    # the run is recorded, not launched (no subprocess in dry-run)
    assert srv.launched and srv.launched[0]["dry_run"] is True
    # the experiment yaml was actually written under the repo's authoring dir
    assert Path(d["experiment_yaml"]).exists()


def test_submit_spec_via_env_var_dryrun(monkeypatch, tmp_path):
    monkeypatch.setenv("MBRL_STUDIO_DRYRUN", "1")
    srv = sb.StudioBridgeServer(repo_root=tmp_path)   # dry_run resolved from env
    assert srv.dry_run is True
    reply = srv.dispatch(sb.make(sb.SUBMIT_SPEC,
                                 {"model_spec": {"model": {"dynamics": "affine"}}}, 1))
    assert reply["data"]["accepted"] is True
    assert reply["data"]["dry_run"] is True


def test_submit_spec_bad_model_spec_is_rejected_not_crash(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    reply = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": "not-an-object"}, 4))
    assert reply["data"]["accepted"] is False
    assert "error" in reply["data"]


# ---------------- pull.runs / pull.metric: defensive on empty dirs ----------
def test_pull_runs_empty_dirs_is_empty_not_crash(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                results_dir=tmp_path / "nope" / "runs",
                                checkpoints_dir=tmp_path / "nope" / "ckpt")
    reply = srv.dispatch(sb.make(sb.PULL_RUNS, {}, 1))
    assert reply["data"]["runs"] == []


def test_pull_metric_missing_run_is_empty_not_crash(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                results_dir=tmp_path / "runs")
    reply = srv.dispatch(sb.make(sb.PULL_METRIC,
                                 {"run": "ghost-s0", "key": "eval/return"}, 2))
    assert reply["data"]["steps"] == [] and reply["data"]["values"] == []


# ---------------- pull.metric_since + prefer-DB / JSONL-fallback ----------
def _write_metrics_db(results_dir: Path, run: str, rows: list[tuple]) -> Path:
    """metrics.db under <results_dir>/runs/<run>/ with the shared SQLite contract.

    `rows`: list of (env_steps, key, value) triples.
    """
    run_dir = results_dir / run
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


def test_pull_metric_since_db_returns_filtered_curve(tmp_path):
    runs = tmp_path / "results" / "runs"
    run = "champion-Pendulum-v1-s0"
    _write_metrics_db(runs, run, [
        (500.0, "eval/return", -900.0),
        (1000.0, "eval/return", -200.0),
        (1500.0, "eval/return", -50.0),
    ])
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True, results_dir=runs)
    reply = srv.dispatch(sb.make(sb.PULL_METRIC_SINCE,
                                 {"run": run, "key": "eval/return", "since": 1000.0}, 21))
    assert reply["type"] == sb.PULL_METRIC_SINCE and reply["id"] == 21
    d = reply["data"]
    # env_steps > since (exclusive): only the 1500 row past the cursor.
    assert d["steps"] == [1500.0]
    assert d["values"] == [-50.0]
    assert d["run"] == run and d["key"] == "eval/return"


def test_pull_metric_prefers_db_over_jsonl(tmp_path):
    runs = tmp_path / "results" / "runs"
    run = "champion-Pendulum-v1-s0"
    # DB carries the real curve; a stale/empty JSONL must be ignored when the db exists.
    _write_metrics_db(runs, run, [
        (500.0, "eval/return", -900.0),
        (1000.0, "eval/return", -200.0),
    ])
    (runs / run / "metrics.jsonl").write_text(
        json.dumps({"env_steps": 9999, "eval/return": 123.0}) + "\n")
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True, results_dir=runs)
    reply = srv.dispatch(sb.make(sb.PULL_METRIC,
                                 {"run": run, "key": "eval/return"}, 22))
    d = reply["data"]
    assert d["steps"] == [500.0, 1000.0]        # from the db, not the jsonl
    assert d["values"] == [-900.0, -200.0]


def test_pull_metric_falls_back_to_jsonl_when_no_db(tmp_path):
    runs = tmp_path / "results" / "runs"
    run = "jsonl-only-Pendulum-v1-s0"
    (runs / run).mkdir(parents=True)
    (runs / run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"env_steps": 500, "eval/return": -900.0},
        {"env_steps": 1000, "eval/return": -200.0}]) + "\n")
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True, results_dir=runs)
    # pull.metric still works for a JSONL-only run (unchanged fallback path).
    reply = srv.dispatch(sb.make(sb.PULL_METRIC,
                                 {"run": run, "key": "eval/return"}, 23))
    assert reply["data"]["steps"] == [500.0, 1000.0]
    assert reply["data"]["values"] == [-900.0, -200.0]
    # pull.metric_since over a JSONL-only run filters by the cursor too.
    reply2 = srv.dispatch(sb.make(sb.PULL_METRIC_SINCE,
                                  {"run": run, "key": "eval/return", "since": 500.0}, 24))
    assert reply2["data"]["steps"] == [1000.0]
    assert reply2["data"]["values"] == [-200.0]


def test_pull_runs_unions_results_and_checkpoints(tmp_path):
    # A8: the union behavior lives in dispatch(pull.runs) -> RunIndex
    # (include_checkpoints=True); the legacy scan_runs is deleted.
    runs = tmp_path / "results" / "runs"
    ckpt = tmp_path / "checkpoints"
    (runs / "champion-Pendulum-v1-s0").mkdir(parents=True)
    (runs / "champion-Pendulum-v1-s0" / "metrics.jsonl").write_text(
        json.dumps({"env_steps": 2000, "eval/return": -120.0}) + "\n")
    # a checkpoint-only run (no metrics yet) still shows up, last_step=None
    (ckpt / "champion-Pendulum-v1-s1" / "abcd").mkdir(parents=True)

    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    out = {r["name"]: r for r in srv.dispatch(sb.make(sb.PULL_RUNS, {}, 1))["data"]["runs"]}
    assert float(out["champion-Pendulum-v1-s0"]["last_step"]) == 2000.0
    assert out["champion-Pendulum-v1-s0"]["group"] == "champion-Pendulum-v1"
    assert out["champion-Pendulum-v1-s1"]["last_step"] is None
    # group filter narrows to the arm
    filt = srv.dispatch(sb.make(sb.PULL_RUNS, {"group": "champion-Pendulum-v1"}, 2))
    assert len(filt["data"]["runs"]) == 2


# ---------------- end-to-end over a REAL loopback socket ----------------
def test_end_to_end_hello_and_pull_over_real_socket(tmp_path):
    """Start the server on an ephemeral port in a thread; a Godot-like client
    sends hello and pull.metric and we assert the framed replies."""
    runs = tmp_path / "results" / "runs"
    (runs / "champion-Pendulum-v1-s0").mkdir(parents=True)
    (runs / "champion-Pendulum-v1-s0" / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            {"env_steps": 500, "eval/return": -900.0},
            {"env_steps": 1000, "eval/return": -200.0}]) + "\n")

    srv = sb.StudioBridgeServer(host="127.0.0.1", port=0, repo_root=tmp_path,
                                dry_run=True, results_dir=runs).start()
    try:
        cli = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        cli.settimeout(5)

        # hello -> {"version": 1}
        cli.sendall(sb.frame(sb.make(sb.HELLO, {"version": 1, "role": "studio"}, 1)))
        reply = sb.read_frame(cli)
        assert reply["id"] == 1 and reply["data"]["version"] == sb.VERSION

        # pull.metric over the real wire
        cli.sendall(sb.frame(sb.make(sb.PULL_METRIC,
                    {"run": "champion-Pendulum-v1-s0", "key": "eval/return"}, 2)))
        reply = sb.read_frame(cli)
        assert reply["id"] == 2
        assert reply["data"]["values"] == [-900.0, -200.0]
        assert reply["data"]["steps"] == [500.0, 1000.0]

        cli.close()
    finally:
        srv.stop()


def test_end_to_end_submit_spec_dryrun_over_real_socket(tmp_path):
    """submit.spec across the socket in dry-run returns the launch command —
    the full author+launch path minus the actual subprocess."""
    srv = sb.StudioBridgeServer(host="127.0.0.1", port=0, repo_root=tmp_path,
                                dry_run=True).start()
    try:
        cli = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        cli.settimeout(5)
        spec = {"experiment": {"name": "champion"}, "env": {"name": "HalfCheetah-v5"},
                "seed": 1, "model": {"dynamics": "gaussian"}}
        cli.sendall(sb.frame(sb.make(sb.SUBMIT_SPEC, {"model_spec": spec}, 9)))
        reply = sb.read_frame(cli)
        assert reply["id"] == 9
        d = reply["data"]
        assert d["accepted"] is True and d["run_name"] == "champion-HalfCheetah-v5-s1"
        assert "+experiment=champion" in d["command"]
        assert "env=halfcheetah" in d["command"]
        cli.close()
    finally:
        srv.stop()


def test_pull_surface_step_minus_one_means_latest(tmp_path):
    """A7 regression: Godot's step=-1 (bridge.gd default) must resolve to the
    LATEST surface, not exact-match nothing and return {}."""
    sdir = tmp_path / "results" / "runs" / "r-s0" / "surfaces"
    sdir.mkdir(parents=True)
    (sdir / "surface_s100.json").write_text(json.dumps({"step": 100, "z": [[1]]}))
    (sdir / "surface_s200.json").write_text(json.dumps({"step": 200, "z": [[2]]}))
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    reply = srv.dispatch(sb.make(sb.PULL_SURFACE, {"run": "r-s0", "step": -1}, 3))
    assert reply["data"].get("step") == 200
    exact = srv.dispatch(sb.make(sb.PULL_SURFACE, {"run": "r-s0", "step": 100}, 4))
    assert exact["data"].get("step") == 100
