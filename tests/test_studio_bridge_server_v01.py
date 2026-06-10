"""v0.1 server verbs: pull.datasets, pull.surface, submit.sweep + the submit.spec
validator gate. Stdlib only (the seal) — no torch; dry-run so nothing launches.
Kept separate from test_studio_bridge_server.py so the existing suite is untouched.
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import studio_bridge_server as sb


def test_pull_datasets_lists_checkpoints(tmp_path):
    ckpt = tmp_path / "checkpoints" / "abc123"
    ckpt.mkdir(parents=True)
    (ckpt / "ckpt_step2000.pt").write_bytes(b"x" * 8)
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                checkpoints_dir=tmp_path / "checkpoints")
    reply = srv.dispatch(sb.make(sb.PULL_DATASETS, {"kind": "checkpoint"}, 1))
    items = reply["data"]["items"]
    assert any(it["tag"] == "step2000" and it["step"] == 2000 for it in items)


def test_pull_surface_reads_artifact(tmp_path):
    surfaces = tmp_path / "results" / "runs" / "r" / "surfaces"
    surfaces.mkdir(parents=True)
    (surfaces / "surface_s2000.json").write_text(
        json.dumps({"z": [[0.0]], "curv": [[0.0]], "budget": 0.16, "step": 2000}))
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                results_dir=tmp_path / "results" / "runs")
    reply = srv.dispatch(sb.make(sb.PULL_SURFACE, {"run": "r"}, 2))
    assert reply["data"]["budget"] == 0.16 and reply["data"]["step"] == 2000


def test_pull_surface_missing_is_empty(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                results_dir=tmp_path / "results" / "runs")
    reply = srv.dispatch(sb.make(sb.PULL_SURFACE, {"run": "ghost"}, 3))
    assert reply["data"] == {}


def test_pull_artifacts_reads_manifest(tmp_path):
    from mbrl.studio.artifacts import record_artifact
    record_artifact(tmp_path / "results", "champ-Pendulum-v1-s0",
                    {"name": "model-xyz", "type": "checkpoint", "env_steps": 200000})
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                results_dir=tmp_path / "results" / "runs")
    arts = srv.dispatch(sb.make(sb.PULL_ARTIFACTS,
                                {"run": "champ-Pendulum-v1-s0"}, 1))["data"]["artifacts"]
    assert len(arts) == 1 and arts[0]["name"] == "model-xyz"


def test_submit_sweep_dryrun_expands_and_records(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    data = {"base_spec": {"experiment": {"name": "champ"},
                          "env": {"name": "Pendulum-v1"}},
            "axes": [{"path": "penalty.lambda", "values": [1e-3, 1e-2]}],
            "seeds": [0, 1]}
    reply = srv.dispatch(sb.make(sb.SUBMIT_SWEEP, data, 7))
    d = reply["data"]
    assert d["accepted"] is True and d["n"] == 4 and len(d["runs"]) == 4
    assert d["dry_run"] is True
    assert len(srv.launched) == 4              # every arm recorded a (dry) launch


def test_submit_spec_surfaces_warnings_but_launches_by_default(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    spec = {"experiment": {"name": "champ"}, "env": {"name": "Pendulum-v1"},
            "spectral": {"enabled": True},
            "penalty": {"schedule": {"kind": "step", "floor": 0.0}},
            "model": {"latent_cap_mult": 4}}
    d = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": spec}, 8))["data"]
    assert d["accepted"] is True               # default = warn-but-launch
    assert d["warnings"] and any("zero-touching" in w for w in d["warnings"])


def test_launch_monitor_verbs_via_dispatch(tmp_path):
    import time
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    assert srv.dispatch(sb.make(sb.PULL_RUN_STATUS, {"run": "ghost"}, 1))["data"]["state"] == "unknown"
    # launch a tiny job through the registry, then watch it through dispatch
    srv.launcher.launch("watch-s0", [sys.executable, "-c", "print('hi from run')"], tmp_path)
    st = {"state": "running"}
    for _ in range(500):
        st = srv.dispatch(sb.make(sb.PULL_RUN_STATUS, {"run": "watch-s0"}, 2))["data"]
        if st["state"] != "running":
            break
        time.sleep(0.02)
    assert st["state"] == "finished"
    log = srv.dispatch(sb.make(sb.PULL_LOG, {"run": "watch-s0", "since_line": 0}, 3))["data"]
    assert "hi from run" in log["lines"]
    runs = srv.dispatch(sb.make(sb.PULL_LAUNCHED, {}, 4))["data"]["runs"]
    assert any(r["run_name"] == "watch-s0" for r in runs)
    cancelled = srv.dispatch(sb.make(sb.RUN_CANCEL, {"run": "watch-s0"}, 5))["data"]
    assert cancelled["cancelled"] is False   # already finished -> no-op


def test_strict_mode_rejects_bad_spec_passes_clean(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True, strict_validation=True)
    bad = {"experiment": {"name": "champ"}, "env": {"name": "Pendulum-v1"},
           "spectral": {"enabled": True},
           "penalty": {"schedule": {"kind": "step", "floor": 0.0}},
           "model": {"latent_cap_mult": 4}}
    d = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": bad}, 9))["data"]
    assert d["accepted"] is False and d["warnings"]
    good = {"experiment": {"name": "champ"}, "env": {"name": "Pendulum-v1"},
            "spectral": {"enabled": True},
            "penalty": {"schedule": {"kind": "cuberoot", "floor": 1e-5}},
            "model": {"latent_cap_mult": 1}}
    ok = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": good}, 10))["data"]
    assert ok["accepted"] is True and ok["warnings"] == []
