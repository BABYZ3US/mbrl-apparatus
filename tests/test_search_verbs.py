"""W9 search verbs: submit samples + persists + launches the first batch;
tick syncs states, stops median losers (cancel), launches queued; status
enriches with last metric values. Dry-run server in a tmp results root."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import studio_bridge_server as sb


def _srv(tmp_path):
    return sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                 results_dir=tmp_path / "results" / "runs")


AXES = [{"path": "optim.model_lr", "kind": "loguniform", "low": 1e-5, "high": 1e-2}]
BASE = {"env": {"name": "Pendulum-v1"}, "seed": 0,
        "model": {"encoder": "mlp", "dynamics": "affine"}}


def _submit(srv, name="s1", n=4, parallel=2):
    return srv.dispatch(sb.make(sb.SEARCH_SUBMIT,
        {"base_spec": BASE, "axes": AXES, "n_arms": n, "parallel": parallel,
         "metric": "eval/return", "mode": "max", "seed": 7, "name": name}, 1))["data"]


def _write_metrics(tmp_path, run, pairs):
    d = tmp_path / "results" / "runs" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.jsonl").write_text("\n".join(
        json.dumps({"env_steps": s, "eval/return": v}) for s, v in pairs))


def test_submit_samples_persists_and_launches_first_batch(tmp_path):
    srv = _srv(tmp_path)
    reply = _submit(srv, n=4, parallel=2)
    assert reply["accepted"] and reply["n"] == 4
    assert len(reply["launched"]) == 2                       # the parallel cap
    status = srv.dispatch(sb.make(sb.SEARCH_STATUS, {"name": "s1"}, 2))["data"]
    assert status["found"]
    by_status = {}
    for a in status["arms"]:
        by_status[a["status"]] = by_status.get(a["status"], 0) + 1
    assert by_status == {"running": 2, "queued": 2}
    # duplicate name refuses
    again = _submit(srv, name="s1")
    assert not again["accepted"] and "exists" in again["error"]


def test_tick_stops_the_median_loser_and_launches_queued(tmp_path):
    srv = _srv(tmp_path)
    reply = _submit(srv, n=4, parallel=3)
    running = reply["launched"]
    assert len(running) == 3
    # histories: two strong arms, one clear loser
    _write_metrics(tmp_path, running[0], [(100, 1.0), (200, 2.0), (300, 3.0)])
    _write_metrics(tmp_path, running[1], [(100, 0.9), (200, 1.8), (300, 2.7)])
    _write_metrics(tmp_path, running[2], [(100, 0.1), (200, 0.2), (300, 0.3)])
    tick = srv.dispatch(sb.make(sb.SEARCH_TICK, {"name": "s1"}, 3))["data"]
    assert tick["stopped"] == [running[2]]                   # the loser
    assert len(tick["launched"]) == 1                        # the queued arm fills the slot
    assert not tick["done"]
    status = srv.dispatch(sb.make(sb.SEARCH_STATUS, {"name": "s1"}, 4))["data"]
    stopped = [a for a in status["arms"] if a["status"] == "stopped"]
    assert len(stopped) == 1 and stopped[0]["name"] == running[2]
    assert stopped[0]["last"] == 0.3                         # metric enrichment


def test_status_catalog_and_missing(tmp_path):
    srv = _srv(tmp_path)
    assert srv.dispatch(sb.make(sb.SEARCH_STATUS, {}, 5))["data"]["items"] == []
    missing = srv.dispatch(sb.make(sb.SEARCH_TICK, {"name": "ghost"}, 6))["data"]
    assert missing["found"] is False
    _submit(srv)
    assert srv.dispatch(sb.make(sb.SEARCH_STATUS, {}, 7))["data"]["items"] == ["s1"]
