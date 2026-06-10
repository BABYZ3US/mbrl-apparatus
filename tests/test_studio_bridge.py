"""Studio bridge server — framing byte-exactness + viz reads over the REAL dispatch.

The framing test is load-bearing: the 4-byte length prefix MUST be little-endian
to match Godot's PackedByteArray.encode_u32 (godot_studio/addons/mbrl_bridge/
protocol.gd). A big-endian prefix would desync the whole channel. Stdlib only.

2026-06-09 (A8): migrated off the legacy pure handle() onto
StudioBridgeServer.dispatch()/start() — one server, one read path (RunIndex).
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import studio_bridge_server as sb


def _server(tmp_path) -> "sb.StudioBridgeServer":
    return sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)


def _write_run(repo_root: Path, name: str, rows: list[dict]) -> None:
    d = repo_root / "results" / "runs" / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_frame_is_little_endian_and_roundtrips():
    msg = {"type": "pull.runs", "id": 7, "data": {"group": "champion"}}
    blob = sb.frame(msg)
    (n_le,) = struct.unpack("<I", blob[:4])
    (n_be,) = struct.unpack(">I", blob[:4])
    assert n_le == len(blob) - 4
    assert n_be != n_le or n_le < 256  # for a small payload the two differ
    assert json.loads(blob[4:].decode("utf-8")) == msg


def test_make_echoes_type_and_id():
    m = sb.make("hello", {"version": 1}, 3)
    assert m == {"type": "hello", "id": 3, "data": {"version": 1}}


def test_pull_runs_groups_and_last_step(tmp_path):
    _write_run(tmp_path, "champion-HalfCheetah-v5-s0",
               [{"env_steps": 1000, "eval/return": -300.0},
                {"env_steps": 2000, "eval/return": -120.0}])
    _write_run(tmp_path, "champion-HalfCheetah-v5-s1",
               [{"env_steps": 1000, "eval/return": -310.0}])
    reply = _server(tmp_path).dispatch(sb.make(sb.PULL_RUNS, {}, 1))
    runs = {r["name"]: r for r in reply["data"]["runs"]}
    assert reply["id"] == 1
    # group = seed-stripped name when no meta.json (RunIndex fallback rule)
    assert runs["champion-HalfCheetah-v5-s0"]["group"] == "champion-HalfCheetah-v5"
    assert float(runs["champion-HalfCheetah-v5-s0"]["last_step"]) == 2000.0
    assert float(runs["champion-HalfCheetah-v5-s1"]["last_step"]) == 1000.0


def test_pull_metric_returns_aligned_arrays(tmp_path):
    _write_run(tmp_path, "champion-HalfCheetah-v5-s0",
               [{"env_steps": 1000, "loss/total": 0.08},
                {"env_steps": 2000},                       # row without the key
                {"env_steps": 3000, "loss/total": 0.04, "eval/return": -90.0}])
    reply = _server(tmp_path).dispatch(sb.make(sb.PULL_METRIC,
                                               {"run": "champion-HalfCheetah-v5-s0",
                                                "key": "loss/total"}, 5))
    d = reply["data"]
    assert reply["id"] == 5
    assert d["steps"] == [1000.0, 3000.0]      # the keyless row is skipped
    assert d["values"] == [0.08, 0.04]


def test_pull_metric_unknown_key_is_empty_not_crash(tmp_path):
    _write_run(tmp_path, "r-s0", [{"env_steps": 1000, "loss/total": 0.08}])
    reply = _server(tmp_path).dispatch(sb.make(sb.PULL_METRIC,
                                               {"run": "r-s0", "key": "does/not/exist"}, 0))
    assert reply["data"]["steps"] == [] and reply["data"]["values"] == []


def test_godot_side_verbs_return_not_served(tmp_path):
    srv = _server(tmp_path)
    for t in (sb.ENV_STEP, sb.ENV_RESET, sb.ENV_SPEC, sb.INFER_LOAD, sb.INFER_RUN):
        reply = srv.dispatch(sb.make(t, {}, 9))
        assert reply["type"] == sb.ERROR
        assert reply["data"]["code"] == "not_served"
        assert reply["id"] == 9


def test_hello_replies_version(tmp_path):
    reply = _server(tmp_path).dispatch(sb.make(sb.HELLO, {"version": 1, "role": "studio"}, 1))
    assert reply["data"]["version"] == sb.VERSION


def test_end_to_end_over_real_socket(tmp_path):
    """Round-trip frames against the REAL server (start/accept loop), as a Godot
    client would — proves TCP framing + dispatch, not just the pure handler."""
    import socket
    _write_run(tmp_path, "champion-Pendulum-v1-s0",
               [{"env_steps": 500, "eval/return": -900.0},
                {"env_steps": 1000, "eval/return": -200.0}])
    srv = sb.StudioBridgeServer(host="127.0.0.1", port=0, repo_root=tmp_path, dry_run=True)
    srv.start()
    try:
        cli = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        cli.sendall(sb.frame(sb.make(sb.HELLO, {"version": 1, "role": "studio"}, 1)))
        assert sb.read_frame(cli)["data"]["version"] == sb.VERSION
        cli.sendall(sb.frame(sb.make(sb.PULL_METRIC,
                    {"run": "champion-Pendulum-v1-s0", "key": "eval/return"}, 2)))
        reply = sb.read_frame(cli)
        assert reply["id"] == 2
        assert reply["data"]["values"] == [-900.0, -200.0]
        cli.close()
    finally:
        srv.stop()
