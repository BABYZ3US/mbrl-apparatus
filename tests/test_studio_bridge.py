"""Studio bridge server — framing byte-exactness + viz handlers.

The framing test is load-bearing: the 4-byte length prefix MUST be little-endian
to match Godot's PackedByteArray.encode_u32 (godot_studio/addons/mbrl_bridge/
protocol.gd). A big-endian prefix would desync the whole channel. Stdlib only.
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import studio_bridge_server as sb


def test_frame_is_little_endian_and_roundtrips():
    msg = {"type": "pull.runs", "id": 7, "data": {"group": "champion"}}
    blob = sb.frame(msg)
    # prefix is little-endian u32 (matches Godot encode_u32) — NOT big-endian
    (n_le,) = struct.unpack("<I", blob[:4])
    (n_be,) = struct.unpack(">I", blob[:4])
    assert n_le == len(blob) - 4
    assert n_be != n_le or n_le < 256  # for a small payload the two differ
    assert json.loads(blob[4:].decode("utf-8")) == msg


def test_make_echoes_type_and_id():
    m = sb.make("hello", {"version": 1}, 3)
    assert m == {"type": "hello", "id": 3, "data": {"version": 1}}


def _write_run(root: Path, name: str, rows: list[dict]) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_pull_runs_groups_and_last_step(tmp_path):
    _write_run(tmp_path, "champion-HalfCheetah-v5-s0",
               [{"env_steps": 1000, "eval/return": -300.0},
                {"env_steps": 2000, "eval/return": -120.0}])
    _write_run(tmp_path, "champion-HalfCheetah-v5-s1",
               [{"env_steps": 1000, "eval/return": -310.0}])
    reply = sb.handle(sb.make(sb.PULL_RUNS, {}, 1), tmp_path)
    runs = {r["name"]: r for r in reply["data"]["runs"]}
    assert reply["id"] == 1
    assert runs["champion-HalfCheetah-v5-s0"]["group"] == "champion-HalfCheetah-v5"
    assert runs["champion-HalfCheetah-v5-s0"]["last_step"] == 2000.0
    assert runs["champion-HalfCheetah-v5-s1"]["last_step"] == 1000.0


def test_pull_metric_returns_aligned_arrays(tmp_path):
    _write_run(tmp_path, "champion-HalfCheetah-v5-s0",
               [{"env_steps": 1000, "loss/total": 0.08},
                {"env_steps": 2000},                       # row without the key
                {"env_steps": 3000, "loss/total": 0.04, "eval/return": -90.0}])
    reply = sb.handle(sb.make(sb.PULL_METRIC,
                              {"run": "champion-HalfCheetah-v5-s0",
                               "key": "loss/total"}, 5), tmp_path)
    d = reply["data"]
    assert reply["id"] == 5
    assert d["steps"] == [1000.0, 3000.0]      # the keyless row is skipped
    assert d["values"] == [0.08, 0.04]


def test_pull_metric_unknown_key_is_empty_not_crash(tmp_path):
    _write_run(tmp_path, "r-s0", [{"env_steps": 1000, "loss/total": 0.08}])
    reply = sb.handle(sb.make(sb.PULL_METRIC,
                              {"run": "r-s0", "key": "does/not/exist"}, 0), tmp_path)
    assert reply["data"]["steps"] == [] and reply["data"]["values"] == []


def test_env_and_submit_return_not_implemented(tmp_path):
    for t in (sb.ENV_STEP, sb.ENV_RESET, sb.SUBMIT_SPEC, sb.INFER_RUN):
        reply = sb.handle(sb.make(t, {}, 9), tmp_path)
        assert reply["type"] == sb.ERROR
        assert reply["data"]["code"] == "not_implemented"
        assert reply["id"] == 9


def test_hello_replies_version(tmp_path):
    reply = sb.handle(sb.make(sb.HELLO, {"version": 1, "role": "studio"}, 1), tmp_path)
    assert reply["data"]["version"] == sb.VERSION


def test_end_to_end_over_real_socket(tmp_path):
    """Spin the server on a real socket and round-trip frames as a Godot client
    would — proves the TCP framing + dispatch work, not just the pure handler."""
    import socket
    import threading
    _write_run(tmp_path, "champion-Pendulum-v1-s0",
               [{"env_steps": 500, "eval/return": -900.0},
                {"env_steps": 1000, "eval/return": -200.0}])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)

    def _serve_one():
        conn, _ = srv.accept()
        with conn:
            while True:
                msg = sb.read_frame(conn)
                if msg is None:
                    break
                conn.sendall(sb.frame(sb.handle(msg, tmp_path)))
        srv.close()

    t = threading.Thread(target=_serve_one, daemon=True)
    t.start()
    cli = socket.create_connection(("127.0.0.1", port), timeout=5)
    cli.sendall(sb.frame(sb.make(sb.HELLO, {"version": 1, "role": "studio"}, 1)))
    assert sb.read_frame(cli)["data"]["version"] == sb.VERSION
    cli.sendall(sb.frame(sb.make(sb.PULL_METRIC,
                {"run": "champion-Pendulum-v1-s0", "key": "eval/return"}, 2)))
    reply = sb.read_frame(cli)
    assert reply["id"] == 2
    assert reply["data"]["values"] == [-900.0, -200.0]
    cli.close()
    t.join(timeout=5)
