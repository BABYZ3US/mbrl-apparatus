"""Apparatus-side of the MBRL Studio bridge — the ONE boundary from Python.

Counterpart to godot_studio/addons/mbrl_bridge/bridge.gd. The Godot Bridge
connects OUT to this server; this is the single contact point on the Python
side (godot_studio/docs/architecture.md §1). It does NOT import the Trainer or
touch training — the viz path is read-only over the sha-scoped results JSONL,
so this module is STDLIB ONLY and respects the seal (no torch/numpy).

Wire format (byte-exact with protocol.gd frame()/parse()): a 4-byte
LITTLE-ENDIAN unsigned length prefix, then a UTF-8 JSON object
{"type","id","data"}. Godot's PackedByteArray.encode_u32 is little-endian, so
the prefix is struct '<I' — NOT network/big-endian.

Phasing: this implements the P2 viz-pull path (pull.runs / pull.metric). The
train seam (env.*) and submit.spec / infer.* return a structured
not_implemented stub until P3/P5.

    python scripts/studio_bridge_server.py --port 9009 --results-dir results/runs
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import struct
from pathlib import Path

VERSION = 1

# message types — mirror protocol.gd's constants exactly
HELLO = "hello"
ENV_RESET = "env.reset"
ENV_STEP = "env.step"
ENV_SPEC = "env.spec"
PULL_RUNS = "pull.runs"
PULL_METRIC = "pull.metric"
INFER_LOAD = "infer.load"
INFER_RUN = "infer.run"
SUBMIT_SPEC = "submit.spec"
ERROR = "error"

_SEED_SUFFIX = re.compile(r"-s\d+$")   # group naming matches scripts/status.py


# ---------------- framing (byte-exact with protocol.gd) ----------------
def frame(msg: dict) -> bytes:
    """4-byte little-endian length prefix + UTF-8 JSON. Matches encode_u32."""
    payload = json.dumps(msg).encode("utf-8")
    return struct.pack("<I", len(payload)) + payload


def read_frame(sock: socket.socket) -> dict | None:
    """Read one length-prefixed frame from a blocking socket. None on EOF."""
    head = _recv_exactly(sock, 4)
    if head is None:
        return None
    (n,) = struct.unpack("<I", head)
    body = _recv_exactly(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def make(type_: str, data: dict, id_: int = 0) -> dict:
    return {"type": type_, "id": id_, "data": data}


# ---------------- viz handlers (read-only over results JSONL) ----------------
def _last_step(rows: list[dict]) -> float | None:
    for row in reversed(rows):
        if "env_steps" in row:
            return float(row["env_steps"])
        if "step" in row:
            return float(row["step"])
    return None


def _read_rows(metrics: Path) -> list[dict]:
    out = []
    for line in metrics.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def list_runs(results_dir: Path) -> list[dict]:
    runs = []
    if results_dir.exists():
        for d in sorted(results_dir.iterdir()):
            m = d / "metrics.jsonl"
            if not m.exists():
                continue
            rows = _read_rows(m)
            runs.append({
                "name": d.name,
                "group": _SEED_SUFFIX.sub("", d.name),
                "last_step": _last_step(rows),
            })
    return runs


def read_metric(results_dir: Path, run: str, key: str) -> dict:
    """{steps, values} for `key` in run's metrics.jsonl. Empty arrays if the
    run or key is absent — never raises (a missing key is a normal query)."""
    steps: list[float] = []
    values: list[float] = []
    m = results_dir / run / "metrics.jsonl"
    if m.exists():
        for row in _read_rows(m):
            if key in row:
                step = row.get("env_steps", row.get("step"))
                if step is not None:
                    steps.append(float(step))
                    values.append(float(row[key]))
    return {"run": run, "key": key, "steps": steps, "values": values}


# ---------------- dispatch (unit-testable without a socket) ----------------
def handle(msg: dict, results_dir: Path) -> dict:
    """Map a request to a reply, echoing the request id. Pure function over the
    filesystem — the test calls this directly."""
    type_ = msg.get("type", "")
    data = msg.get("data", {})
    id_ = msg.get("id", 0)

    if type_ == HELLO:
        return make(HELLO, {"version": VERSION}, id_)
    if type_ == PULL_RUNS:
        return make(PULL_RUNS, {"runs": list_runs(results_dir)}, id_)
    if type_ == PULL_METRIC:
        return make(PULL_METRIC, read_metric(
            results_dir, str(data.get("run", "")), str(data.get("key", ""))), id_)
    if type_ in (ENV_RESET, ENV_STEP, ENV_SPEC, SUBMIT_SPEC, INFER_LOAD, INFER_RUN):
        return make(ERROR, {"code": "not_implemented",
                            "message": f"{type_} lands in a later phase (P3/P5); "
                                       "this server serves the P2 viz path only"}, id_)
    return make(ERROR, {"code": "unknown_type", "message": type_}, id_)


# ---------------- server loop (single connection; viz is low-traffic) -------
def serve(host: str, port: int, results_dir: Path) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[studio-bridge] listening on {host}:{port}, results={results_dir}",
          flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[studio-bridge] client {addr} connected", flush=True)
            with conn:
                while True:
                    msg = read_frame(conn)
                    if msg is None:
                        break
                    conn.sendall(frame(handle(msg, results_dir)))
            print("[studio-bridge] client disconnected", flush=True)
    except KeyboardInterrupt:
        print("\n[studio-bridge] shutdown", flush=True)
    finally:
        srv.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9009)
    p.add_argument("--results-dir", default="results/runs")
    args = p.parse_args()
    serve(args.host, args.port, Path(args.results_dir))


if __name__ == "__main__":
    main()
