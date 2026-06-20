#!/usr/bin/env python3
"""bridge_worker.py — the Python data/launch worker behind the Haskell bridge.

The Haskell bridge (``spine/app/bridge``) owns the TCP socket, the wire framing,
and the spectral-house-rule ``validateSpec`` gate. For every verb it does NOT
handle natively (the ``pull.*`` readers, ``submit.*`` launch, ``search.*``,
``run.cancel``) it relays the request to THIS long-lived process over
stdin/stdout using the SAME length-prefixed framing as the wire (4-byte
little-endian length + UTF-8 JSON ``{type,id,data}`` envelope).

We reuse ``StudioBridgeServer.dispatch`` verbatim, so the readers / launcher /
search logic stays the single Python implementation (no duplication). The worker
is PERSISTENT so the ``LaunchRegistry`` (live training subprocess children)
survives across requests — a per-request worker would lose that state.

Stdlib + the sealed, torch-free studio modules only.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))   # mbrl/scripts
_MBRL = os.path.dirname(_HERE)                        # mbrl
# `import studio_bridge_server` (this dir) and `import mbrl...` (src), whether or
# not mbrl is installed editable in the running interpreter.
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_MBRL, "src"))

import studio_bridge_server as sbs  # noqa: E402  (the stdlib-only seal)


def _read_exactly(stream, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_frame(stream):
    hdr = _read_exactly(stream, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("<I", hdr)
    payload = _read_exactly(stream, n)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _write_frame(stream, msg: dict) -> None:
    payload = json.dumps(msg).encode("utf-8")
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


# ---- monitor.poll: incremental run streaming (rows + logs + status) ----------

_STEP_KEYS = ("env_steps", "total_env_steps", "global_step", "step", "_step")


def _xstep(row):
    for k in _STEP_KEYS:
        v = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _monitor_poll(srv, data):
    """For each requested run, return the NEW metric rows (since the cursor step),
    NEW log lines (since the cursor line), the launch status, and an advanced
    cursor; terminal runs also get a result envelope. Cursor-driven + stateless so
    the Studio can call it on a timer to stream a run live.

    request : {runs: [name], cursors: {name: {step: float, line: int}}}
    reply   : {runs: {name: {rows, logs, status, cursor, result?}}}
    """
    run_names = data.get("runs") or []
    cursors = data.get("cursors") or {}
    out = {}
    for run in run_names:
        run = str(run)
        try:
            cur = cursors.get(run) or {}
            try:
                since_step = float(cur.get("step"))
            except (TypeError, ValueError):
                since_step = float("-inf")
            try:
                since_line = int(cur.get("line", 0) or 0)
            except (TypeError, ValueError):
                since_line = 0

            rows = []
            last_step = since_step
            mpath = srv.results_dir / run / "metrics.jsonl"
            if mpath.is_file():
                for row in sbs._read_rows(mpath):
                    s = _xstep(row)
                    if s is not None and s > since_step:
                        rows.append(row)
                        if s > last_step:
                            last_step = s

            tail = srv.launcher.tail(run, since_line)
            status = srv.launcher.status(run)
            entry = {
                "rows": rows,
                "logs": tail.get("lines", []),
                "status": status,
                "cursor": {
                    "step": (None if last_step == float("-inf") else last_step),
                    "line": int(tail.get("next_line", since_line)),
                },
            }
            st = status.get("state")
            if st in ("finished", "failed"):
                entry["result"] = {
                    "id": run, "task": "train",
                    "status": "pass" if st == "finished" else "fail",
                    "ts": _now_iso(), "seed": None, "params": {},
                    "value": (rows[-1] if rows else {}),
                    "evidence": {"exit_code": status.get("exit_code")},
                    "blockers": [],
                }
            out[run] = entry
        except Exception as exc:  # one bad run must not sink the whole poll
            out[run] = {"error": repr(exc)}
    return {"runs": out}


def main() -> None:
    # The frame channel is the ORIGINAL stdout fd. StudioBridgeServer logs via
    # print()/_log to stdout, which would corrupt the frames — so dup the real
    # stdout for frames, then redirect fd 1 (and sys.stdout) to stderr so any
    # logging the dispatch does is harmless.
    frame_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    srv = sbs.StudioBridgeServer(dry_run=os.environ.get("SPINE_DRY_RUN") == "1")
    inp = sys.stdin.buffer
    sys.stderr.write("bridge_worker: ready\n")
    sys.stderr.flush()
    while True:
        try:
            msg = _read_frame(inp)
        except Exception as exc:  # malformed frame on stdin is fatal to this worker
            sys.stderr.write(f"bridge_worker: frame error: {exc!r}\n")
            break
        if msg is None:
            break  # the bridge closed stdin
        try:
            if msg.get("type") == "monitor.poll":
                reply = {"type": "monitor.poll", "id": msg.get("id", 0),
                         "data": _monitor_poll(srv, msg.get("data") or {})}
            else:
                reply = srv.dispatch(msg)
        except Exception as exc:  # never crash the worker on a bad request
            reply = {"type": "error", "id": msg.get("id", 0),
                     "data": {"code": "worker_error", "message": repr(exc)}}
        _write_frame(frame_out, reply)


if __name__ == "__main__":
    main()
