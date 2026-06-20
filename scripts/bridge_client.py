#!/usr/bin/env python3
"""bridge_client.py — a tiny framed TCP client for the Studio bridge.

Connects to the (Haskell) spine-bridge, sends one or more verbs, prints the
replies. Handy for end-to-end smoke tests of the bridge ↔ worker ↔ mbrl loop
without the Godot front-end. Stdlib only.

Usage:
    python3 bridge_client.py [--host H] [--port N] [verb [json-data]] ...
    # default: run a built-in smoke sequence (hello, pull.runs, a bad submit.spec)
"""
from __future__ import annotations

import json
import socket
import struct
import sys


def send_frame(sock: socket.socket, msg: dict) -> None:
    payload = json.dumps(msg).encode("utf-8")
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def recv_frame(sock: socket.socket) -> dict | None:
    hdr = _recv_exactly(sock, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("<I", hdr)
    body = _recv_exactly(sock, n)
    return None if body is None else json.loads(body.decode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def call(host: str, port: int, verb: str, data: dict, mid: int = 1) -> dict | None:
    with socket.create_connection((host, port), timeout=10) as s:
        send_frame(s, {"type": verb, "id": mid, "data": data})
        return recv_frame(s)


def smoke(host: str, port: int) -> int:
    print(f"== spine-bridge smoke @ {host}:{port} ==")
    print("hello       ->", call(host, port, "hello", {"version": 1, "role": "client"}))
    runs = call(host, port, "pull.runs", {}) or {}
    rlist = runs.get("data", {}).get("runs", [])
    print(f"pull.runs   -> {len(rlist)} runs; first:",
          (rlist[0].get("name") if rlist else None))
    bad = call(host, port, "submit.spec", {"model_spec": {
        "spectral": {"enabled": True},
        "model": {"latent_cap_mult": 1},
        "penalty": {"schedule": {"kind": "cuberoot", "floor": 0}},
    }})
    print("submit(bad)  ->", bad.get("data") if bad else None)
    return 0


def main(argv: list[str]) -> int:
    host, port = "127.0.0.1", 9009
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--host":
            host = argv[i + 1]; i += 2
        elif argv[i] == "--port":
            port = int(argv[i + 1]); i += 2
        else:
            rest.append(argv[i]); i += 1
    if not rest:
        return smoke(host, port)
    verb = rest[0]
    data = json.loads(rest[1]) if len(rest) > 1 else {}
    print(json.dumps(call(host, port, verb, data), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
