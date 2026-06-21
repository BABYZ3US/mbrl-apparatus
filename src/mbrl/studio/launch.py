"""launch — pluggable run launcher + in-memory status/log registry for the boundary.

submit.spec / submit.sweep author a ``python scripts/train.py ...`` command; this
module actually runs it and TRACKS it so the Studio can monitor a run from afar:

  * launch(run_name, argv, cwd)   -> spawn; stream stdout+stderr to a per-run log file
  * status(run_name)              -> running | finished | failed (+ exit_code)
  * list()                        -> every launched run + its state (the "launches" view)
  * cancel(run_name)              -> terminate a running child
  * tail(run_name, since_line)    -> incremental log lines (live-tail in the panel)

Stdlib only (subprocess, pathlib, time) — inside the seal (no torch). The backend is
local-subprocess today; a cloud (SkyPilot) backend can implement the SAME surface and
be injected without touching the server (ARCH_RECOMMENDATIONS A3, the cloud-launch gap).
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path


class LaunchRegistry:
    """Tracks launched training subprocesses + their per-run log files (in memory).

    One registry lives on the StudioBridgeServer for its lifetime; it holds the live
    Popen handles so status/cancel/tail work across requests. A run is keyed by its
    run_name; re-launching a name supersedes the previous handle.

    Thread-safe: the StudioBridgeServer serves each client on its own thread, so two
    panels (or the search ticker + a manual pull.launched) can hit launch/list/cancel
    concurrently. A lock guards every `_runs` mutation and snapshot — without it,
    `list()` iterating while `launch()` inserts raises "dict changed size during
    iteration" and kills that client.
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self._runs: dict[str, dict] = {}  # run_name -> {popen, _fh, pid, started_at, log_path, argv}
        self._lock = threading.Lock()

    @staticmethod
    def _close_fh(rec: dict) -> None:
        """Release a run's log file handle once (idempotent). Closing a finished/
        superseded run's handle stops the FD leak (one per launched arm otherwise)."""
        fh = rec.get("_fh")
        if fh is not None and not fh.closed:
            try:
                fh.close()
            except OSError:
                pass
        rec["_fh"] = None

    def launch(self, run_name: str, argv: list, cwd, env: dict | None = None) -> dict:
        """Spawn argv in `cwd`, streaming stdout+stderr to <log_dir>/<run_name>.log.

        `env`, when given, is MERGED over the inherited process environment for the
        CHILD only (e.g. {"WANDB_API_KEY": ..., "WANDB_MODE": "online"} the Studio
        passed in) — secrets reach the run via the env, never the argv/yaml/logs.
        None (the default) inherits the server's env unchanged (byte-exact).
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{run_name}.log"
        fh = open(log_path, "w", buffering=1)  # line-buffered so tail sees live output
        child_env = None
        if env:
            child_env = os.environ.copy()
            child_env.update({str(k): str(v) for k, v in env.items()})
        popen = subprocess.Popen(argv, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                                 env=child_env)
        rec = {"popen": popen, "_fh": fh, "pid": popen.pid,
               "started_at": time.time(), "log_path": str(log_path), "argv": list(argv)}
        with self._lock:
            superseded = self._runs.get(run_name)
            self._runs[run_name] = rec
        if superseded is not None:
            self._close_fh(superseded)  # don't leak the replaced run's log handle
        return {"run_name": run_name, "pid": popen.pid,
                "started_at": rec["started_at"], "log_path": str(log_path)}

    def _state(self, rec: dict) -> tuple[str, int | None]:
        """(state, exit_code). Reaps the child (poll) and closes its log handle on
        the running->terminal transition."""
        rc = rec["popen"].poll()
        if rc is None:
            return "running", None
        self._close_fh(rec)
        return ("finished" if rc == 0 else "failed"), rc

    @staticmethod
    def _status_dict(run_name: str, rec: dict, state: str, rc: int | None) -> dict:
        return {"run_name": run_name, "state": state, "exit_code": rc,
                "pid": rec["pid"], "started_at": rec["started_at"],
                "log_path": rec["log_path"]}

    def status(self, run_name: str) -> dict:
        """{run_name, state, exit_code, pid, started_at, log_path} or state='unknown'."""
        with self._lock:
            rec = self._runs.get(run_name)
            if rec is None:
                return {"run_name": run_name, "state": "unknown"}
            state, rc = self._state(rec)
            return self._status_dict(run_name, rec, state, rc)

    def list(self) -> list[dict]:
        """Status of every launched run (newest first by start time)."""
        with self._lock:
            snapshot = list(self._runs.items())  # snapshot under lock; build off it
            out = [self._status_dict(name, rec, *self._state(rec))
                   for name, rec in snapshot]
        return sorted(out, key=lambda s: s.get("started_at") or 0.0, reverse=True)

    def cancel(self, run_name: str) -> dict:
        """Terminate a running child (TERM, then KILL after 5s). Idempotent."""
        with self._lock:
            rec = self._runs.get(run_name)
            state = self._state(rec)[0] if rec is not None else "unknown"
        if rec is None:
            return {"run_name": run_name, "cancelled": False, "state": "unknown"}
        if state != "running":
            return {"run_name": run_name, "cancelled": False, "state": state}
        # terminate/wait OUTSIDE the lock so a slow shutdown doesn't block other clients
        rec["popen"].terminate()
        try:
            rec["popen"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            rec["popen"].kill()
        self._close_fh(rec)
        return {"run_name": run_name, "cancelled": True, "state": "failed"}

    def tail(self, run_name: str, since_line: int = 0, max_lines: int = 200) -> dict:
        """Incremental log lines from `since_line`. The panel passes back next_line."""
        with self._lock:
            rec = self._runs.get(run_name)
        if rec is None:
            return {"run_name": run_name, "lines": [], "next_line": int(since_line),
                    "total_lines": 0}
        try:
            all_lines = Path(rec["log_path"]).read_text().splitlines()
        except OSError:
            all_lines = []
        start = max(0, int(since_line))
        chunk = all_lines[start:start + int(max_lines)]
        return {"run_name": run_name, "lines": chunk,
                "next_line": start + len(chunk), "total_lines": len(all_lines)}
