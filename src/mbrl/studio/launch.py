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

import subprocess
import time
from pathlib import Path


class LaunchRegistry:
    """Tracks launched training subprocesses + their per-run log files (in memory).

    One registry lives on the StudioBridgeServer for its lifetime; it holds the live
    Popen handles so status/cancel/tail work across requests. A run is keyed by its
    run_name; re-launching a name supersedes the previous handle.
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self._runs: dict[str, dict] = {}  # run_name -> {popen, _fh, pid, started_at, log_path, argv}

    def launch(self, run_name: str, argv: list, cwd) -> dict:
        """Spawn argv in `cwd`, streaming stdout+stderr to <log_dir>/<run_name>.log."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{run_name}.log"
        fh = open(log_path, "w", buffering=1)  # line-buffered so tail sees live output
        popen = subprocess.Popen(argv, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
        self._runs[run_name] = {
            "popen": popen, "_fh": fh, "pid": popen.pid,
            "started_at": time.time(), "log_path": str(log_path), "argv": list(argv),
        }
        return {"run_name": run_name, "pid": popen.pid,
                "started_at": self._runs[run_name]["started_at"], "log_path": str(log_path)}

    @staticmethod
    def _state(popen) -> tuple[str, int | None]:
        rc = popen.poll()
        if rc is None:
            return "running", None
        return ("finished" if rc == 0 else "failed"), rc

    def status(self, run_name: str) -> dict:
        """{run_name, state, exit_code, pid, started_at, log_path} or state='unknown'."""
        rec = self._runs.get(run_name)
        if rec is None:
            return {"run_name": run_name, "state": "unknown"}
        state, rc = self._state(rec["popen"])
        return {"run_name": run_name, "state": state, "exit_code": rc,
                "pid": rec["pid"], "started_at": rec["started_at"],
                "log_path": rec["log_path"]}

    def list(self) -> list[dict]:
        """Status of every launched run (newest first by start time)."""
        return sorted((self.status(n) for n in self._runs),
                      key=lambda s: s.get("started_at") or 0.0, reverse=True)

    def cancel(self, run_name: str) -> dict:
        """Terminate a running child (TERM, then KILL after 5s). Idempotent."""
        rec = self._runs.get(run_name)
        if rec is None:
            return {"run_name": run_name, "cancelled": False, "state": "unknown"}
        state, _ = self._state(rec["popen"])
        if state != "running":
            return {"run_name": run_name, "cancelled": False, "state": state}
        rec["popen"].terminate()
        try:
            rec["popen"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            rec["popen"].kill()
        return {"run_name": run_name, "cancelled": True, "state": "failed"}

    def tail(self, run_name: str, since_line: int = 0, max_lines: int = 200) -> dict:
        """Incremental log lines from `since_line`. The panel passes back next_line."""
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
