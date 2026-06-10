"""Apparatus-side of the MBRL Studio bridge — the ONE boundary from Python.

Counterpart to godot_studio/addons/mbrl_bridge/bridge.gd. The Godot Bridge
connects OUT to this server; this is the single contact point on the Python
side (godot_studio/docs/architecture.md §1).

Two responsibilities over ONE TCP socket, multiplexed by message `type`:
  * VIZ PULL (read-only): pull.runs / pull.metric scan results/runs and
    checkpoints/ — no training code, stays inside the seal.
  * AUTHOR + LAUNCH: submit.spec takes a ModelSpec authored in the Studio's node
    graph, writes a Hydra experiment yaml + overrides (mbrl.studio.spec_to_config)
    and SPAWNS `python scripts/train.py <overrides>` as a SUBPROCESS. The runner
    never imports the Trainer / torch itself — launching out-of-process is what
    keeps the boundary thin and respects docs/remote_execution.md §1 (training
    code never crosses into the boundary module).

The train seam (env.*) is served on the GODOT side (bridge.gd serve_env, the
counterpart to scripts/godot_env.py) — Godot IS the environment, so here env.*
and infer.* return a documented `not_served` stub.

Wire format (byte-exact with protocol.gd frame()/parse()): a 4-byte
LITTLE-ENDIAN unsigned length prefix, then a UTF-8 JSON object
{"type","id","data"}. Godot's PackedByteArray.encode_u32 is little-endian, so
the prefix is struct '<I' — NOT network/big-endian.

    python scripts/studio_bridge_server.py --host 127.0.0.1 --port 9009
    python scripts/studio_bridge_server.py --dry-run   # echo the train command, don't launch
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

# Repo root is scripts/.. — the cwd train.py expects (config_path=../configs).
REPO_ROOT = Path(__file__).resolve().parents[1]
# Make `import mbrl.studio...` work the same way train.py wires up src/ on path.
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mbrl.studio import spec_to_overrides, write_experiment_yaml, run_name_for_spec  # noqa: E402
# v0.1 (oversight): stdlib read backends + sweep engine + validator — all inside
# the seal (no torch). Back pull.datasets / pull.surface / submit.sweep + the gate.
from mbrl.studio.run_index import RunIndex  # noqa: E402
from mbrl.studio.surface_index import SurfaceIndex  # noqa: E402
from mbrl.studio.cells_index import CellsIndex  # noqa: E402
from mbrl.studio.sweep import plan_sweep  # noqa: E402
from mbrl.studio.spec_validator import validate_spec  # noqa: E402
# Incremental SQLite metric reader (stdlib sqlite3, inside the seal). Preferred over
# the JSONL scan when a run has a metrics.db; backs pull.metric_since.
from mbrl.studio import metric_db  # noqa: E402
# launch/monitor seam: spawn + track runs (status/cancel/log-tail) so the Studio can
# WATCH a run. Stdlib subprocess, inside the seal; cloud-pluggable surface (A3).
from mbrl.studio.launch import LaunchRegistry  # noqa: E402

VERSION = 1

# message types — mirror protocol.gd's constants exactly
HELLO = "hello"
ENV_RESET = "env.reset"
ENV_STEP = "env.step"
ENV_SPEC = "env.spec"
PULL_RUNS = "pull.runs"
PULL_METRIC = "pull.metric"
PULL_METRIC_SINCE = "pull.metric_since"   # incremental metric pull (env_steps > since)
INFER_LOAD = "infer.load"
INFER_RUN = "infer.run"
SUBMIT_SPEC = "submit.spec"
PULL_DATASETS = "pull.datasets"   # v0.1
PULL_ARTIFACTS = "pull.artifacts"  # F1: per-run artifact manifest
PULL_SURFACE = "pull.surface"     # v0.1
SUBMIT_SWEEP = "submit.sweep"     # v0.1
PULL_SWEEP = "pull.sweep"         # sweep cells: catalog or flattened arm-rows
PULL_RUN_STATUS = "pull.run_status"   # launch/monitor seam
PULL_LAUNCHED = "pull.launched"
PULL_LOG = "pull.log"
RUN_CANCEL = "run.cancel"
ERROR = "error"

# Authored experiment yamls land here (gitignored authoring dir, NOT configs/ —
# another agent owns configs/). Hydra is pointed at it via a search-path append
# in the launched command so `+experiment=<name>` resolves.
STUDIO_EXP_DIR = REPO_ROOT / "results" / "studio" / "experiments"


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


class FrameDecoder:
    """Accumulates raw bytes and yields whole length-prefixed frames.

    The clean recv loop the Bridge uses (bridge.gd `_drain`): feed it whatever
    `recv()` returned, pull out every complete '<I'-prefixed frame, keep the
    partial tail buffered for the next chunk.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes):
        """Add bytes; yield each fully-arrived frame as a parsed dict."""
        self._buf.extend(chunk)
        while True:
            if len(self._buf) < 4:
                return
            (n,) = struct.unpack("<I", self._buf[:4])
            if len(self._buf) < 4 + n:
                return
            payload = bytes(self._buf[4:4 + n])
            del self._buf[:4 + n]
            yield json.loads(payload.decode("utf-8"))


# ---------------- viz scanning (read-only over results / checkpoints) --------
def _read_rows(metrics: Path) -> list[dict]:
    out = []
    for line in metrics.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn write from a killed run; skip
    return out


def _read_metric_jsonl(results_dir: Path, run: str, key: str,
                       since: float | None = None) -> dict:
    """JSONL fallback: {run, key, steps, values} for `key` in metrics.jsonl.

    Empty arrays if the run or key is absent — never raises (a missing key is a
    normal query). x-axis is env_steps (preferred) or step. When `since` is given,
    only rows with env_steps > since are returned (ascending file order).
    """
    steps: list[float] = []
    values: list[float] = []
    m = results_dir / run / "metrics.jsonl"
    if m.exists():
        for row in _read_rows(m):
            if key in row:
                step = row.get("env_steps", row.get("step"))
                if step is not None:
                    s = float(step)
                    if since is not None and s <= float(since):
                        continue
                    steps.append(s)
                    values.append(float(row[key]))
    return {"run": run, "key": key, "steps": steps, "values": values}


def read_metric(results_dir: Path, run: str, key: str) -> dict:
    """{run, key, steps, values} for `key`, preferring the SQLite metrics.db.

    Uses the per-run metrics.db (stdlib sqlite3 reader, inside the seal) when it
    exists; otherwise falls back to the UNCHANGED metrics.jsonl scan. Empty arrays
    if the run or key is absent — never raises.

    `results_dir` here IS the runs dir (server convention: self.results_dir =
    <root>/runs). The metric_db contract is keyed off the results ROOT
    (<root>/runs/<run>/metrics.db), so we hand it results_dir.parent — the same
    parent RunIndex/SurfaceIndex are constructed with.
    """
    root = results_dir.parent
    if metric_db.has_db(root, run):
        return metric_db.read_metric_db(root, run, key)
    return _read_metric_jsonl(results_dir, run, key)


def read_metric_since(results_dir: Path, run: str, key: str, since: float) -> dict:
    """Incremental {run, key, steps, values}: only rows with env_steps > `since`.

    Prefers the SQLite metrics.db (its index makes the > since slice cheap); falls
    back to filtering the JSONL read by `since` when no db is present. `results_dir`
    is the runs dir; the db lives off its parent (see read_metric).
    """
    root = results_dir.parent
    if metric_db.has_db(root, run):
        return metric_db.read_metric_since(root, run, key, since)
    return _read_metric_jsonl(results_dir, run, key, since=since)


# (A8 2026-06-09: the legacy pure handle() + list_runs/scan_runs are DELETED —
# one server, one read path. dispatch() below is the only request->reply map;
# tests/test_studio_bridge.py now exercises it directly.)

# ---------------- the server: author + launch + viz over one socket ----------
def _log(event: str, **fields) -> None:
    """One structured JSON line per event to stdout (greppable, parseable)."""
    rec = {"t": round(time.time(), 3), "src": "studio-bridge", "event": event}
    rec.update(fields)
    print(json.dumps(rec, default=str), flush=True)


class StudioBridgeServer:
    """TCP server that authors AND launches training runs over one socket.

    Construct, then call serve_forever() (blocks) or start() (background thread).
    dispatch(msg) is the pure-ish request->reply map used by both the socket loop
    and the tests; submit.spec is the only side-effecting branch (it spawns
    train.py, or records the command when dry_run is set).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9009,
                 repo_root: Path | None = None,
                 results_dir: Path | None = None,
                 checkpoints_dir: Path | None = None,
                 experiments_dir: Path | None = None,
                 dry_run: bool | None = None,
                 python_exe: str | None = None,
                 strict_validation: bool = False) -> None:
        self.host = host
        self.port = port
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.results_dir = (Path(results_dir) if results_dir
                            else self.repo_root / "results" / "runs")
        self.checkpoints_dir = (Path(checkpoints_dir) if checkpoints_dir
                                else self.repo_root / "checkpoints")
        self.experiments_dir = (Path(experiments_dir) if experiments_dir
                                else STUDIO_EXP_DIR)
        # Dry-run: honor the constructor flag, else the MBRL_STUDIO_DRYRUN env var.
        if dry_run is None:
            dry_run = os.environ.get("MBRL_STUDIO_DRYRUN", "") in ("1", "true", "True")
        self.dry_run = bool(dry_run)
        self.python_exe = python_exe or sys.executable
        # v0.1: True => submit.spec/submit.sweep REJECT specs that trip the spectral
        # house rules; default False = warn-but-launch (compile.gd warning parity).
        self.strict_validation = bool(strict_validation)
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self.launched: list[dict] = []   # record of spawned/recorded commands
        # launch/monitor seam: tracks live train.py children + per-run logs under
        # results/logs/studio, backing pull.run_status / pull.launched / pull.log / run.cancel.
        self.launcher = LaunchRegistry(self.repo_root / "results" / "logs" / "studio")

    # ---- the heart: turn a ModelSpec into a launched (or recorded) command ----
    def build_train_command(self, spec: dict, seed: int | None = None) -> tuple[list[str], str, Path]:
        """ModelSpec -> (argv, run_name, experiment_yaml_path).

        Writes the experiment yaml first (so the run is reproducible from a file),
        then assembles `python scripts/train.py <overrides>` with a hydra
        search-path append pointing at the authored yaml's directory.
        """
        run_name = run_name_for_spec(spec, seed=seed)
        # name the experiment file after the spec's experiment name (or the run)
        exp = spec.get("experiment")
        exp_name = (exp.get("name") if isinstance(exp, dict) and exp.get("name")
                    else (exp if isinstance(exp, str) and exp else run_name))
        yaml_path = write_experiment_yaml(spec, self.experiments_dir, str(exp_name))

        overrides = spec_to_overrides(spec)
        if seed is not None:
            overrides = [o for o in overrides if not o.startswith("seed=")]
            overrides.append(f"seed={int(seed)}")

        argv = [self.python_exe, "scripts/train.py"]
        # let `+experiment=<name>` resolve the just-written yaml without touching
        # configs/ (owned by another agent): append its dir to Hydra's searchpath
        argv.append(f"hydra.searchpath=[file://{yaml_path.parent.as_posix()}]")
        argv.extend(overrides)
        return argv, run_name, yaml_path

    def submit_spec(self, data: dict) -> dict:
        """Handle submit.spec's data: write config, launch (or record) train.py.

        Returns the reply `data`: {accepted, run_name, command[, dry_run, pid]}.
        Never raises into the socket loop — a bad spec returns accepted=false with
        an error string so the Studio can surface it.
        """
        spec = data.get("model_spec", data.get("spec", {}))
        if not isinstance(spec, dict):
            return {"accepted": False, "error": "model_spec must be an object"}
        warnings = validate_spec(spec)   # v0.1 spectral house-rule gate
        if self.strict_validation and warnings:
            return {"accepted": False, "warnings": warnings,
                    "error": "spec violates the spectral house rules (strict mode)"}
        seed = data.get("seed")
        try:
            argv, run_name, yaml_path = self.build_train_command(
                spec, seed=int(seed) if seed is not None else None)
        except Exception as exc:  # noqa: BLE001 — surface authoring errors, don't crash
            _log("submit_spec.error", error=repr(exc))
            return {"accepted": False, "error": f"spec->config failed: {exc}"}

        reply = {"accepted": True, "run_name": run_name,
                 "command": argv, "experiment_yaml": str(yaml_path),
                 "warnings": warnings}

        if self.dry_run:
            reply["dry_run"] = True
            self.launched.append({"run_name": run_name, "command": argv, "dry_run": True})
            _log("submit_spec.dryrun", run_name=run_name, command=argv)
            return reply

        try:
            info = self.launcher.launch(run_name, argv, self.repo_root)
        except Exception as exc:  # noqa: BLE001
            _log("submit_spec.launch_failed", error=repr(exc), command=argv)
            return {"accepted": False, "run_name": run_name,
                    "command": argv, "error": f"launch failed: {exc}"}
        reply["pid"] = info["pid"]
        reply["log_path"] = info["log_path"]   # the panel can pull.log this run
        self.launched.append({"run_name": run_name, "command": argv, "pid": info["pid"]})
        _log("submit_spec.launched", run_name=run_name, pid=info["pid"], command=argv)
        return reply

    # ---- submit.sweep: expand a SweepSpec -> launch every arm (v0.1) ----
    def submit_sweep(self, data: dict) -> dict:
        """Expand {base_spec, axes, seeds} into arms (mbrl.studio.sweep), then
        author+launch (or dry-run record) each via the submit.spec machinery.
        Returns {accepted, group, n, runs, warnings_by_arm[, dry_run]}. In strict
        mode a sweep with any house-rule warning is rejected wholesale."""
        base = data.get("base_spec", data.get("base", {}))
        if not isinstance(base, dict):
            return {"accepted": False, "error": "base_spec must be an object"}
        try:
            plan = plan_sweep(base, data.get("axes", []), data.get("seeds", [0]),
                              group=data.get("group"))
        except Exception as exc:  # noqa: BLE001
            _log("submit_sweep.error", error=repr(exc))
            return {"accepted": False, "error": f"sweep expansion failed: {exc}"}

        warnings_by_arm = {a.run_name: a.warnings for a in plan.arms if a.warnings}
        if self.strict_validation and not plan.ok:
            return {"accepted": False, "group": plan.group, "n": plan.n,
                    "warnings_by_arm": warnings_by_arm,
                    "error": "one or more arms violate the spectral house rules"}

        runs = []
        for arm in plan.arms:
            r = self.submit_spec({"model_spec": arm.spec, "seed": arm.seed})
            runs.append({"run_name": r.get("run_name", arm.run_name),
                         "accepted": r.get("accepted", False)})
        reply = {"accepted": True, "group": plan.group, "n": plan.n,
                 "runs": runs, "warnings_by_arm": warnings_by_arm}
        if self.dry_run:
            reply["dry_run"] = True
        _log("submit_sweep.done", group=plan.group, n=plan.n, dry_run=self.dry_run)
        return reply

    # ---- dispatch: request -> reply, echoing id ----
    def dispatch(self, msg: dict) -> dict:
        type_ = msg.get("type", "")
        data = msg.get("data", {}) or {}
        id_ = msg.get("id", 0)

        if type_ == HELLO:
            return make(HELLO, {"version": VERSION}, id_)

        if type_ == PULL_RUNS:
            # the ONE canonical reader (RunIndex): group = meta.json["group"] else
            # seed-stripped name; {n_points, keys} per the contract;
            # include_checkpoints unions checkpoint-only runs.
            idx = RunIndex(self.results_dir.parent, ckpt_root=self.checkpoints_dir)
            runs = idx.list_runs(group=data.get("group") or None,
                                 include_checkpoints=True)
            return make(PULL_RUNS, {"runs": runs}, id_)

        if type_ == PULL_METRIC:
            return make(PULL_METRIC, read_metric(
                self.results_dir, str(data.get("run", "")), str(data.get("key", ""))), id_)

        if type_ == PULL_METRIC_SINCE:
            try:
                since = float(data.get("since", 0.0) or 0.0)
            except (TypeError, ValueError):
                since = 0.0
            return make(PULL_METRIC_SINCE, read_metric_since(
                self.results_dir, str(data.get("run", "")),
                str(data.get("key", "")), since), id_)

        if type_ == SUBMIT_SPEC:
            return make(SUBMIT_SPEC, self.submit_spec(data), id_)

        # v0.1 read verbs (stdlib backends, inside the seal) + batch submit
        if type_ == PULL_DATASETS:
            idx = RunIndex(self.results_dir.parent, ckpt_root=self.checkpoints_dir)
            return make(PULL_DATASETS,
                        {"items": idx.list_datasets(data.get("kind"))}, id_)
        if type_ == PULL_ARTIFACTS:
            idx = RunIndex(self.results_dir.parent, ckpt_root=self.checkpoints_dir)
            return make(PULL_ARTIFACTS,
                        {"artifacts": idx.list_artifacts(str(data.get("run", "")))}, id_)
        if type_ == PULL_SURFACE:
            surf = SurfaceIndex(self.results_dir.parent).get_surface(
                str(data.get("run", "")), data.get("step"))
            return make(PULL_SURFACE, surf, id_)
        if type_ == SUBMIT_SWEEP:
            return make(SUBMIT_SWEEP, self.submit_sweep(data), id_)
        if type_ == PULL_SWEEP:
            # sweep OUTCOME grids (results/*_cells.jsonl — NOT time-series): no name
            # asked → the catalog; a name → that sweep's flattened arm-rows. The
            # cells files live at the results ROOT (results_dir is <root>/runs).
            cells = CellsIndex(self.results_dir.parent)
            sweep_name = str(data.get("sweep", "") or "")
            if not sweep_name:
                return make(PULL_SWEEP, {"sweeps": cells.list_sweeps()}, id_)
            return make(PULL_SWEEP, cells.read_cells(sweep_name), id_)

        # launch/monitor seam — watch + control the runs the server launched
        if type_ == PULL_RUN_STATUS:
            return make(PULL_RUN_STATUS,
                        self.launcher.status(str(data.get("run", ""))), id_)
        if type_ == PULL_LAUNCHED:
            return make(PULL_LAUNCHED, {"runs": self.launcher.list()}, id_)
        if type_ == PULL_LOG:
            try:
                since = int(data.get("since_line", 0) or 0)
            except (TypeError, ValueError):
                since = 0
            return make(PULL_LOG,
                        self.launcher.tail(str(data.get("run", "")), since), id_)
        if type_ == RUN_CANCEL:
            return make(RUN_CANCEL,
                        self.launcher.cancel(str(data.get("run", ""))), id_)

        # env.* / infer.* are served by the GODOT side (the train seam). Document
        # the boundary explicitly rather than silently dropping.
        if type_ in (ENV_RESET, ENV_STEP, ENV_SPEC):
            return make(ERROR, {"code": "not_served",
                                "message": f"{type_}: the env train seam is served by "
                                           "Godot (bridge.gd serve_env), not the runner"}, id_)
        if type_ in (INFER_LOAD, INFER_RUN):
            return make(ERROR, {"code": "not_served",
                                "message": f"{type_}: inference runs in-engine on a frozen "
                                           ".onnx (architecture.md §1), not the runner"}, id_)

        return make(ERROR, {"code": "unknown_type", "message": type_}, id_)

    # ---- socket lifecycle ----
    def _handle_client(self, conn: socket.socket, addr) -> None:
        decoder = FrameDecoder()
        _log("client.connected", peer=str(addr))
        try:
            with conn:
                while not self._stop.is_set():
                    chunk = conn.recv(65536)
                    if not chunk:
                        break  # peer closed
                    for msg in decoder.feed(chunk):
                        reply = self.dispatch(msg)
                        _log("rx", type=msg.get("type"), id=msg.get("id"))
                        conn.sendall(frame(reply))
        except (ConnectionError, OSError) as exc:
            _log("client.error", peer=str(addr), error=repr(exc))
        finally:
            _log("client.disconnected", peer=str(addr))

    def start(self) -> "StudioBridgeServer":
        """Bind + listen + accept loop on a daemon thread. Returns self.

        Use .port after start() to read the bound port when constructed with
        port=0 (ephemeral) — handy for tests.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        self.port = srv.getsockname()[1]
        self._srv = srv
        self._stop.clear()
        _log("listening", host=self.host, port=self.port,
             results=str(self.results_dir), checkpoints=str(self.checkpoints_dir),
             dry_run=self.dry_run)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        return self

    def _accept_loop(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                break  # socket closed under us (stop())
            threading.Thread(target=self._handle_client,
                             args=(conn, addr), daemon=True).start()

    def serve_forever(self) -> None:
        """Blocking serve: start() then wait until stop()/KeyboardInterrupt."""
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.25)
        except KeyboardInterrupt:
            _log("shutdown", reason="keyboard_interrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
            self._srv = None


def main() -> None:
    p = argparse.ArgumentParser(description="MBRL Studio bridge server (Python side).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9009)
    p.add_argument("--results-dir", default=None,
                   help="runs dir (default: <repo>/results/runs)")
    p.add_argument("--checkpoints-dir", default=None,
                   help="checkpoints dir (default: <repo>/checkpoints)")
    p.add_argument("--dry-run", action="store_true",
                   help="echo the train command on submit.spec instead of launching")
    p.add_argument("--strict", action="store_true",
                   help="reject specs/sweeps that trip the spectral house rules")
    args = p.parse_args()
    server = StudioBridgeServer(
        host=args.host, port=args.port,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        checkpoints_dir=Path(args.checkpoints_dir) if args.checkpoints_dir else None,
        dry_run=args.dry_run or None,
        strict_validation=args.strict,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
