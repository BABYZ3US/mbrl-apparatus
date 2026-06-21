"""End-to-end coverage of the Studio submit -> launch -> metrics-readback loop.

This is the full trip the Godot studio drives over the bridge: a champion-like
ModelSpec authored in the node graph arrives as submit.spec, the runner authors a
Hydra experiment yaml + overrides (mbrl.studio.spec_to_config) and either RECORDS
the `python scripts/train.py ...` command (dry-run) or SPAWNS it (LaunchRegistry),
and the same server reads the resulting metric curve back via read_metric /
pull.metric.

Two layers, kept independent so `-m "not slow"` runs PART A alone:

  PART A (fast, deterministic, NO torch): submit.spec in dry-run asserts the exact
    train argv + the spec->overrides mapping, and that a bad spec is REJECTED at
    the boundary and never produces a launch command. Mirrors the seal the rest of
    tests/test_studio_bridge_server*.py keep (stdlib + the server / spec_to_config
    / spec_validator modules only — never imports torch or scripts/train.py).

  PART B (@pytest.mark.slow, REAL training smoke): a tiny Pendulum spec launched
    for real through LaunchRegistry; we poll the run's metrics.jsonl and assert a
    metric reads back through the server. This is the ONLY path here that runs the
    training stack — it spawns scripts/train.py as a subprocess (out of process,
    so the seal still holds for the importing test module).

Run:
    pytest -q tests/test_studio_e2e.py -m "not slow"   # PART A only (no torch)
    pytest -q tests/test_studio_e2e.py                 # + the PART B slow smoke
"""
import json
import sys
import time
from pathlib import Path

import pytest

# scripts/ on path for the server module; src/ for the pure mapping + validator.
# (Identical wiring to tests/test_studio_bridge_server.py / _v01.py.)
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import studio_bridge_server as sb
from mbrl.studio import spec_to_config as s2c
from mbrl.studio import spec_validator as sv


# ============================ PART A — fast, no torch ============================

def _clean_champion_spec() -> dict:
    """A realistic CLEAN ModelSpec — what compile.gd ships for a champion-like
    graph submit. Deliberately NON-spectral (no spectral block) so NO spectral
    house rule applies, and no non-default algo.* selector (those warn), so
    validate_spec() returns [] and the submit is unconditionally accepted.
    """
    return {
        "experiment": {"name": "champion"},
        "env": {"name": "Walker2d-v5"},
        "seed": 3,
        "model": {"encoder": "mlp", "dynamics": "gaussian"},
        "training": {"total_env_steps": 200000},
    }


def test_partA_submit_dryrun_records_exact_train_command(tmp_path):
    """submit.spec in DRY-RUN: accepted, the recorded argv is
    [<python>, "scripts/train.py", <hydra.searchpath>, *spec_to_overrides(spec)],
    and NO real process is spawned (the run is recorded, not launched).
    """
    # experiments_dir under tmp_path so the authored yaml never lands in the real
    # repo's results/studio/experiments (which is the module-level default).
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True,
                                experiments_dir=tmp_path / "experiments")
    assert srv.dry_run is True

    spec = _clean_champion_spec()
    reply = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": spec}, 101))
    assert reply["type"] == sb.SUBMIT_SPEC and reply["id"] == 101
    d = reply["data"]

    # accepted, dry-run, clean (no house-rule warnings on a non-spectral graph)
    assert d["accepted"] is True
    assert d["dry_run"] is True
    assert d["warnings"] == []
    assert d["run_name"] == "champion-Walker2d-v5-s3"

    cmd = d["command"]
    # `python scripts/train.py ...` — sys.executable then the Hydra entry point.
    assert cmd[0] == sys.executable
    assert cmd[1] == "scripts/train.py"
    # The server appends a hydra.searchpath pointing at the authored yaml's GROUP
    # PARENT dir (yaml lives at <dir>/experiment/<name>.yaml) so `+experiment=<name>`
    # resolves without touching configs/ (owned elsewhere).
    assert cmd[2].startswith("hydra.searchpath=[file://")
    assert Path(d["experiment_yaml"]).parent.parent.as_posix() in cmd[2]

    # The tail of the command is EXACTLY what the pure mapping produces, in order:
    # group selectors first (+experiment / env=), then the sorted dotted flatten.
    expected_overrides = s2c.spec_to_overrides(spec)
    assert cmd[3:] == expected_overrides

    # Spell out the spec->overrides mapping this asserts:
    #   experiment.name "champion"  -> "+experiment=champion"  (group-add form, FIRST)
    #   env.name "Walker2d-v5"      -> "env=walker2d"           (group-select form, FIRST)
    #   seed 3                      -> "seed=3"                 (flat scalar)
    #   model.dynamics "gaussian"   -> "model.dynamics=gaussian" (dotted flatten)
    #   model.encoder "mlp"         -> "model.encoder=mlp"       (dotted flatten)
    #   training.total_env_steps    -> "training.total_env_steps=200000" (nested flatten)
    assert expected_overrides[0] == "+experiment=champion"   # group selectors come first
    assert expected_overrides[1] == "env=walker2d"           # display name -> group file
    assert "seed=3" in cmd
    assert "model.dynamics=gaussian" in cmd
    assert "model.encoder=mlp" in cmd
    assert "training.total_env_steps=200000" in cmd
    # group selectors precede every dotted field override (Hydra applies the group
    # file before the field tweaks)
    assert cmd.index("+experiment=champion") < cmd.index("model.dynamics=gaussian")
    assert cmd.index("env=walker2d") < cmd.index("model.dynamics=gaussian")
    # NO experiment.name field override leaks alongside the +experiment= form.
    assert "experiment.name=champion" not in cmd

    # NO subprocess in dry-run: the run is RECORDED (launched ledger), but the
    # LaunchRegistry never spawned anything.
    assert len(srv.launched) == 1
    assert srv.launched[0]["run_name"] == "champion-Walker2d-v5-s3"
    assert srv.launched[0]["dry_run"] is True
    assert "pid" not in d                       # a real launch would carry a pid
    assert srv.launcher.list() == []            # registry empty == no process spawned

    # The experiment yaml was actually authored under the temp repo's authoring dir.
    assert Path(d["experiment_yaml"]).exists()


def _bad_spectral_spec() -> dict:
    """A spec that TRIPS the spectral house rules two ways: a zero-floor schedule
    AND latent_cap_mult > 1 on the spectral path (ledger 2026-06-07). Either alone
    is a rejection; both make the gate unambiguous.
    """
    return {
        "experiment": {"name": "champ"},
        "env": {"name": "Pendulum-v1"},
        "spectral": {"enabled": True},
        "penalty": {"schedule": {"kind": "cuberoot", "floor": 0.0}},  # floor <= 0 -> zero-touching
        "model": {"latent_cap_mult": 4},                              # > 1 -> over-resolves
    }


def test_partA_bad_spec_rejected_at_boundary_no_launch(tmp_path):
    """A bad spec is REJECTED at the validation gate and never yields a launch.

    Asserts BOTH boundary forms: the pure raise_if_invalid hard-gate, and the
    server's strict_validation gate (accepted=False, no command, empty ledger).
    """
    bad = _bad_spectral_spec()

    # 1) the pure house-rule gate flags it (both rules fire) ...
    warns = sv.validate_spec(bad)
    assert warns, "bad spectral spec must produce house-rule warnings"
    assert any("zero-touching" in w for w in warns)
    assert any("latent_cap_mult" in w for w in warns)
    # ... and raise_if_invalid hard-rejects rather than returning warnings.
    with pytest.raises(sv.SpecValidationError):
        sv.raise_if_invalid(bad)

    # 2) the server in strict mode REJECTS at the boundary: accepted=False, NO
    #    train command authored, and NOTHING recorded in the launch ledger.
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True, strict_validation=True)
    reply = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": bad}, 102))
    d = reply["data"]
    assert d["accepted"] is False
    assert d["warnings"]                         # the gate surfaces WHY it was rejected
    assert "command" not in d                    # no launch command produced
    assert "run_name" not in d
    assert srv.launched == []                    # never recorded -> never launchable
    assert srv.launcher.list() == []             # and certainly never spawned


# ===================== PART B — slow, REAL training smoke =======================

def _tiny_pendulum_spec() -> dict:
    """The smallest/fastest real run that still writes ONE metric row.

    Pendulum-v1 is the cheapest env (configs/env/pendulum.yaml: obs_dim 3). train.py
    logs the first metrics.jsonl row when iteration 1 COMPLETES, and an eval/return
    row lands when `iteration % eval_every_iters == 0` — so eval_every_iters=1 and
    total_env_steps == steps_per_iter == ONE iteration gets us a row with eval keys.
    Every update count is floored to the minimum, num_envs=1, and the expensive
    extras are disabled (video off; auto_dose warmup zeroed) to keep wall-clock low.
    These are training KNOBS only (training.* / logging.* / penalty.auto_dose.*),
    not algo.* selectors, so the spec stays clean past the validator gate.
    """
    return {
        "experiment": {"name": "smoke"},
        "env": {"name": "Pendulum-v1"},
        "seed": 0,
        "training": {
            "total_env_steps": 200,       # exactly one iteration's worth
            "steps_per_iter": 200,        # -> iteration 1 fires once, then the loop ends
            "model_updates_per_iter": 1,  # minimum work per iter
            "behaviour_updates_per_iter": 1,
            "eval_every_iters": 1,        # eval (eval/return) on iteration 1
            "num_envs": 1,                # no AsyncVectorEnv worker pool
        },
        # don't measure an auto-dose over 500 warmup updates (warmup_updates run at
        # lam=0); keep the first iteration cheap.
        "penalty": {"auto_dose": {"enabled": False}},
        # eval rollout video needs GL + extra episodes — off for the smoke run.
        "logging": {"video": {"enabled": False}},
    }


def _poll_metric_row(metrics_path: Path, timeout: float = 180.0,
                     interval: float = 1.0) -> dict | None:
    """Poll metrics.jsonl until the FIRST fully-written JSON row appears.

    Returns the parsed first row, or None if `timeout` elapses. Tolerates the file
    not existing yet and a torn final line (a row mid-write) — only a line that
    json-parses counts.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn write; keep waiting for a complete row
        time.sleep(interval)
    return None


@pytest.mark.slow
def test_partB_real_launch_writes_and_reads_back_a_metric(tmp_path):
    """REAL smoke: submit a tiny Pendulum spec so LaunchRegistry spawns
    scripts/train.py, poll for one metric row, and assert it reads back through the
    server's read_metric. Robust teardown (cancel the child, even on failure).
    """
    # Redirect ALL run output into the temp tree so we never touch mbrl/results:
    #   * the server reads runs from results_dir (== <root>/runs) and the metrics.db
    #     off its parent; checkpoints under checkpoints_dir;
    #   * train.py itself writes to logging.dir (relative to its cwd == repo_root),
    #     so we override logging.dir to the SAME root via the spec.
    results_root = tmp_path / "results"
    runs_dir = results_root / "runs"
    ckpt_dir = tmp_path / "checkpoints"

    srv = sb.StudioBridgeServer(
        repo_root=_REPO,                  # real repo so scripts/train.py + configs/ resolve
        results_dir=runs_dir,
        checkpoints_dir=ckpt_dir,
        experiments_dir=tmp_path / "experiments",
        dry_run=False,                    # REAL launch
    )
    spec = _tiny_pendulum_spec()
    # point train.py's local logger at the temp results root (absolute, so it is
    # cwd-independent); without this the run would write under mbrl/results.
    spec["logging"]["dir"] = str(results_root)

    run_name = s2c.run_name_for_spec(spec, seed=0)   # "smoke-Pendulum-v1-s0"

    reply = srv.dispatch(sb.make(sb.SUBMIT_SPEC, {"model_spec": spec}, 201))
    d = reply["data"]
    try:
        assert d["accepted"] is True, f"launch rejected: {d}"
        assert d["warnings"] == [], f"unexpected house-rule warnings: {d['warnings']}"
        assert d["run_name"] == run_name
        assert "pid" in d and d["pid"] > 0       # a real subprocess was spawned

        metrics_path = runs_dir / run_name / "metrics.jsonl"
        row = _poll_metric_row(metrics_path, timeout=180.0, interval=1.0)

        # If training never produced a row, surface the run's log to make the
        # failure diagnosable (and still fail the assert below).
        if row is None:
            log = srv.dispatch(sb.make(sb.PULL_LOG, {"run": run_name, "since_line": 0}, 202))
            tail = "\n".join(log["data"]["lines"][-40:])
            status = srv.dispatch(sb.make(sb.PULL_RUN_STATUS, {"run": run_name}, 203))["data"]
            pytest.fail(f"no metric row within timeout; run status={status}\n--- log tail ---\n{tail}")

        # train.py always stamps env_steps on every logged row.
        assert "env_steps" in row

        # Pick a numeric key from the row and assert the SERVER reads the same curve
        # back. eval/return is present on the eval iteration; fall back to any
        # numeric key (e.g. loss/total) if the schedule differs.
        key = "eval/return" if "eval/return" in row else next(
            k for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "env_steps")

        out = sb.read_metric(srv.results_dir, run_name, key)
        assert out["run"] == run_name and out["key"] == key
        assert out["steps"], f"read_metric returned no steps for {key}"
        assert out["values"], f"read_metric returned no values for {key}"
        assert len(out["steps"]) == len(out["values"])

        # And the same curve is reachable over the dispatch verb the Godot panel uses.
        pulled = srv.dispatch(sb.make(sb.PULL_METRIC, {"run": run_name, "key": key}, 204))["data"]
        assert pulled["values"] == out["values"]
        assert pulled["steps"] == out["steps"]
    finally:
        # ALWAYS terminate the spawned child, even if an assert above failed, so no
        # orphaned train.py outlives the test. tmp_path is cleaned by pytest.
        try:
            srv.launcher.cancel(run_name)
        except Exception:
            pass
        srv.stop()
