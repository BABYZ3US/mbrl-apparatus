"""Tests for mbrl.studio.launch.LaunchRegistry — the launch/monitor seam.

Uses tiny real subprocesses (python -c ...), no training. Stdlib + pytest.
"""
import sys
import time

from mbrl.studio.launch import LaunchRegistry


def _wait_finished(reg, run, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if reg.status(run)["state"] != "running":
            return reg.status(run)
        time.sleep(0.02)
    return reg.status(run)


def test_launch_then_finished_with_zero_exit(tmp_path):
    reg = LaunchRegistry(tmp_path / "logs")
    info = reg.launch("ok-s0", [sys.executable, "-c", "print('hello world')"], tmp_path)
    assert info["pid"] > 0 and info["log_path"].endswith("ok-s0.log")
    st = _wait_finished(reg, "ok-s0")
    assert st["state"] == "finished" and st["exit_code"] == 0


def test_failed_run_reports_failed(tmp_path):
    reg = LaunchRegistry(tmp_path / "logs")
    reg.launch("bad-s0", [sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path)
    st = _wait_finished(reg, "bad-s0")
    assert st["state"] == "failed" and st["exit_code"] == 3


def test_tail_captures_stdout(tmp_path):
    reg = LaunchRegistry(tmp_path / "logs")
    reg.launch("log-s0", [sys.executable, "-c", "print('line A'); print('line B')"], tmp_path)
    _wait_finished(reg, "log-s0")
    t = reg.tail("log-s0", since_line=0)
    assert "line A" in t["lines"] and "line B" in t["lines"]
    # incremental: from the end yields nothing new
    assert reg.tail("log-s0", since_line=t["next_line"])["lines"] == []


def test_cancel_running_child(tmp_path):
    reg = LaunchRegistry(tmp_path / "logs")
    reg.launch("sleep-s0", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
    assert reg.status("sleep-s0")["state"] == "running"
    out = reg.cancel("sleep-s0")
    assert out["cancelled"] is True
    assert reg.status("sleep-s0")["state"] in ("failed", "finished")


def test_unknown_run_is_state_unknown(tmp_path):
    reg = LaunchRegistry(tmp_path / "logs")
    assert reg.status("ghost")["state"] == "unknown"
    assert reg.cancel("ghost") == {"run_name": "ghost", "cancelled": False, "state": "unknown"}
    assert reg.tail("ghost")["lines"] == []


def test_list_reports_all_launches(tmp_path):
    reg = LaunchRegistry(tmp_path / "logs")
    reg.launch("a-s0", [sys.executable, "-c", "pass"], tmp_path)
    reg.launch("b-s0", [sys.executable, "-c", "pass"], tmp_path)
    names = {r["run_name"] for r in reg.list()}
    assert names == {"a-s0", "b-s0"}
