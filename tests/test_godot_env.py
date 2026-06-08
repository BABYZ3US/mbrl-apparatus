"""GodotEnv train-seam adapter — driven against a fake Godot env over a real
socket. Proves the apparatus can treat a Godot environment as a gymnasium env
through the bridge protocol (the third backend of the one-adapter seam).
"""
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import studio_bridge_server as sb
from godot_env import GodotEnv


def _fake_godot_env(port: int, n_steps_to_terminate: int = 3) -> None:
    """A Python stand-in for a Godot env: connects to the apparatus, then
    answers env.spec/reset/step requests with a trivial 4-dim env."""
    cli = socket.create_connection(("127.0.0.1", port), timeout=5)
    steps = 0
    with cli:
        while True:
            req = sb.read_frame(cli)
            if req is None:
                break
            t, data, rid = req["type"], req["data"], req["id"]
            if t == sb.ENV_SPEC:
                reply = {"obs_dim": 4, "action_dim": 2,
                         "action_low": [-1.0, -1.0], "action_high": [1.0, 1.0]}
            elif t == sb.ENV_RESET:
                steps = 0
                reply = {"obs": [0.0, 0.0, 0.0, 0.0], "info": {"seed": data["seed"]}}
            elif t == sb.ENV_STEP:
                steps += 1
                a = data["action"]
                reply = {"obs": [a[0], a[1], float(steps), 0.0],
                         "reward": -float(steps),
                         "terminated": steps >= n_steps_to_terminate,
                         "truncated": False, "info": {"steps": steps}}
            else:
                reply = {"error": {"code": "unknown", "message": t}}
            cli.sendall(sb.frame(sb.make(t, reply, rid)))


def test_godot_env_drives_a_fake_env_over_the_bridge():
    env = GodotEnv(port=0)
    t = threading.Thread(target=_fake_godot_env, args=(env.port,), daemon=True)
    t.start()
    try:
        spec = env.wait_for_env()
        assert spec["obs_dim"] == 4 and spec["action_dim"] == 2
        # gymnasium spaces built from the spec
        assert env.observation_space.shape == (4,)
        assert env.action_space.shape == (2,)

        obs, info = env.reset(seed=11)
        assert isinstance(obs, np.ndarray) and obs.shape == (4,)
        assert info["seed"] == 11

        # step until termination; the fake terminates at step 3
        obs, r, term, trunc, info = env.step([0.5, -0.5])
        assert obs[0] == pytest.approx(0.5) and obs[2] == 1.0
        assert r == -1.0 and not term and not trunc
        env.step([0.0, 0.0])
        _, _, term3, _, info3 = env.step([0.0, 0.0])
        assert term3 is True and info3["steps"] == 3
    finally:
        env.close()
        t.join(timeout=5)


def test_godot_env_propagates_env_errors():
    """If the Godot side replies with an error frame, step raises (not silent)."""
    def _erroring(port):
        cli = socket.create_connection(("127.0.0.1", port), timeout=5)
        with cli:
            req = sb.read_frame(cli)   # the env.spec request
            cli.sendall(sb.frame(sb.make(sb.ERROR,
                        {"code": "boom", "message": "no scene"}, req["id"])))

    env = GodotEnv(port=0)
    t = threading.Thread(target=_erroring, args=(env.port,), daemon=True)
    t.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            env.wait_for_env()
    finally:
        env.close()
        t.join(timeout=5)
