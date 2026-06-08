"""GodotEnv — the apparatus-side of the train seam, as a gymnasium.Env.

The THIRD backend of the one-adapter interface (pipeline.md §2, remote_execution
§4): the Trainer drives reset/step exactly as it does gymnasium or pufferlib.
Here the env lives in Godot and is reached over the studio bridge protocol
(godot_studio/addons/mbrl_bridge/protocol.gd) — same little-endian-framed JSON
the viz server speaks, same one socket.

DIRECTIONALITY (architecture §1): the Godot Bridge connects OUT (client); the
apparatus listens. So this adapter opens a listening socket, waits for a Godot
env to connect, then SENDS env.spec / env.reset / env.step requests and reads
replies — the apparatus is the authority, Godot serves env semantics.

Framing helpers are reused from studio_bridge_server (the single wire-format
source on the Python side). gymnasium is a core dep, so importing it here is
within the seal; this is training-side, not the stdlib viz server.
"""
from __future__ import annotations

import socket
from pathlib import Path
import sys

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gym is a core dep, but keep import-safe
    gym = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio_bridge_server as sb   # frame/read_frame/make + message types

_Base = gym.Env if gym is not None else object


class GodotEnv(_Base):
    """Drive a Godot environment over the studio bridge as a gymnasium env."""

    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 9009,
                 connect_timeout: float = 30.0):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.settimeout(connect_timeout)
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._conn: socket.socket | None = None
        self._id = 0
        self._spec: dict | None = None

    # ---- connection ----
    def wait_for_env(self) -> dict:
        """Accept the Godot env's connection and fetch its spec. Returns the
        spec dict; also builds observation_space / action_space."""
        self._conn, _ = self._srv.accept()
        self._spec = self._request(sb.ENV_SPEC, {})
        if gym is not None:
            obs_dim = int(self._spec["obs_dim"])
            act_dim = int(self._spec["action_dim"])
            low = np.array(self._spec.get("action_low", [-1.0] * act_dim), np.float32)
            high = np.array(self._spec.get("action_high", [1.0] * act_dim), np.float32)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (obs_dim,), np.float32)
            self.action_space = spaces.Box(low, high, (act_dim,), np.float32)
        return self._spec

    def _request(self, type_: str, data: dict) -> dict:
        assert self._conn is not None, "no Godot env connected; call wait_for_env()"
        self._id += 1
        self._conn.sendall(sb.frame(sb.make(type_, data, self._id)))
        reply = sb.read_frame(self._conn)
        if reply is None:
            raise ConnectionError("Godot env closed the connection")
        if reply.get("type") == sb.ERROR:
            raise RuntimeError(f"Godot env error: {reply['data']}")
        return reply["data"]

    # ---- gymnasium API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        data = self._request(sb.ENV_RESET, {"seed": int(seed or 0),
                                            "options": options or {}})
        obs = np.asarray(data["obs"], dtype=np.float32)
        return obs, data.get("info", {})

    def step(self, action):
        act = np.asarray(action, dtype=np.float32).tolist()
        data = self._request(sb.ENV_STEP, {"action": act})
        obs = np.asarray(data["obs"], dtype=np.float32)
        return (obs, float(data["reward"]), bool(data["terminated"]),
                bool(data["truncated"]), data.get("info", {}))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._srv.close()
