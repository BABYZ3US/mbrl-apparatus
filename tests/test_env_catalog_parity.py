"""W2 gap-fill: every curated env group parses with load-bearing dims, and the
in-house TraceAtlas-v0 is gym-registered with spaces matching its yaml."""
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

GROUPS = ["pendulum", "halfcheetah", "walker2d", "ant", "humanoid",
          "hopper", "swimmer", "reacher", "trace_atlas",
          "halfcheetah_vel", "pendulum_target"]


def test_every_env_group_parses_with_dims():
    for g in GROUPS:
        doc = yaml.safe_load((_REPO / "configs" / "env" / f"{g}.yaml").read_text())
        assert doc.get("name"), g
        assert int(doc["obs_dim"]) > 0 and int(doc["action_dim"]) > 0, g


def test_trace_atlas_registered_and_dims_match_yaml():
    import gymnasium as gym
    import mbrl.envs  # noqa: F401 — registration side effect

    assert "TraceAtlas-v0" in gym.registry
    env = gym.make("TraceAtlas-v0")
    doc = yaml.safe_load((_REPO / "configs" / "env" / "trace_atlas.yaml").read_text())
    assert env.observation_space.shape[0] == int(doc["obs_dim"])
    assert env.action_space.shape[0] == int(doc["action_dim"])
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == int(doc["obs_dim"])
    env.close()
