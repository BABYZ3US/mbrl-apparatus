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


def test_trace_atlas_registered_and_dims_match_yaml(tmp_path):
    """HERMETIC (CI run 4 fix): the real corpus lives OUTSIDE this repo
    (../trace-atlas — present on dev machines, absent on a fresh runner), and
    the first CI run failed on exactly that. A 3-row synthetic corpus with the
    yaml's spectral width proves registration + dims everywhere."""
    import json

    import gymnasium as gym
    import mbrl.envs  # noqa: F401 — registration side effect

    assert "TraceAtlas-v0" in gym.registry
    doc = yaml.safe_load((_REPO / "configs" / "env" / "trace_atlas.yaml").read_text())
    k = int(doc["obs_dim"])
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(
        json.dumps({"id": f"thm{i}", "spectral": [0.1 * (i + j) for j in range(k)]})
        for i in range(3)))
    env = gym.make("TraceAtlas-v0", corpus_path=str(corpus))
    assert env.observation_space.shape[0] == k
    assert env.action_space.shape[0] == int(doc["action_dim"])
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == k
    env.close()


def test_trace_atlas_real_corpus_when_present():
    """Opportunistic: on machines that DO carry ../trace-atlas, also prove the
    default corpus path loads (skipped honestly elsewhere)."""
    import pytest

    from mbrl.envs.trace_atlas import DEFAULT_CORPUS

    if not Path(DEFAULT_CORPUS).exists():
        pytest.skip(f"real corpus absent ({DEFAULT_CORPUS}) — dev-machine-only check")
    import gymnasium as gym
    import mbrl.envs  # noqa: F401

    env = gym.make("TraceAtlas-v0")
    assert env.reset(seed=0)[0].shape[0] == env.observation_space.shape[0]
    env.close()
