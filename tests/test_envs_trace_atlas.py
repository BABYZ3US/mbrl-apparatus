"""Tests for mbrl.envs.trace_atlas — the proof-spectral reconstruction env."""
import json

import numpy as np

from mbrl.envs.trace_atlas import (TraceAtlasReconstructionEnv, code_vector,
                                   load_corpus, make_trace_atlas_env)


def _corpus(tmp_path, n=5, k=8):
    p = tmp_path / "corpus.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"thm{i}", "tier": i % 3, "merkle": f"m{i}", "godel_code": i,
                "spectral": [float(i + j) for j in range(k)],
            }) + "\n")
        f.write("{ torn line\n")  # must be skipped
    return p


def test_load_corpus_skips_torn_lines(tmp_path):
    rows = load_corpus(_corpus(tmp_path, n=3))
    assert len(rows) == 3 and all("spectral" in r for r in rows)


def test_reset_returns_spectral_obs(tmp_path):
    env = make_trace_atlas_env(_corpus(tmp_path), embed_dim=16, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,) and obs.dtype == np.float32
    assert "id" in info and "godel_code" in info
    assert env.observation_space.contains(obs)


def test_perfect_action_scores_reward_and_correct(tmp_path):
    env = TraceAtlasReconstructionEnv(_corpus(tmp_path), embed_dim=16, seed=1,
                                      correctness_bonus=1.0)
    _, info0 = env.reset(seed=3)
    target = env.codebook[info0["id"]]              # the exact target embedding
    obs, reward, term, trunc, info = env.step(target)
    assert info["correct"] is True
    assert info["cosine"] > 0.999
    assert reward > 1.9                              # cosine(~1) + bonus(1)
    assert term is True and trunc is False


def test_wrong_action_is_lower(tmp_path):
    env = TraceAtlasReconstructionEnv(_corpus(tmp_path, n=8), embed_dim=16, seed=2)
    _, info0 = env.reset(seed=5)
    target = env.codebook[info0["id"]]
    obs, reward, term, trunc, info = env.step(-target)   # anti-aligned
    assert info["cosine"] < 0.0
    assert reward < 1.0                              # no bonus, negative cosine


def test_deterministic_reset(tmp_path):
    env = make_trace_atlas_env(_corpus(tmp_path), seed=0)
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert np.array_equal(a, b)


def test_code_vector_is_unit_and_stable():
    v1, v2 = code_vector("thm7", 32), code_vector("thm7", 32)
    assert np.allclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert np.array_equal(v1, v2)                    # deterministic by id
    assert not np.array_equal(v1, code_vector("thm8", 32))


def test_spaces_shapes(tmp_path):
    env = TraceAtlasReconstructionEnv(_corpus(tmp_path), embed_dim=24)
    assert env.observation_space.shape == (8,)
    assert env.action_space.shape == (24,)
