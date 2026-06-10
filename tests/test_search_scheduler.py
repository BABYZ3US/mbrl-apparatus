"""W9 scheduler core: deterministic sampling, median-rule stopping semantics,
slot transitions. Pure functions — no launches, no files."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.search import sample_axes, decide_stops, next_actions

AXES = [
    {"path": "optim.model_lr", "kind": "loguniform", "low": 1e-5, "high": 1e-2},
    {"path": "model.latent_dim", "kind": "int", "low": 2, "high": 16},
    {"path": "model.encoder", "kind": "choice", "values": ["mlp", "vae"]},
    {"path": "imagination.pessimism", "kind": "uniform", "low": 0.0, "high": 1.0},
]


def test_sampling_is_seeded_bounded_and_typed():
    a = sample_axes(AXES, 16, seed=7)
    b = sample_axes(AXES, 16, seed=7)
    assert a == b                                        # resume-safe
    assert a != sample_axes(AXES, 16, seed=8)
    for o in a:
        assert 1e-5 <= o["optim.model_lr"] <= 1e-2
        assert 2 <= o["model.latent_dim"] <= 16 and isinstance(o["model.latent_dim"], int)
        assert o["model.encoder"] in ("mlp", "vae")
        assert 0.0 <= o["imagination.pessimism"] <= 1.0


def test_sampling_rejections():
    with pytest.raises(ValueError, match="unknown axis kind"):
        sample_axes([{"path": "x", "kind": "normal"}], 2)
    with pytest.raises(ValueError, match="low > 0"):
        sample_axes([{"path": "x", "kind": "loguniform", "low": 0, "high": 1}], 2)
    with pytest.raises(ValueError, match="no values"):
        sample_axes([{"path": "x", "kind": "choice", "values": []}], 2)
    with pytest.raises(ValueError, match="positive"):
        sample_axes(AXES, 0)


def test_median_rule_stops_strict_losers_only():
    histories = {
        "a": [(100, 1.0), (200, 2.0), (300, 3.0)],       # winner
        "b": [(100, 0.9), (200, 1.8), (300, 2.7)],       # median-ish
        "c": [(100, 0.1), (200, 0.2), (300, 0.3)],       # loser
    }
    stops = decide_stops(histories, mode="max")
    assert stops == {"c"}                                # b == median survives
    # min mode flips the verdict direction
    assert decide_stops(histories, mode="min") == {"a"}


def test_median_rule_is_conservative_early():
    # too few arms reporting -> no verdicts
    assert decide_stops({"a": [(1, 1), (2, 2), (3, 3)]}) == set()
    # too few points per arm -> arm not eligible, leaving < min_arms
    histories = {"a": [(1, 1)], "b": [(1, 2)], "c": [(1, 0)]}
    assert decide_stops(histories, min_points=3) == set()


def test_median_rule_compares_at_common_step():
    # 'slow' has only reached step 100 — everyone is compared THERE, so a fast
    # arm's later points don't unfairly beat slow's early ones
    histories = {
        "fast": [(100, 1.0), (200, 9.0), (300, 9.9)],
        "slow": [(100, 2.0), (101, 2.1), (102, 2.2)],
        "mid":  [(100, 1.5), (200, 5.0), (300, 6.0)],
    }
    stops = decide_stops(histories, mode="max")
    assert stops == {"fast"}                             # at step ~102: 1.0 < median 1.5


def test_transitions_fill_slots_and_finish():
    state = {"parallel": 2, "arms": [
        {"name": "s1", "status": "running"},
        {"name": "s2", "status": "running"},
        {"name": "s3", "status": "queued"},
        {"name": "s4", "status": "queued"},
    ]}
    # stop one runner -> exactly one slot frees -> one launch
    acts = next_actions(state, stops={"s1"})
    assert acts["stop"] == ["s1"] and acts["launch"] == ["s3"]
    assert acts["done"] is False
    # everything finished/stopped -> done
    done_state = {"parallel": 2, "arms": [
        {"name": "s1", "status": "stopped"},
        {"name": "s2", "status": "finished"},
    ]}
    acts2 = next_actions(done_state)
    assert acts2 == {"launch": [], "stop": [], "done": True}
