"""Tests for mbrl.studio.sweep — SweepSpec expansion + validation (v0.1 M3).

Pure stdlib + pytest. Verifies cartesian expansion, unique arm identity, the
validator gate propagating per-arm, and dotted-path merging.
"""
from __future__ import annotations

from mbrl.studio.sweep import Arm, expand_sweep, plan_sweep

BASE = {"experiment": {"name": "champ"}, "env": {"name": "Pendulum-v1"}}


def test_cartesian_product_times_seeds():
    arms = expand_sweep(
        BASE,
        axes=[{"path": "penalty.lambda", "values": [1e-4, 1e-3]},
              {"path": "lr", "values": [3e-4, 1e-3]}],
        seeds=[0, 1])
    assert len(arms) == 2 * 2 * 2  # 2 lambda x 2 lr x 2 seeds


def test_axis_values_land_in_spec_via_dotted_path():
    arms = expand_sweep(BASE, axes=[{"path": "penalty.lambda", "values": [0.01]}], seeds=[0])
    spec = arms[0].spec
    assert spec["penalty"]["lambda"] == 0.01
    assert spec["seed"] == 0
    assert spec["env"]["name"] == "Pendulum-v1"  # base preserved


def test_arms_have_unique_run_names_and_distinct_groups():
    arms = expand_sweep(BASE, axes=[{"path": "penalty.lambda", "values": [1e-4, 1e-3]}], seeds=[0, 1])
    names = [a.run_name for a in arms]
    assert len(set(names)) == len(names) == 4
    # same lambda, different seed -> same experiment.name (groups together)
    by_exp = {a.spec["experiment"]["name"] for a in arms}
    assert len(by_exp) == 2  # two arms, distinct experiment names; seeds share


def test_base_spec_is_not_mutated():
    before = BASE["experiment"]["name"]
    expand_sweep(BASE, axes=[{"path": "penalty.lambda", "values": [0.01]}], seeds=[0])
    assert BASE["experiment"]["name"] == before  # deep-copied per arm
    assert "lambda" not in BASE.get("penalty", {})


def test_no_axes_still_fans_over_seeds():
    arms = expand_sweep(BASE, axes=[], seeds=[0, 1, 2])
    assert len(arms) == 3
    assert {a.label for a in arms} == {"base"}


def test_seeds_default_to_single_zero():
    arms = expand_sweep(BASE, axes=[], seeds=None)
    assert [a.seed for a in arms] == [0]


def test_overrides_include_experiment_seed_and_axis():
    arms = expand_sweep(BASE, axes=[{"path": "penalty.lambda", "values": [0.01]}], seeds=[7])
    ov = arms[0].overrides
    assert any(o.startswith("+experiment=champ") for o in ov)  # group selector first
    assert "seed=7" in ov
    assert "penalty.lambda=0.01" in ov


def test_validator_gate_propagates_per_arm():
    # spectral base with the wrong cap -> EVERY arm warns; a clean cap -> none warn
    bad = {"experiment": {"name": "spec"}, "env": {"name": "Pendulum-v1"},
           "spectral": {"enabled": True},
           "penalty": {"schedule": {"kind": "cuberoot", "floor": 1e-5}},
           "model": {"latent_cap_mult": 4}}
    plan = plan_sweep(bad, axes=[{"path": "penalty.lambda", "values": [1e-3, 1e-2]}], seeds=[0])
    assert plan.n == 2
    assert plan.ok is False
    assert all(not a.ok for a in plan.arms)

    bad["model"]["latent_cap_mult"] = 1
    plan2 = plan_sweep(bad, axes=[{"path": "penalty.lambda", "values": [1e-3, 1e-2]}], seeds=[0])
    assert plan2.ok is True


def test_plan_as_dict_is_json_shaped():
    plan = plan_sweep(BASE, axes=[{"path": "lr", "values": [3e-4]}], seeds=[0])
    d = plan.as_dict()
    assert d["n"] == 1 and d["group"] == "champ"
    assert d["arms"][0]["run_name"] == plan.arms[0].run_name
    assert isinstance(d["arms"][0]["overrides"], list)
