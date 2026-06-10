"""Tests for mbrl.studio.spec_validator — apparatus mirror of compile.gd::validate.

Pure stdlib + pytest. Every rule + the GDScript default/coercion edge cases that
compile.gd (godot_studio/scripts/graph/compile.gd:45-59) relies on.
"""
from __future__ import annotations

import pytest

from mbrl.studio.spec_validator import (
    SpecValidationError,
    raise_if_invalid,
    validate,
    validate_spec,
)


def _spectral(schedule=None, latent_cap_mult=1, enabled=True):
    """A minimal spectral ModelSpec with knobs for the two house rules."""
    spec = {"spectral": {"enabled": enabled}, "model": {"latent_cap_mult": latent_cap_mult}}
    if schedule is not None:
        spec["penalty"] = {"schedule": schedule}
    else:
        spec["penalty"] = {"schedule": {"kind": "cuberoot", "floor": 1e-5}}
    return spec


def test_clean_spectral_spec_has_no_warnings():
    spec = _spectral(schedule={"kind": "cuberoot", "floor": 1e-5}, latent_cap_mult=1)
    assert validate_spec(spec) == []
    assert validate(spec).ok is True


def test_non_spectral_spec_is_exempt_even_when_otherwise_bad():
    # enabled=False -> house rules don't apply, even with a zero-touching sched + wide latent
    spec = _spectral(schedule={"kind": "step", "floor": 0.0}, latent_cap_mult=4, enabled=False)
    assert validate_spec(spec) == []
    # spectral block missing entirely -> also exempt
    assert validate_spec({"penalty": {"schedule": {"kind": "step"}}, "model": {"latent_cap_mult": 8}}) == []


@pytest.mark.parametrize("kind", ["step", "sincos", "sin2chirp"])
def test_zero_touching_kinds_warn(kind):
    warns = validate_spec(_spectral(schedule={"kind": kind, "floor": 1e-5}, latent_cap_mult=1))
    assert len(warns) == 1
    assert "zero-touching" in warns[0]


def test_floor_zero_or_negative_warns():
    assert any("zero-touching" in w for w in
               validate_spec(_spectral(schedule={"kind": "cuberoot", "floor": 0.0}, latent_cap_mult=1)))
    assert any("zero-touching" in w for w in
               validate_spec(_spectral(schedule={"kind": "cuberoot", "floor": -1.0}, latent_cap_mult=1)))


def test_missing_floor_defaults_to_zero_and_warns():
    # GDScript float(sched.get("floor", 0.0)) -> 0.0 -> zero-touching
    warns = validate_spec(_spectral(schedule={"kind": "cuberoot"}, latent_cap_mult=1))
    assert any("zero-touching" in w for w in warns)


def test_string_floor_is_parsed():
    # "1e-5" parses to a positive float -> no zero-touching warn
    assert validate_spec(_spectral(schedule={"kind": "cuberoot", "floor": "1e-5"}, latent_cap_mult=1)) == []


@pytest.mark.parametrize("cap,expect_warn", [(1, False), (2, True), (4, True)])
def test_latent_cap_rule(cap, expect_warn):
    warns = validate_spec(_spectral(schedule={"kind": "cuberoot", "floor": 1e-5}, latent_cap_mult=cap))
    assert any("over-resolves" in w for w in warns) is expect_warn


def test_missing_latent_cap_mult_defaults_to_four_and_warns():
    # GDScript int(model.get("latent_cap_mult", 4)) -> 4 -> warns; spectral specs must set 1
    spec = {"spectral": {"enabled": True},
            "penalty": {"schedule": {"kind": "cuberoot", "floor": 1e-5}}, "model": {}}
    assert any("over-resolves" in w for w in validate_spec(spec))


def test_both_rules_can_fire_together():
    warns = validate_spec(_spectral(schedule={"kind": "step", "floor": 0.0}, latent_cap_mult=4))
    assert len(warns) == 2


def test_raise_if_invalid_raises_then_passes():
    bad = _spectral(schedule={"kind": "step", "floor": 0.0}, latent_cap_mult=4)
    with pytest.raises(SpecValidationError) as ei:
        raise_if_invalid(bad)
    assert ei.value.warnings  # carries the messages
    raise_if_invalid(_spectral())  # clean -> no raise


def test_robust_to_malformed_blocks():
    # non-dict spectral/penalty/model must not crash (GDScript .get(k, {}) parity)
    assert validate_spec({"spectral": None}) == []
    assert validate_spec({"spectral": {"enabled": True}, "penalty": None, "model": None}) != []  # floor->0, cap->4
