"""spec_validator — apparatus-side mirror of the Studio's compile.gd::validate.

The Studio validates the spectral "house rules" at author time
(godot_studio/scripts/graph/compile.gd::validate). This module re-implements the
SAME rules on the apparatus side so scripts/studio_bridge_server.py can check a
ModelSpec the moment it arrives over submit.spec / submit.sweep — *before* a bad
config is ever launched:

  * a spectral path with a zero-touching schedule (step/sincos/sin2chirp, or
    floor <= 0) collapses to an unregularized interpolator — the closed-form
    refit has no inertia, so lambda ~ 0 means instant overfit;
  * a latent wider than 1x obs_dim over-resolves the closed-form fit.

Both are ledger findings (2026-06-07). Keep this in LOCK-STEP with
compile.gd::validate (godot_studio/scripts/graph/compile.gd:45-59) — the GDScript
defaults are load-bearing and mirrored exactly (a missing `floor` defaults to 0
and therefore warns; a missing `latent_cap_mult` defaults to 4 and therefore
warns — spectral specs must set both explicitly).

Pure stdlib — safe to import inside the boundary (docs/remote_execution.md §1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ._spine_validate import validate as _generated_validate

# Schedule kinds that touch (or cross) zero — forbidden on the spectral path.
# Mirror of compile.gd:52.
ZERO_TOUCHING_KINDS = ("step", "sincos", "sin2chirp")


def _as_dict(value) -> dict:
    """A nested block, or {} if absent/wrong-typed (GDScript .get(k, {}) parity)."""
    return value if isinstance(value, dict) else {}


def _to_float(value, default: float = 0.0) -> float:
    # Mirror GDScript float(): unparseable / null -> 0.0, never raises. bool is an
    # int subclass but is not a numeric config value here -> default.
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    # Mirror GDScript int(): truncates floats; unparseable / null -> default (0).
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class SpecValidationError(ValueError):
    """Raised by raise_if_invalid when any house-rule warning fires."""

    def __init__(self, warnings: list[str]):
        self.warnings = list(warnings)
        super().__init__("; ".join(warnings))


@dataclass(frozen=True)
class ValidationResult:
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def as_dict(self) -> dict:
        return {"ok": self.ok, "warnings": list(self.warnings)}


def validate_spec(spec: dict) -> list[str]:
    """Return the spectral house-rule warnings for one ModelSpec ([] if clean).

    Delegates to the vendored SSOT validator (``_spine_validate.validate``,
    re-synced from ``spine/generated/python/validate.py``) so the spectral house
    rules have ONE source — the Haskell spine — instead of a hand-maintained
    Python copy. The generated validator is parity-proven against this module's
    former hand-written logic (11 cases); rule parity with compile.gd::validate is
    therefore preserved transitively through the spine.

    The torch-backed encoder_net chain-rank check is NOT part of the stdlib SSOT
    validator, so it is layered on here for a custom encoder with a wired net
    (ImportError-guarded -> skipped inside the seal where torch is absent, exactly
    as before).
    """
    warns = list(_generated_validate(spec))
    model = _as_dict(spec.get("model"))
    if str(model.get("encoder", "")) == "custom":
        net = [dict(layer) for layer in (model.get("encoder_net", []) or [])]
        if net:
            try:
                from ..models.net_builder import check_net_ranks
            except ImportError:
                pass
            else:
                warns.extend("encoder_net: " + e for e in check_net_ranks(net, 1))
    return warns


def validate(spec: dict) -> ValidationResult:
    """Structured form of validate_spec for callers that want .ok / .as_dict()."""
    return ValidationResult(warnings=validate_spec(spec))


def raise_if_invalid(spec: dict) -> None:
    """Hard-gate a spec: raise SpecValidationError if any house-rule warning fires.

    submit.spec / submit.sweep call this to REJECT a bad config at the boundary
    instead of launching it. (The Studio surfaces the same strings as soft
    warnings at author time; the server chooses the policy — warn vs reject.)
    """
    warns = validate_spec(spec)
    if warns:
        raise SpecValidationError(warns)


def spec_completeness(spec: dict) -> list[str]:
    """The spec-level SHADOW of the graph's minimal-trainable check
    (godot_studio/docs/graph_ports.md): key presence over the compiled
    ModelSpec. It cannot see wires — the graph-side Compile.completeness is
    authoritative for wiring; this catches specs assembled WITHOUT the graph
    (CLI, tests) that forgot a required block. [] = complete.
    """
    msgs: list[str] = []
    env = _as_dict(spec.get("env"))
    if not str(env.get("name", "")):
        msgs.append("missing env.name (the run block)")
    model = _as_dict(spec.get("model"))
    if not str(model.get("encoder", "")):
        msgs.append("missing model.encoder")
    if not str(model.get("dynamics", "")):
        msgs.append("missing model.dynamics")
    spectral = _as_dict(spec.get("spectral"))
    if not bool(spectral.get("enabled", False)) and not str(model.get("reward", "")):
        msgs.append("missing a reward head (no spectral block, no model.reward)")
    return msgs
