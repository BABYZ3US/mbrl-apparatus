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

    Rule parity with compile.gd::validate: identical trigger conditions and
    defaults. Only the float rendering inside the messages differs (Python repr
    vs GDScript str) — *which* specs warn is identical, and that behavior is what
    the shared contract pins.
    """
    warns: list[str] = []
    # authorable-but-unwired algo selectors (battle-tested components, integration
    # pending per arm): warn so an authored spec is honest about what trains today.
    algo = _as_dict(spec.get("algo"))
    if str(algo.get("critic", "value")) != "value":
        warns.append("algo.critic '%s' is implemented + tested but not yet consumed "
                     "by the Trainer — the run trains with the default value head"
                     % algo.get("critic"))
    if str(algo.get("actor", "gaussian")) != "gaussian":
        warns.append("algo.actor '%s' is implemented + tested but not yet consumed "
                     "by the Trainer — the run trains with the default Gaussian policy"
                     % algo.get("actor"))
    if str(algo.get("planner", "none")) != "none":
        warns.append("algo.planner '%s' is implemented + tested but not yet consumed "
                     "by the Trainer — actions come from the policy, not MPC"
                     % algo.get("planner"))
    spectral = _as_dict(spec.get("spectral"))
    if not bool(spectral.get("enabled", False)):
        return warns  # non-spectral path: house rules don't apply

    sched = _as_dict(_as_dict(spec.get("penalty")).get("schedule"))
    kind = sched.get("kind", "")
    floor = _to_float(sched.get("floor", 0.0))  # missing -> 0.0 -> warns
    if kind in ZERO_TOUCHING_KINDS or floor <= 0.0:
        warns.append(
            f"spectral path: schedule '{kind}' (floor {floor}) is zero-touching"
            " — use cuberoot with floor > 0 (ledger 2026-06-07)"
        )

    # GDScript: int(spec.model.get("latent_cap_mult", 4)). Missing key -> 4 -> warns;
    # present-but-unparseable -> int() -> 0 -> no warn.
    cap = _to_int(_as_dict(spec.get("model")).get("latent_cap_mult", 4), default=0)
    if cap > 1:
        warns.append(
            f"spectral path: latent_cap_mult {cap} > 1 over-resolves the"
            " closed-form fit — set 1 (ledger 2026-06-07)"
        )
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
