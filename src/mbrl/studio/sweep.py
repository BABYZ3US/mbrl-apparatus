"""sweep — expand a Studio SweepSpec into validated, launchable arms.

A SweepSpec (the Bridge's submit.sweep payload, see
godot_studio/test/fixtures/v0_1_protocol_contract.json) is::

    {base_spec: {...}, axes: [{path: "penalty.lambda", values: [...]}, ...], seeds: [int]}

This module takes the cartesian product over the axes' values, crosses it with the
seeds, and produces one Arm per (combo, seed): a fully-merged ModelSpec, a UNIQUE
``experiment.name`` (so results/runs + W&B grouping stay distinct — seeds within an
arm form one mean±CI band, arms sit side by side), the Hydra overrides train.py
wants (mbrl.studio.spec_to_config), and the spectral house-rule warnings
(mbrl.studio.spec_validator).

The server's submit.sweep handler and the CLI (scripts/studio_sweep.py) both call
plan_sweep(). Pure: stdlib + the two sibling studio modules — inside the seal
(docs/remote_execution.md §1), no torch.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from .spec_to_config import run_name_for_spec, spec_to_overrides
from .spec_validator import validate_spec

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)  # 0.001 -> "0.001", 1e-05 -> "1e-05" (stable, compact)
    return str(v)


def _safe(s: str) -> str:
    return _UNSAFE.sub("_", s).strip("_") or "x"


def _deep_copy(obj):
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def _set_path(spec: dict, path: str, value) -> None:
    """Set a dotted path, creating intermediate dicts (Hydra dotted-key semantics)."""
    keys = path.split(".")
    node = spec
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


@dataclass(frozen=True)
class Arm:
    label: str          # human: "penalty.lambda=0.001,lr=0.0003"
    spec: dict          # fully-merged ModelSpec for this arm
    axes: dict          # {path: chosen_value}
    seed: int
    run_name: str       # "<group>__<arm>-<env>-s<seed>"
    overrides: list     # Hydra overrides (spec_to_overrides)
    warnings: list      # spec_validator warnings ([] if clean)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def as_dict(self) -> dict:
        return {"label": self.label, "axes": self.axes, "seed": self.seed,
                "run_name": self.run_name, "overrides": self.overrides,
                "warnings": self.warnings}


@dataclass(frozen=True)
class SweepPlan:
    group: str
    arms: list  # list[Arm]

    @property
    def n(self) -> int:
        return len(self.arms)

    @property
    def ok(self) -> bool:
        return all(a.ok for a in self.arms)

    def as_dict(self) -> dict:
        return {"group": self.group, "n": self.n, "ok": self.ok,
                "arms": [a.as_dict() for a in self.arms]}


def _group_of(base_spec: dict, group: str | None) -> str:
    if group:
        return _safe(group)
    exp = base_spec.get("experiment")
    if isinstance(exp, dict) and exp.get("name"):
        return _safe(str(exp["name"]))
    if isinstance(exp, str) and exp:
        return _safe(exp)
    return "sweep"


def expand_sweep(base_spec: dict, axes, seeds, group: str | None = None) -> list:
    """Cartesian product over the axes' values × seeds -> list[Arm].

    Each arm gets a unique ``experiment.name`` (``<group>__<axis-leaf>-<value>...``)
    so its run_name (and the seed-stripped W&B group) is distinct; seeds within an
    arm share the experiment.name and aggregate. A no-axes sweep still fans over
    seeds (one base arm per seed).
    """
    if not isinstance(base_spec, dict):
        raise TypeError("base_spec must be a dict")
    axes = list(axes or [])
    seeds = [int(s) for s in (seeds or [0])]
    grp = _group_of(base_spec, group)

    paths = [str(a["path"]) for a in axes]
    value_lists = [list(a["values"]) for a in axes]

    arms: list[Arm] = []
    # itertools.product() with no args yields a single empty tuple -> base arm.
    for combo in itertools.product(*value_lists):
        chosen = dict(zip(paths, combo))
        label = ",".join(f"{p}={_fmt(v)}" for p, v in chosen.items()) or "base"
        token = "__".join(f"{_leaf(p)}-{_safe(_fmt(v))}" for p, v in chosen.items())
        exp_name = grp if not token else f"{grp}__{token}"
        for seed in seeds:
            spec = _deep_copy(base_spec)
            _set_path(spec, "experiment.name", exp_name)
            for p, v in chosen.items():
                _set_path(spec, p, v)
            _set_path(spec, "seed", seed)
            arms.append(Arm(
                label=label, spec=spec, axes=chosen, seed=seed,
                run_name=run_name_for_spec(spec, seed=seed),
                overrides=spec_to_overrides(spec),
                warnings=validate_spec(spec),
            ))
    return arms


def plan_sweep(base_spec: dict, axes, seeds, group: str | None = None) -> SweepPlan:
    """Full plan: group id + every validated Arm. Backs submit.sweep's reply."""
    grp = _group_of(base_spec, group)
    return SweepPlan(group=grp, arms=expand_sweep(base_spec, axes, seeds, group=grp))
