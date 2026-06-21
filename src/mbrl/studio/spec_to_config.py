"""ModelSpec (Godot node graph) -> Hydra config, the spec->config seam.

The Studio's components each contribute a Hydra-config fragment via to_spec()
(godot_studio/scripts/components/*.gd); compile.gd deep-merges them into ONE
nested dict — the ModelSpec — and ships it over the Bridge as submit.spec's
{"model_spec": {...}}. This module is the apparatus-side translation of that
dict into something scripts/train.py can run:

  * spec_to_overrides(spec)        -> ["model.dynamics=gaussian", "env=walker2d",
                                       "+experiment=champion", ...]
  * write_experiment_yaml(spec,..) -> a "# @package _global_" yaml capturing the
                                       whole spec (so a run is reproducible from a
                                       single config file, not just CLI flags).

Mapping rules (mirroring the fragments emitted in components/*.gd):
  - Nested dicts flatten to dotted keys:  {"model":{"dynamics":"gaussian"}}
        -> "model.dynamics=gaussian"
  - Lists become Hydra inline lists:      {"poly":{"degrees":[1,3]}}
        -> "poly.degrees=[1,3]"   (bracketed, comma-joined, NO spaces)
  - Bools lower-case:                     True -> "true", False -> "false"
  - A top-level {"experiment":{"name":"champion", ...}} pulls `name` into the
    Hydra group-add form  "+experiment=champion"  (matches train.py's
    `+experiment=champion`); any OTHER experiment.* keys still flatten so the
    graph can override a field the experiment file sets.
  - A top-level {"env":{"name":"Walker2d-v5", ...}} pulls `name` into the config
    GROUP form  "env=walker2d"  (env configs are selected by group filename, not
    by their `name:` value — see configs/env/*.yaml). Display names like
    "Walker2d-v5" map to the "walker2d" group file via ENV_GROUP_BY_NAME; an
    unknown name falls back to a lower-cased, suffix-stripped slug. Other env.*
    keys flatten.

Stdlib + pyyaml only. No torch / no mbrl.training import — this stays inside the
seal so scripts/studio_bridge_server.py can import it without dragging in the
training stack (docs/remote_execution.md §1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---- env display-name -> config-group filename (configs/env/<group>.yaml) ----
# The env defaults list selects by GROUP (the file stem), e.g. `env=walker2d`,
# not by the `name:` value inside the file ("Walker2d-v5"). Keep this aligned
# with configs/env/*.yaml. Unknown names fall back to _slug() below.
ENV_GROUP_BY_NAME: dict[str, str] = {
    "Pendulum-v1": "pendulum",
    "HalfCheetah-v5": "halfcheetah",
    "Walker2d-v5": "walker2d",
    "Ant-v5": "ant",
    "Humanoid-v5": "humanoid",
    "halfcheetah_vel": "halfcheetah_vel",
    "pendulum_target": "pendulum_target",
}


def _slug(name: str) -> str:
    """Best-effort env display-name -> group slug for names not in the table.

    "Walker2d-v5" -> "walker2d"; "HalfCheetah-v5" -> "halfcheetah". Lower-case,
    drop a trailing -v<N> version suffix, strip stray separators.
    """
    s = str(name).strip().lower()
    # drop a trailing version tag like "-v5" / "_v1"
    for sep in ("-v", "_v"):
        idx = s.rfind(sep)
        if idx != -1 and s[idx + len(sep):].isdigit():
            s = s[:idx]
            break
    return s.strip("-_")


def env_group(name: str) -> str:
    """Map an env display-name to its Hydra config-group filename."""
    return ENV_GROUP_BY_NAME.get(str(name), _slug(name))


def _scalar(value: Any) -> str:
    """Render a scalar the way Hydra's CLI parser wants it.

    bools lower-case; ints/floats via str(); everything else as-is. (Floats keep
    Python repr — OmegaConf parses "0.001" and "1e-3" alike.)
    """
    if isinstance(value, bool):  # MUST precede int — bool is a subclass of int
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _render_list(values: list) -> str:
    """A Hydra inline list: [a,b,c] — bracketed, comma-joined, no spaces.

    Nested lists are rendered recursively. Scalars use _scalar so bools stay
    lower-case inside lists too.
    """
    parts = []
    for v in values:
        if isinstance(v, (list, tuple)):
            parts.append(_render_list(list(v)))
        else:
            parts.append(_scalar(v))
    return "[" + ",".join(parts) + "]"


def _flatten(prefix: str, obj: Any, out: list[str]) -> None:
    """Walk a nested dict, appending "dotted.key=value" overrides to `out`."""
    if isinstance(obj, dict):
        for k in obj:
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, obj[k], out)
    elif isinstance(obj, (list, tuple)):
        out.append(f"{prefix}={_render_list(list(obj))}")
    else:
        out.append(f"{prefix}={_scalar(obj)}")


def spec_to_overrides(spec: dict) -> list[str]:
    """Flatten a merged ModelSpec into a list of Hydra dotted overrides.

    The `experiment` and `env` blocks are special-cased into the group-select
    forms train.py expects (`+experiment=<name>`, `env=<group>`); everything
    else flattens to dotted `key=value`. Group selectors are emitted FIRST so
    Hydra applies the group file before the field overrides that tweak it.

    Pure: no I/O, deterministic order (group selectors, then a sorted flatten).
    """
    if not isinstance(spec, dict):
        raise TypeError(f"spec must be a dict, got {type(spec).__name__}")

    group_overrides: list[str] = []
    field_overrides: list[str] = []
    work = dict(spec)  # shallow copy so we can pop the special blocks

    # --- experiment: pull `name` into +experiment=<name>, flatten the rest ---
    exp = work.pop("experiment", None)
    if isinstance(exp, dict):
        exp = dict(exp)
        name = exp.pop("name", None)
        if name is not None:
            group_overrides.append(f"+experiment={name}")
        if exp:  # any remaining experiment.* fields
            _flatten("experiment", exp, field_overrides)
    elif exp is not None:
        # a bare string/scalar under "experiment" — treat as the group name
        group_overrides.append(f"+experiment={exp}")

    # --- env: pull `name` into env=<group>, flatten the rest ---
    env = work.pop("env", None)
    if isinstance(env, dict):
        env = dict(env)
        name = env.pop("name", None)
        if name is not None:
            group_overrides.append(f"env={env_group(name)}")
        if env:  # remaining env.* fields (obs_dim, action_scale, ...)
            _flatten("env", env, field_overrides)
    elif env is not None:
        group_overrides.append(f"env={env_group(env)}")

    # --- seed: a common top-level scalar; keep it a plain override ---
    # (left in `work`, flattened below — no special form needed)

    # --- everything else flattens to dotted overrides ---
    for k in work:
        _flatten(str(k), work[k], field_overrides)

    return group_overrides + sorted(field_overrides)


def run_name_for_spec(spec: dict, seed: int | None = None) -> str:
    """Reconstruct the run name train.py will use: "<exp>-<env>-s<seed>".

    Mirrors scripts/train.py's `f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}"`
    and the results/runs/<name>/ layout that pull.runs scans. Falls back to "dev"
    / "Pendulum-v1" / seed 0 to match configs/base.yaml's defaults when the spec
    omits them.
    """
    exp = "dev"
    env_name = "Pendulum-v1"
    if isinstance(spec.get("experiment"), dict):
        exp = str(spec["experiment"].get("name", exp))
    elif isinstance(spec.get("experiment"), str):
        exp = spec["experiment"]
    if isinstance(spec.get("env"), dict):
        env_name = str(spec["env"].get("name", env_name))
    elif isinstance(spec.get("env"), str):
        env_name = spec["env"]
    if seed is None:
        seed = int(spec.get("seed", 0)) if str(spec.get("seed", "")).lstrip("-").isdigit() else 0
    return f"{exp}-{env_name}-s{seed}"


def write_experiment_yaml(spec: dict, out_dir: Path, name: str) -> Path:
    """Write the merged spec as a `# @package _global_` experiment yaml.

    Returns the path written: <out_dir>/experiment/<name>.yaml. The leading
    "# @package _global_" directive makes Hydra splice the file's keys at the
    config ROOT (the same convention configs/experiment/champion.yaml uses), so
    `+experiment=<name>` with out_dir on the search path reproduces the graph.

    The `experiment.name` is stamped into the body if absent so the file is
    self-describing (train.py reads cfg.experiment.name for the run id).
    """
    out_dir = Path(out_dir)
    # Hydra resolves `+experiment=<name>` against the `experiment` config GROUP —
    # i.e. `<search-path-root>/experiment/<name>.yaml` — so the file must live under an
    # `experiment/` subdir of out_dir (and out_dir, the group's parent, is what goes on
    # hydra.searchpath). Writing it flat at out_dir/<name>.yaml left the group
    # unresolvable and real launches failed with "experiment '<name>' not found".
    grp_dir = out_dir / "experiment"
    grp_dir.mkdir(parents=True, exist_ok=True)

    body: dict = _deepcopy_jsonish(spec)
    exp = body.get("experiment")
    if isinstance(exp, dict):
        exp.setdefault("name", name)
    elif exp is None:
        body["experiment"] = {"name": name}

    text = (
        "# @package _global_\n"
        f"# Authored by the MBRL Studio (submit.spec) — experiment '{name}'.\n"
        "# Splices at the config root like configs/experiment/champion.yaml.\n"
        + yaml.safe_dump(body, sort_keys=True, default_flow_style=False)
    )
    path = grp_dir / f"{name}.yaml"
    path.write_text(text)
    return path


def _deepcopy_jsonish(obj: Any) -> Any:
    """Cheap deep copy for JSON-ish spec dicts (dict/list/scalars only)."""
    if isinstance(obj, dict):
        return {k: _deepcopy_jsonish(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deepcopy_jsonish(v) for v in obj]
    return obj
