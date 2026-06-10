"""mbrl.studio — the apparatus side of the Godot MBRL Studio seam.

This package is the Python counterpart to godot_studio/addons/mbrl_bridge. It
holds the PURE, import-light logic that the one-boundary TCP server
(scripts/studio_bridge_server.py) leans on: turning a ModelSpec authored in the
Studio's node graph into the Hydra overrides / experiment yaml that
scripts/train.py consumes.

Deliberately stdlib + pyyaml only (pyyaml is a core dep). It imports NOTHING
from mbrl.training / torch — the seal (docs/remote_execution.md §1) keeps
training code out of the boundary so the server stays a thin launcher that
spawns train.py as a subprocess rather than importing it.
"""
from __future__ import annotations

from .spec_to_config import (
    spec_to_overrides,
    write_experiment_yaml,
    run_name_for_spec,
    ENV_GROUP_BY_NAME,
)

__all__ = [
    "spec_to_overrides",
    "write_experiment_yaml",
    "run_name_for_spec",
    "ENV_GROUP_BY_NAME",
]
