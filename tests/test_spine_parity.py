"""Parity: the vendored studio protocol stays in lock-step with the spine artifact.

``mbrl/src/mbrl/studio/protocol.py`` is a thin shim over a vendored copy of the
spine-generated wire contract (``_spine_protocol``). If someone regenerates
``spine/generated/python/protocol.py`` but forgets to re-vendor it into the package
(via ``spine/scripts/sync-mbrl.sh``), these assertions fail — the vendored copy is
compared against the canonical file loaded BY PATH. Stdlib only; safe inside the seal.
"""
import importlib.util
import sys
from pathlib import Path

# repo root: tests -> mbrl -> <repo>
_TESTS = Path(__file__).resolve().parent
_MBRL = _TESTS.parent
_REPO = _MBRL.parent
_CANONICAL = _REPO / "spine" / "generated" / "python" / "protocol.py"

sys.path.insert(0, str(_MBRL / "src"))

from mbrl.studio import protocol
from mbrl.studio import _spine_protocol


def _load_canonical():
    """Load spine/generated/python/protocol.py directly from disk (not the vendored copy)."""
    assert _CANONICAL.is_file(), f"canonical spine artifact missing: {_CANONICAL}"
    spec = importlib.util.spec_from_file_location("_spine_canonical_protocol", _CANONICAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_matches_canonical_spine_artifact():
    if not _CANONICAL.is_file():
        # mbrl-only checkout (e.g. mbrl's CI) doesn't contain the sibling spine/
        # tree. The vendored copy is still checked internally below; the canonical
        # cross-check runs wherever spine/ is present (local + the monorepo CI).
        import pytest
        pytest.skip(f"canonical spine artifact not in this checkout: {_CANONICAL}")
    canonical = _load_canonical()
    assert _spine_protocol.VERBS == canonical.VERBS, (
        "vendored _spine_protocol.VERBS has drifted from spine/generated/python/protocol.py "
        "— re-sync via spine/scripts/sync-mbrl.sh"
    )
    assert _spine_protocol.VERSION == canonical.VERSION


def test_shim_verbs_equal_vendored_verbs():
    assert protocol.VERBS == _spine_protocol.VERBS


def test_shim_version_equals_vendored_version():
    assert protocol.VERSION == _spine_protocol.VERSION


def test_served_and_godot_partition_the_verbs():
    served = protocol.SERVED
    godot = protocol.GODOT
    assert served, "SERVED must be non-empty"
    assert godot, "GODOT must be non-empty"
    # disjoint
    assert not (served & godot), f"SERVED and GODOT overlap: {served & godot}"
    # every served/godot verb is a real verb in the table
    assert served <= set(protocol.VERBS)
    assert godot <= set(protocol.VERBS)
    # status partition is exact: served | godot | planned == all verbs
    assert served | godot | protocol.PLANNED == set(protocol.VERBS)
