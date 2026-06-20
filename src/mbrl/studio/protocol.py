"""protocol — the single source of truth for the Studio Bridge wire contract.

One TCP socket, length-prefixed (4-byte little-endian uint32) UTF-8 JSON; every
message is ``{type, id, data}``. This module is the AUTHORITATIVE verb list for the
Python side: ``scripts/studio_bridge_server.py`` and the conformance test
(``tests/test_studio_protocol.py``) both check against ``VERBS`` here, and the Godot
side (``godot_studio/addons/mbrl_bridge/protocol.gd`` +
``godot_studio/test/fixtures/v0_1_protocol_contract.json``) should be reconciled
against it. (The historical fixture had drifted — it listed verbs served by neither
side and omitted one that is.)

Pure stdlib — safe to import inside the seal.

Status per verb:
  * ``served``  — handled by ``StudioBridgeServer.dispatch`` today.
  * ``godot``   — served on the GODOT side (the ``env.*`` train seam, ``infer.*``
                  in-engine ONNX); the Python server returns a ``not_served`` stub.
  * ``planned`` — in the v0.1 plan but NOT implemented on either side yet.

SHIM. The verb table is no longer hand-written here — it is derived from the
spine-generated artifact, vendored into the package as ``_spine_protocol`` (re-sync via
``spine/scripts/sync-mbrl.sh``). This module keeps the IDENTICAL public surface
(``VERSION``, ``VERBS``, ``verbs_by_status``, ``SERVED``, ``GODOT``, ``PLANNED``) so the
bridge server and the conformance test are unaffected. Pure stdlib — safe inside the seal.
"""
from __future__ import annotations

from ._spine_protocol import VERSION, VERBS  # single source of truth (spine-generated)


def verbs_by_status(status: str) -> set[str]:
    return {name for name, v in VERBS.items() if v["status"] == status}


SERVED = verbs_by_status("served")    # handled by the Python dispatch
GODOT = verbs_by_status("godot")      # documented not_served stubs on the Python side
PLANNED = verbs_by_status("planned")  # in the plan, not implemented anywhere yet
