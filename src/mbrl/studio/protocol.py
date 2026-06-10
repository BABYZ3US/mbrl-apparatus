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
"""
from __future__ import annotations

VERSION = 1

# verb name -> {status, dir, desc}
VERBS: dict[str, dict] = {
    "hello":             {"status": "served",  "dir": "->",  "desc": "handshake; reply {version}"},
    "pull.runs":         {"status": "served",  "dir": "<->", "desc": "run list: name, group, last_step, n_points, keys"},
    "pull.metric":       {"status": "served",  "dir": "<->", "desc": "full curve {steps, values} (sqlite-preferred)"},
    "pull.metric_since": {"status": "served",  "dir": "<->", "desc": "incremental curve, env_steps > since"},
    "pull.datasets":     {"status": "served",  "dir": "<->", "desc": "checkpoint / dataset catalog"},
    "pull.surface":      {"status": "served",  "dir": "<->", "desc": "reward-surface slice {z, curv, budget, path, ...}"},
    "submit.spec":       {"status": "served",  "dir": "->",  "desc": "author + launch one run"},
    "submit.sweep":      {"status": "served",  "dir": "->",  "desc": "expand axes x seeds -> launch each arm"},
    "pull.run_status":   {"status": "served",  "dir": "<->", "desc": "one run's launch state (running/finished/failed)"},
    "pull.launched":     {"status": "served",  "dir": "<->", "desc": "all launched runs + their states"},
    "pull.log":          {"status": "served",  "dir": "<->", "desc": "incremental run-log lines (live-tail)"},
    "run.cancel":        {"status": "served",  "dir": "->",  "desc": "terminate a running launched child"},
    "search.submit":     {"status": "served",  "dir": "<->", "desc": "W9 random search: sample arms (typed distributions), persist state, launch the first batch"},
    "search.status":     {"status": "served",  "dir": "<->", "desc": "a search's arm table (+ last metric per arm) or the catalog (no name)"},
    "search.tick":       {"status": "served",  "dir": "<->", "desc": "advance a search: sync states, median-rule stop losers (run.cancel), launch queued"},
    "env.reset":         {"status": "godot",   "dir": "->",  "desc": "train seam — served by Godot serve_env"},
    "env.step":          {"status": "godot",   "dir": "->",  "desc": "train seam — served by Godot serve_env"},
    "env.spec":          {"status": "godot",   "dir": "->",  "desc": "train seam — served by Godot serve_env"},
    "infer.load":        {"status": "godot",   "dir": "->",  "desc": "in-engine ONNX — not served by the runner"},
    "infer.run":         {"status": "godot",   "dir": "->",  "desc": "in-engine ONNX — not served by the runner"},
    "pull.artifacts":    {"status": "served",  "dir": "<->", "desc": "a run's artifact manifest (checkpoints + W&B artifacts) + its resolved config"},
    "pull.sweep":        {"status": "served",  "dir": "<->", "desc": "sweep cells grid: catalog (no name) or flattened arm-rows from results/<name>_cells.jsonl"},
    "pull.diagnostics":  {"status": "served",  "dir": "<->", "desc": "PCA/cross-validation reports: catalog (no name) or the named results/diagnostics/<name>.json payload"},
    "error":             {"status": "served",  "dir": "<-",  "desc": "error envelope {code, message}"},
}


def verbs_by_status(status: str) -> set[str]:
    return {name for name, v in VERBS.items() if v["status"] == status}


SERVED = verbs_by_status("served")    # handled by the Python dispatch
GODOT = verbs_by_status("godot")      # documented not_served stubs on the Python side
PLANNED = verbs_by_status("planned")  # in the plan, not implemented anywhere yet
