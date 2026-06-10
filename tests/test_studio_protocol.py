"""Conformance: the Python server's verbs match the protocol SSOT (mbrl.studio.protocol).

Locks the wire contract against drift — the exact failure mode that produced two run
readers with two group rules. Stdlib only; dry-run so nothing launches.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import studio_bridge_server as sb
from mbrl.studio import protocol


def test_every_server_verb_constant_is_in_the_ssot():
    server_verbs = {sb.HELLO, sb.PULL_RUNS, sb.PULL_METRIC, sb.PULL_METRIC_SINCE,
                    sb.PULL_DATASETS, sb.PULL_ARTIFACTS, sb.PULL_SURFACE, sb.SUBMIT_SPEC,
                    sb.SUBMIT_SWEEP, sb.PULL_RUN_STATUS, sb.PULL_LAUNCHED, sb.PULL_LOG,
                    sb.RUN_CANCEL, sb.ENV_RESET, sb.ENV_STEP, sb.ENV_SPEC, sb.INFER_LOAD,
                    sb.INFER_RUN, sb.ERROR}
    unknown = server_verbs - set(protocol.VERBS)
    assert not unknown, f"server defines verbs absent from the SSOT: {unknown}"


def test_served_verbs_are_actually_handled(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    for verb in protocol.SERVED:
        if verb == "error":
            continue  # an outbound envelope, not an inbound request
        reply = srv.dispatch(sb.make(verb, {}, 1))
        code = reply["data"].get("code") if isinstance(reply["data"], dict) else None
        assert code != "unknown_type", f"SSOT says '{verb}' is served but dispatch rejected it"


def test_godot_verbs_return_not_served_stub(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    for verb in protocol.GODOT:
        assert srv.dispatch(sb.make(verb, {}, 1))["data"].get("code") == "not_served"


def test_planned_verbs_are_not_yet_served(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    for verb in protocol.PLANNED:
        assert srv.dispatch(sb.make(verb, {}, 1))["data"].get("code") == "unknown_type"


def test_unknown_verb_is_rejected(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    assert srv.dispatch(sb.make("totally.bogus", {}, 1))["data"]["code"] == "unknown_type"
