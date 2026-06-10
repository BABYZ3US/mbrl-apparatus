"""DiagnosticsIndex + the pull.diagnostics verb (catalog-or-named contract)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import studio_bridge_server as sb
from mbrl.diagnostics.export import write_diagnostics_json
from mbrl.studio.diagnostics_index import DiagnosticsIndex


def _write(root: Path, name: str = "atlas") -> None:
    write_diagnostics_json(
        {"pca": {"explained_variance_ratio": [0.7, 0.2, 0.1], "cumulative": [0.7, 0.9, 1.0],
                 "effective_dim": 1.9, "n_rows": 100, "n_features": 3},
         "crossval": {"r2_per_fold": [0.8, 0.82], "r2_mean": 0.81, "r2_std": 0.01,
                      "folds": 2, "alpha": 1.0, "probe": "ridge", "seed": 0}},
        root, name)


def test_catalog_and_named(tmp_path):
    _write(tmp_path / "results", "atlas")
    idx = DiagnosticsIndex(tmp_path / "results")
    cat = idx.list_reports()
    assert [c["name"] for c in cat] == ["atlas"]
    assert "created" in cat[0]
    rep = idx.get_report("atlas")
    assert rep["found"] is True
    assert rep["pca"]["effective_dim"] == 1.9


def test_missing_and_torn(tmp_path):
    root = tmp_path / "results"
    (root / "diagnostics").mkdir(parents=True)
    (root / "diagnostics" / "torn.json").write_text('{"broken":')
    idx = DiagnosticsIndex(root)
    assert idx.list_reports() == []                      # torn file skipped
    assert idx.get_report("ghost") == {"name": "ghost", "found": False}
    assert idx.get_report("torn")["found"] is False


def test_dispatch_pull_diagnostics(tmp_path):
    _write(tmp_path / "results", "atlas")
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    cat = srv.dispatch(sb.make(sb.PULL_DIAGNOSTICS, {}, 3))
    assert [i["name"] for i in cat["data"]["items"]] == ["atlas"]
    named = srv.dispatch(sb.make(sb.PULL_DIAGNOSTICS, {"name": "atlas"}, 4))
    assert named["data"]["found"] is True and named["id"] == 4
    ghost = srv.dispatch(sb.make(sb.PULL_DIAGNOSTICS, {"name": "ghost"}, 5))
    assert ghost["data"]["found"] is False
