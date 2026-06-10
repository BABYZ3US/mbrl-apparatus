"""CellsIndex + the pull.sweep verb: catalog and flattened arm-rows over
results/*_cells.jsonl (the sweep OUTCOME grids — NOT time-series).

Stdlib only; server constructed dry-run on a tmp repo so nothing launches.
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import studio_bridge_server as sb
from mbrl.studio.cells_index import CellsIndex


def _write_cells(results_root: Path, name: str = "bench") -> Path:
    """Two cells x two arms, with the real files' quirks: a string metric
    ("choice"), a list metric (dropped), and a torn trailing line (skipped)."""
    results_root.mkdir(parents=True, exist_ok=True)
    lines = [
        {"n": 256, "seed": 0,
         "armA": {"lam": 0.01, "test_mse": 0.2, "wall_s": 0.1},
         "armB": {"lam": 1.0, "test_mse": 0.4, "choice": "lam=1"}},
        {"n": 256, "seed": 1,
         "armA": {"lam": 0.01, "test_mse": 0.4},
         "armB": {"lam": 1.0, "test_mse": 0.6, "band_snrs": [1, 2, 3]}},
    ]
    p = results_root / f"{name}_cells.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + '\n{"torn": ')
    return p


# ---------------- CellsIndex ----------------

def test_list_sweeps_catalogs_cells_files(tmp_path):
    _write_cells(tmp_path / "results", "bench")
    _write_cells(tmp_path / "results", "other")
    idx = CellsIndex(tmp_path / "results")
    cat = idx.list_sweeps()
    assert [c["name"] for c in cat] == ["bench", "other"]
    assert cat[0]["file"] == "bench_cells.jsonl"
    assert cat[0]["n_lines"] == 3  # 2 cells + the torn line (size hint, not row count)


def test_list_sweeps_missing_root_is_empty(tmp_path):
    assert CellsIndex(tmp_path / "nope").list_sweeps() == []


def test_read_cells_flattens_arm_rows(tmp_path):
    _write_cells(tmp_path / "results")
    out = CellsIndex(tmp_path / "results").read_cells("bench")
    assert out["found"] is True
    assert out["arms"] == ["armA", "armB"]
    assert len(out["rows"]) == 4  # 2 lines x 2 arms; the torn line skipped

    a0 = next(r for r in out["rows"] if r["arm"] == "armA" and r["seed"] == 0)
    assert a0["n"] == 256 and a0["lam"] == 0.01 and a0["test_mse"] == 0.2

    # numeric fields are offered as metric candidates; strings/lists are not
    assert "test_mse" in out["fields"] and "lam" in out["fields"]
    assert "choice" not in out["fields"]
    # the string metric itself is kept on the row (categorical label) ...
    b0 = next(r for r in out["rows"] if r["arm"] == "armB" and r["seed"] == 0)
    assert b0["choice"] == "lam=1"
    # ... but list values are dropped entirely
    assert all("band_snrs" not in r for r in out["rows"])


def test_read_cells_missing_sweep(tmp_path):
    out = CellsIndex(tmp_path / "results").read_cells("ghost")
    assert out == {"sweep": "ghost", "found": False,
                   "rows": [], "arms": [], "fields": []}


def test_read_cells_flat_line_degrades_to_single_row(tmp_path):
    root = tmp_path / "results"
    root.mkdir(parents=True)
    (root / "flat_cells.jsonl").write_text(
        json.dumps({"n": 64, "seed": 3, "test_mse": 0.5}) + "\n")
    out = CellsIndex(root).read_cells("flat")
    assert len(out["rows"]) == 1
    assert out["rows"][0] == {"arm": "", "n": 64, "seed": 3, "test_mse": 0.5}
    assert out["arms"] == []  # "" is not a named arm


# ---------------- pull.sweep through the server ----------------

def test_dispatch_pull_sweep_catalog_and_named(tmp_path):
    _write_cells(tmp_path / "results")
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)

    cat = srv.dispatch(sb.make(sb.PULL_SWEEP, {}, 7))
    assert cat["type"] == sb.PULL_SWEEP and cat["id"] == 7
    assert [s["name"] for s in cat["data"]["sweeps"]] == ["bench"]

    named = srv.dispatch(sb.make(sb.PULL_SWEEP, {"sweep": "bench"}, 8))
    assert named["data"]["found"] is True
    assert len(named["data"]["rows"]) == 4


def test_dispatch_pull_sweep_empty_repo(tmp_path):
    srv = sb.StudioBridgeServer(repo_root=tmp_path, dry_run=True)
    assert srv.dispatch(sb.make(sb.PULL_SWEEP, {}, 1))["data"] == {"sweeps": []}
    missing = srv.dispatch(sb.make(sb.PULL_SWEEP, {"sweep": "ghost"}, 2))
    assert missing["data"]["found"] is False
