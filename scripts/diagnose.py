"""diagnose — produce a PCA + cross-validation diagnostics artifact from a JSONL dataset.

Pure numpy (no torch): runs anywhere, including over the trace-atlas corpus and
the sweep cells files. Writes results/diagnostics/<name>.json, which the bridge
serves to the Studio's Diagnostics panel via pull.diagnostics.

    # PCA + 5-fold ridge probe: do spectral features predict the Godel code?
    python scripts/diagnose.py --source ../trace-atlas/train.jsonl \
        --features spectral --target godel_code --name atlas_spectral

    # PCA only (no target): the scree of any list-valued feature column
    python scripts/diagnose.py --source data.jsonl --features spectral --name my_report

Feature column: a LIST-valued key (one vector per row) or a comma-list of scalar
keys. Target: a scalar key (optional; enables the K-fold ridge probe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mbrl.diagnostics import pca_diagnostics, kfold_ridge  # noqa: E402
from mbrl.diagnostics.export import write_diagnostics_json  # noqa: E402


def load_jsonl(path: Path, features: str, target: str | None,
               limit: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    keys = [k.strip() for k in features.split(",")]
    X_rows, y_rows = [], []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line
            if len(keys) == 1 and isinstance(row.get(keys[0]), list):
                feats = row[keys[0]]
            else:
                feats = [row.get(k) for k in keys]
            if any(not isinstance(v, (int, float)) for v in feats):
                continue
            if target is not None and not isinstance(row.get(target), (int, float)):
                continue
            X_rows.append([float(v) for v in feats])
            if target is not None:
                y_rows.append(float(row[target]))
            if limit and len(X_rows) >= limit:
                break
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64) if target is not None else None
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="input .jsonl")
    ap.add_argument("--features", required=True,
                    help="list-valued key (e.g. spectral) or comma-list of scalar keys")
    ap.add_argument("--target", default=None, help="scalar key for the K-fold ridge probe")
    ap.add_argument("--name", required=True, help="report name (artifact filename)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--components", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (quick looks)")
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    X, y = load_jsonl(Path(args.source), args.features, args.target, args.limit)
    if X.shape[0] < max(2, args.folds):
        raise SystemExit(f"only {X.shape[0]} usable rows in {args.source} — nothing to diagnose")

    payload: dict = {
        "source": str(args.source), "features": args.features,
        "target": args.target, "pca": pca_diagnostics(X, args.components),
    }
    if y is not None:
        payload["crossval"] = kfold_ridge(X, y, k=args.folds, alpha=args.alpha, seed=args.seed)

    out = write_diagnostics_json(payload, args.results_root, args.name)
    print(f"wrote {out}")
    print(f"  rows={X.shape[0]} d={X.shape[1]} "
          f"effective_dim={payload['pca']['effective_dim']:.2f}")
    if y is not None:
        cv = payload["crossval"]
        print(f"  ridge {args.folds}-fold R^2 = {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")


if __name__ == "__main__":
    main()
