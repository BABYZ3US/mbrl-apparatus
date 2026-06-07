"""Recompute headline numbers from results JSON and check the ledger quotes
them (improvement plan #14). Catches prose/number rot.

    python scripts/ledger_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEDGER = (ROOT / "docs" / "claims_ledger.md").read_text()


def find(name: str):
    """Newest copy of a results file across legacy + sha-scoped locations."""
    hits = sorted(ROOT.glob(f"results/**/{name}"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def check(label: str, token: str):
    ok = token in LEDGER
    print(f"  {'OK     ' if ok else 'MISSING'} {label}: {token}")
    return ok


def main():
    bad = 0
    rec = find("bridge_recipe_test.json")
    if rec:
        rows = json.load(open(rec))
        base = "single05_quartic"
        arms = [a for a in rows[0] if isinstance(rows[0][a], dict)
                and "test_mse" in rows[0][a] and a != base]
        print(f"recipe results: {rec.relative_to(ROOT)} ({len(rows)} cells)")
        for arm in arms:
            rel = np.mean([(r[base]["test_mse"] - r[arm]["test_mse"])
                           / r[base]["test_mse"] for r in rows])
            bad += not check(arm, f"{rel:+.1%}".replace("%", "%"))
        cross = [r["sigma_at_snr1"] for r in rows if "sigma_at_snr1" in r]
        if cross:
            bad += not check("SNR=1 crossing", f"{np.mean(cross):.3f}")
    br = find("bridge_experiment.json")
    if br:
        rows = json.load(open(br))
        cells = [r for r in rows if all(not r[a].get("degenerate")
                 for a in ("frobenius_diag", "lap2_positive", "lap2_indefinite"))]
        hits = sum(1 for r in cells
                   if r["lap2_positive"]["test_mse"] < r["frobenius_diag"]["test_mse"]
                   < r["lap2_indefinite"]["test_mse"])
        print(f"bridge run-1 results: {br.relative_to(ROOT)}")
        bad += not check("run-1 ordering", f"0/{len(cells)}" if hits == 0
                         else f"{hits}/{len(cells)}")
    print("\nledger-check:", "PASS" if bad == 0 else f"{bad} headline(s) not "
          "found in docs/claims_ledger.md — update the prose or the numbers")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
