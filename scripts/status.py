"""One-command experiment status (improvement plan #10).

Answers "what is running / finished / missing" from local artifacts only:
results/runs/<name>/metrics.jsonl (step counts, last write time) and
checkpoints/<name>/<cfg-hash>/ (lineages). Groups runs by arm (name minus the
trailing -s<seed>) and flags the ledger-pending validations whose arms have no
runs yet.

    python scripts/status.py            # table
    python scripts/status.py --stale 6  # mark groups idle > 6 h
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
CKPTS = ROOT / "checkpoints"

# arms the ledger is waiting on (colab_spectral, the spectral RL validation)
PENDING_VALIDATION = ("spec-ladder", "spec-single", "spec-auto",
                      "spec-learned", "mlp-recipe")


def last_step(metrics: Path):
    """(last step, mtime) from the JSONL tail without reading the whole file."""
    try:
        with open(metrics, "rb") as f:
            f.seek(max(-4096, -metrics.stat().st_size), 2)
            lines = f.read().decode(errors="ignore").strip().splitlines()
        row = json.loads(lines[-1])
        return row.get("step"), metrics.stat().st_mtime
    except Exception:
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stale", type=float, default=12.0,
                   help="hours of inactivity before a group is marked idle")
    args = p.parse_args()

    groups = defaultdict(list)   # arm -> [(seed_name, step, mtime)]
    if RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            m = d / "metrics.jsonl"
            if not m.exists():
                continue
            step, mtime = last_step(m)
            arm = re.sub(r"-s\d+$", "", d.name)
            groups[arm].append((d.name, step, mtime))

    now = time.time()
    print(f"{'group':<38} {'runs':>4} {'last step':>10} {'idle':>8}  ckpt lineages")
    for arm in sorted(groups):
        rows = groups[arm]
        steps = [s for _, s, _ in rows if s is not None]
        mtimes = [t for _, _, t in rows if t is not None]
        idle_h = (now - max(mtimes)) / 3600 if mtimes else float("inf")
        idle = f"{idle_h:.1f}h" + ("  IDLE" if idle_h > args.stale else "")
        lineages = 0
        for _, _, _ in rows:
            pass
        for d in CKPTS.glob(f"{arm}-*"):
            lineages += sum(1 for h in d.iterdir() if h.is_dir())
        print(f"{arm:<38} {len(rows):>4} {max(steps) if steps else '-':>10} "
              f"{idle:>8}  {lineages}")

    missing = [a for a in PENDING_VALIDATION
               if not any(g.startswith(a) for g in groups)]
    if missing:
        print(f"\nledger-pending spectral validation, arms with NO runs yet: "
              f"{', '.join(missing)}")
        print("  -> python scripts/parallel_runs.py --preset colab_spectral "
              "--overrides env=halfcheetah --seeds 0 1 2")


if __name__ == "__main__":
    main()
