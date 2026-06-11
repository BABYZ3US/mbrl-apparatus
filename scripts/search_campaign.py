#!/usr/bin/env python
"""Random-search sweep over {penalty dose, spectral bandwidth, schedule x gate}
for the champion + spectral stacks (PM axes 1/2/4, 2026-06-11).

A full grid of these 4 knobs is 70+ arms; random search samples the joint space
instead. Screen at 1 seed (this script), read W&B, then promote the top configs
to a 3-seed confirmation. Uses mbrl.search.sample_axes (the W9 core) so the draw
is seeded/reproducible. Launches train.py subprocesses directly (no bridge/tick
loop — robust on a flaky pod), round-robin across GPUs, throttled to JOBS.

  ARMS=10 STEPS=150000 JOBS=4 SEED=0 python scripts/search_campaign.py
  python scripts/search_campaign.py --dry-run        # print the sampled arms, launch nothing

Axes (each arm draws one value per axis):
  penalty.schedule.lam0        loguniform[1e-2, 1.0]   (dose; auto_dose forced OFF)
  spectral.n_features          choice {256, 512, 1024} (RFF bandwidth)
  penalty.schedule.kind        choice {cuberoot, constant} (both non-zero-touching;
                               the spectral rule forbids zero-touching schedules)
  penalty.disagreement_gate.enabled  choice {true, false}
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mbrl.search import sample_axes

AXES = [
    {"path": "penalty.schedule.lam0", "kind": "loguniform", "low": 1e-2, "high": 1.0},
    {"path": "spectral.n_features", "kind": "choice", "values": [256, 512, 1024]},
    {"path": "penalty.schedule.kind", "kind": "choice", "values": ["cuberoot", "constant"]},
    {"path": "penalty.disagreement_gate.enabled", "kind": "choice", "values": ["true", "false"]},
]
STACKS = {"champion": "champion", "spectral": "spectral_ladder"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=int, default=int(os.environ.get("ARMS", 10)))
    ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", 150000)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 4)))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ngpu = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                     text=True).stdout.strip().splitlines())
               if not args.dry_run else 1)
    logdir = ROOT / "results" / "gridlogs"
    logdir.mkdir(parents=True, exist_ok=True)
    py = str(ROOT / ".venv" / "bin" / "python")

    # build the full arm list (both stacks), distinct sample seeds per stack
    arms = []
    for si, (stack, exp) in enumerate(STACKS.items()):
        for i, ov in enumerate(sample_axes(AXES, args.arms, seed=args.seed + 1000 * si)):
            tag = f"srch-{stack}-a{i:02d}"
            over = [f"+experiment={exp}", "env=halfcheetah", "seed=0",
                    "penalty.auto_dose.enabled=false", "penalty.schedule.floor=1e-5",
                    f"experiment.name={tag}", f"training.total_env_steps={args.steps}",
                    "logging.video.enabled=false", f"hydra.run.dir=outputs/{tag}"]
            for path, val in ov.items():
                over.append(f"{path}={val}")
            arms.append((tag, over))

    print(f"{len(arms)} arms ({args.arms}/stack x {len(STACKS)} stacks), "
          f"{args.steps} steps, {ngpu} GPU(s), JOBS={args.jobs}")
    for tag, over in arms:
        sampled = " ".join(o for o in over if any(o.startswith(a["path"]) for a in AXES))
        print(f"  {tag}: {sampled}")
    if args.dry_run:
        return 0

    procs, idx = [], 0
    for tag, over in arms:
        while sum(1 for p, _ in procs if p.poll() is None) >= args.jobs:
            time.sleep(5)
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   CUDA_VISIBLE_DEVICES=str(idx % ngpu))
        if "WANDB_API_KEY" not in env and (ROOT / ".wandb_key").exists():
            env["WANDB_API_KEY"] = (ROOT / ".wandb_key").read_text().strip()
        log = open(logdir / f"{tag}.log", "w")
        procs.append((subprocess.Popen([py, "scripts/train.py", *over], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, env=env), tag))
        idx += 1
        time.sleep(2)
        print(f"launched {tag} on GPU {(idx - 1) % ngpu}")

    fail = 0
    for p, tag in procs:
        if p.wait() != 0:
            fail += 1
            print(f"FAILED: {tag}")
    print(f"search done: {len(arms) - fail}/{len(arms)} succeeded")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
