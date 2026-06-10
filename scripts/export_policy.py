#!/usr/bin/env python
"""Export a run's policy to ONNX (W11).

    python scripts/export_policy.py --run <run_name> \
        [--ckpt-dir checkpoints/<lineage>] [--out results/runs/<run>/media/policy.onnx]

Reads the run's resolved config (results/runs/<run>/config.json — the W8
dump), rebuilds the Trainer, resumes the checkpoint lineage when --ckpt-dir
is given (otherwise exports the INITIALIZED policy — useful only for smoke),
and writes the artifact + manifest entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.results_root)
    cfg_path = root / "runs" / args.run / "config.json"
    if not cfg_path.exists():
        print(f"no config.json for '{args.run}' (pre-W8 run?) — cannot rebuild the trainer")
        return 1
    cfg = OmegaConf.create(json.loads(cfg_path.read_text()))

    from mbrl.export import export_policy
    from mbrl.training import Trainer
    from mbrl.utils.checkpoint import CheckpointManager

    trainer = Trainer(cfg, int(cfg.env.obs_dim), int(cfg.env.action_dim), device="cpu")
    steps = 0
    if args.ckpt_dir:
        cm = CheckpointManager(args.ckpt_dir, OmegaConf.to_container(cfg, resolve=True))
        steps = cm.resume(trainer)
        if steps == 0:
            print(f"warning: nothing resumed from {args.ckpt_dir} — exporting the fresh policy")
    out = Path(args.out) if args.out else root / "runs" / args.run / "media" / "policy.onnx"
    path = export_policy(trainer, int(cfg.env.obs_dim), out, results_root=root,
                         run_name=args.run, env_steps=steps,
                         action_scale=float(cfg.env.get("action_scale", 1.0)))
    print(f"exported: {path} (resumed step {steps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
