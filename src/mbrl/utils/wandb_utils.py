"""W&B init/teardown with the project's conventions (PLAN.md section 3)."""
from __future__ import annotations

import os


def init_wandb(cfg, job_type: str = "train"):
    """One run per (experiment, env, seed); group = experiment name.
    Set WANDB_MODE=offline to run without network; sync later with `wandb sync`."""
    import wandb
    run = wandb.init(
        project=cfg.logging.project,
        group=cfg.experiment.name,
        job_type=job_type,
        name=f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}",
        config=dict(cfg) if not hasattr(cfg, "to_container") else None,
        sync_tensorboard=cfg.logging.tensorboard,
        dir=cfg.logging.dir,
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    return run
