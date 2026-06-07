"""W&B init/teardown with the project's conventions (PLAN.md section 3)."""
from __future__ import annotations

import os


def init_wandb(cfg, job_type: str = "train"):
    """One run per (experiment, env, seed); group = experiment name.
    Set WANDB_MODE=offline to run without network; sync later with `wandb sync`."""
    import importlib.util
    import wandb
    # sync_tensorboard hard-crashes wandb.init if tensorboard isn't installed;
    # degrade to plain logging instead
    tb_ok = (cfg.logging.tensorboard
             and importlib.util.find_spec("tensorboard") is not None)
    run = wandb.init(
        project=cfg.logging.project,
        group=cfg.experiment.name,
        job_type=job_type,
        name=f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}",
        config=dict(cfg) if not hasattr(cfg, "to_container") else None,
        sync_tensorboard=tb_ok,
        dir=cfg.logging.dir,
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    return run
