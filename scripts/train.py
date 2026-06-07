"""Hydra entry point — runs the full MBRL loop (Mode A). GPU or CPU.

Usage:
  python scripts/train.py                                  # dev run, Pendulum, CPU/GPU auto
  python scripts/train.py +experiment=multienv env=walker2d seed=3
  python scripts/train.py checkpoint.resume=auto           # resume after Colab disconnect
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mbrl.training import Trainer, ReplayBuffer
from mbrl.utils import CheckpointManager, seed_everything, init_wandb
from mbrl.utils.metrics_logger import MetricsLogger


def make_env(cfg, num_envs: int):
    import gymnasium as gym
    if num_envs > 1:
        return gym.vector.AsyncVectorEnv(
            [lambda: gym.make(cfg.env.name) for _ in range(num_envs)])
    return gym.make(cfg.env.name)


def evaluate(trainer, cfg, device, episodes: int = 3) -> float:
    import gymnasium as gym
    env = gym.make(cfg.env.name)
    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed * 1000 + ep)
        done = False
        while not done:
            with torch.no_grad():
                z = trainer.encoder(torch.as_tensor(obs, dtype=torch.float32,
                                                    device=device).unsqueeze(0))
                a, _ = trainer.policy.sample(z)
            obs, r, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            total += r
            done = term or trunc
    env.close()
    return total / episodes


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    if cfg.device == "auto":
        # cuda (Colab) > cpu. MPS is deliberately NOT auto-selected: the penalty's
        # double backward fails on MPS (verified on the M2 via check_mps.py).
        # If a future torch fixes it, re-verify with check_mps.py, then opt in
        # explicitly with device=mps.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    seed_everything(cfg.seed)

    run = init_wandb(cfg, job_type="train")
    local_log = MetricsLogger(cfg.logging.dir,
                              f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}",
                              meta={"group": cfg.experiment.name, "env": cfg.env.name,
                                    "seed": cfg.seed})
    trainer = Trainer(cfg, cfg.env.obs_dim, cfg.env.action_dim, device=device)
    buffer = ReplayBuffer(int(1e6), cfg.env.obs_dim, cfg.env.action_dim, cfg.seed)
    ckpt = CheckpointManager("checkpoints/" + run.name,
                             OmegaConf.to_container(cfg, resolve=True),
                             every=cfg.checkpoint.every,
                             keep_last=cfg.checkpoint.keep_last,
                             milestone_every=cfg.checkpoint.milestone_every,
                             wandb_run=run if cfg.checkpoint.push_wandb else None)
    env_steps = ckpt.resume(trainer) if cfg.checkpoint.resume == "auto" else 0
    ckpt.install_signal_handler(trainer, lambda: env_steps)

    env = make_env(cfg, 1)
    obs, _ = env.reset(seed=cfg.seed)

    while env_steps < cfg.training.total_env_steps:
        # ---- collect (CPU) ----
        for _ in range(cfg.training.steps_per_iter):
            with torch.no_grad():
                z = trainer.encoder(torch.as_tensor(obs, dtype=torch.float32,
                                                    device=device).unsqueeze(0))
                a, _ = trainer.policy.sample(z)
            a_np = a.squeeze(0).cpu().numpy()
            obs_next, r, term, trunc, _ = env.step(a_np)
            buffer.add(obs, a_np, r, obs_next)
            obs = obs_next
            env_steps += 1
            if term or trunc:
                obs, _ = env.reset()

        # ---- model learning (GPU) ----
        for _ in range(cfg.training.model_updates_per_iter):
            metrics = trainer.model_update(buffer.sample(cfg.optim.batch_size))
        # ---- behaviour learning on imagined rollouts (GPU) ----
        z0 = trainer.encoder(buffer.sample(cfg.optim.batch_size)[0].to(device)).detach()
        metrics |= trainer.behaviour_update(z0)

        metrics["env_steps"] = env_steps
        iteration = env_steps // cfg.training.steps_per_iter
        if iteration % cfg.training.eval_every_iters == 0:
            metrics["eval/return"] = evaluate(trainer, cfg, device)
        run.log(metrics)
        local_log.log(metrics)   # offline mirror -> figures without network
        ckpt.maybe_save(trainer, env_steps)

    ckpt.save(trainer, env_steps, tag=f"step{trainer.step}")
    local_log.close()
    run.finish()


if __name__ == "__main__":
    main()
