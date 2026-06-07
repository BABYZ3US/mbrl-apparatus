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

from mbrl.training import Trainer, ReplayBuffer, collect_vectorized
from mbrl.utils import CheckpointManager, seed_everything, init_wandb
from mbrl.utils.metrics_logger import MetricsLogger


def make_env(cfg, num_envs: int):
    import gymnasium as gym
    return gym.vector.AsyncVectorEnv(
        [lambda: gym.make(cfg.env.name) for _ in range(num_envs)])


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


def record_episode_frames(trainer, cfg, device) -> list:
    """One extra eval episode with rgb_array rendering — frames for wandb.Video."""
    import gymnasium as gym
    env = gym.make(cfg.env.name, render_mode="rgb_array")
    frames = []
    obs, _ = env.reset(seed=cfg.seed)
    done = False
    while not done:
        frames.append(env.render())
        with torch.no_grad():
            z = trainer.encoder(torch.as_tensor(obs, dtype=torch.float32,
                                                device=device).unsqueeze(0))
            a, _ = trainer.policy.sample(z)
        obs, _, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
        done = term or trunc
    env.close()
    return frames


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

    num_envs = int(cfg.training.get("num_envs", 1))
    env = make_env(cfg, num_envs)
    obs, _ = env.reset(seed=cfg.seed)
    autoreset = np.zeros(num_envs, dtype=bool)
    import time as _time
    _t_run = _time.perf_counter()
    print(f"[train] {cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}: starting "
          f"at env_steps={env_steps}, target={cfg.training.total_env_steps} "
          f"({cfg.training.steps_per_iter} steps + "
          f"{cfg.training.model_updates_per_iter} model updates / iter — the "
          "FIRST W&B point lands when iteration 1 completes)", flush=True)

    video_cfg = cfg.logging.get("video", None)
    video_enabled = bool(video_cfg and video_cfg.get("enabled", False))
    video_every = int(video_cfg.get("every_evals", 4)) if video_cfg else 4
    video_warned = False
    eval_count = 0

    while env_steps < cfg.training.total_env_steps:
        # ---- collect (CPU, vectorized; gymnasium next-step autoreset masked) ----
        obs, autoreset, taken = collect_vectorized(
            trainer, env, buffer, obs, autoreset,
            cfg.training.steps_per_iter, device=device)
        env_steps += taken

        # ---- model learning (GPU) ----
        for _u in range(cfg.training.model_updates_per_iter):
            metrics = trainer.model_update(buffer.sample(cfg.optim.batch_size))
            if _u % 50 == 0:   # live telemetry between iteration commits
                run.log({"live/loss_total": metrics["loss/total"],
                         "live/model_update": trainer.step})
        # ---- behaviour learning on imagined rollouts (GPU) ----
        # (was a single update per iteration — a bug that starved the policy at
        # ~100 updates per run and pinned all schedule-ablation arms at
        # random-policy level; the config key was always meant to be consumed)
        for _ in range(cfg.training.behaviour_updates_per_iter):
            z0 = trainer.encoder(
                buffer.sample(cfg.optim.batch_size)[0].to(device)).detach()
            metrics |= trainer.behaviour_update(z0)

        metrics["env_steps"] = env_steps
        iteration = env_steps // cfg.training.steps_per_iter
        if iteration % cfg.training.eval_every_iters == 0:
            metrics["eval/return"] = evaluate(trainer, cfg, device)
            eval_count += 1
            if video_enabled and eval_count % video_every == 0:
                # never let a headless/render failure kill training
                try:
                    import wandb
                    frames = record_episode_frames(trainer, cfg, device)
                    metrics["eval/video"] = wandb.Video(
                        np.stack(frames).transpose(0, 3, 1, 2),
                        fps=30, format="mp4")
                except Exception as e:  # noqa: BLE001
                    if not video_warned:
                        print(f"[warn] eval video logging failed ({e!r}); "
                              "training continues without videos")
                        video_warned = True
        run.log(metrics)
        local_log.log(metrics)   # offline mirror -> figures without network
        print(f"[train] iter {iteration} env_steps={env_steps} "
              f"loss/total={metrics.get('loss/total', float('nan')):.4f} "
              f"eval={metrics.get('eval/return', '-')} "
              f"({_time.perf_counter() - _t_run:.0f}s elapsed)", flush=True)
        ckpt.maybe_save(trainer, env_steps)

    ckpt.save(trainer, env_steps, tag=f"step{trainer.step}")
    env.close()
    local_log.close()
    run.finish()


if __name__ == "__main__":
    main()
