"""Multi-task zero-shot generalization experiment (validation item 9).

Train one task-conditioned world model + policy on n_train tasks (tau values);
periodically evaluate zero-shot on held-out taus — interpolation (inside the
training range) and extrapolation (outside), reported separately. The science
ablation is the same command with penalty.schedule.lam0=0.

  python scripts/train_multitask.py                                   # Pendulum family, CPU
  python scripts/train_multitask.py env=halfcheetah_vel               # Colab
  python scripts/train_multitask.py penalty.schedule.lam0=0 seed=0    # ablation arm
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mbrl.envs.tasks import make_task_env, task_split
from mbrl.training import Trainer, ReplayBuffer
from mbrl.utils import CheckpointManager, seed_everything, init_wandb
from mbrl.utils.metrics_logger import MetricsLogger


def eval_zero_shot(trainer, family: str, taus: list[float], device, seed: int,
                   episodes: int = 2) -> dict[float, float]:
    out = {}
    for tau in taus:
        env = make_task_env(family, tau)
        total = 0.0
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed * 7919 + ep)
            tau_t = torch.as_tensor(env.tau, device=device).unsqueeze(0)
            done = False
            while not done:
                with torch.no_grad():
                    z = trainer.encoder(torch.as_tensor(
                        obs, dtype=torch.float32, device=device).unsqueeze(0))
                    a, _ = trainer.policy.sample(z, tau_t)
                obs, r, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
                total += r
                done = term or trunc
        env.close()
        out[tau] = total / episodes
    return out


@hydra.main(config_path="../configs", config_name="multitask", version_base=None)
def main(cfg: DictConfig):
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg.device == "auto" else cfg.device
    seed_everything(cfg.seed)

    family = cfg.env.family
    split = task_split(family, n_train=cfg.tasks.n_train, seed=cfg.tasks.split_seed)
    probe = make_task_env(family, split["train"][0])
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    task_dim = probe.task_dim
    probe.close()

    run = init_wandb(cfg, job_type="train")
    run.summary["task_split"] = split
    local_log = MetricsLogger(cfg.logging.dir,
                              f"{cfg.experiment.name}-{family}-s{cfg.seed}",
                              meta={"group": cfg.experiment.name, "env": family,
                                    "seed": cfg.seed, "task_split": split,
                                    "lam0": cfg.penalty.schedule.lam0})
    trainer = Trainer(cfg, obs_dim, act_dim, device=device, task_dim=task_dim)
    buffer = ReplayBuffer(int(1e6), obs_dim, act_dim, cfg.seed, task_dim=task_dim)
    ckpt = CheckpointManager("checkpoints/" + run.name,
                             OmegaConf.to_container(cfg, resolve=True),
                             every=cfg.checkpoint.every, wandb_run=run)
    env_steps = ckpt.resume(trainer) if cfg.checkpoint.resume == "auto" else 0

    rng = np.random.default_rng(cfg.seed)
    env, tau_now = None, None

    while env_steps < cfg.training.total_env_steps:
        # ---- collect: resample a training task each outer iteration ----
        tau_now = float(rng.choice(split["train"]))
        if env is not None:
            env.close()
        env = make_task_env(family, tau_now)
        obs, _ = env.reset(seed=cfg.seed + env_steps)
        tau_t = torch.as_tensor(env.tau, device=device).unsqueeze(0)
        for _ in range(cfg.training.steps_per_iter):
            with torch.no_grad():
                z = trainer.encoder(torch.as_tensor(
                    obs, dtype=torch.float32, device=device).unsqueeze(0))
                a, _ = trainer.policy.sample(z, tau_t)
            a_np = a.squeeze(0).cpu().numpy()
            obs_next, r, term, trunc, _ = env.step(a_np)
            buffer.add(obs, a_np, r, obs_next, tau=env.tau)
            obs = obs_next
            env_steps += 1
            if term or trunc:
                obs, _ = env.reset()

        # ---- model + behaviour learning ----
        for _ in range(cfg.training.model_updates_per_iter):
            metrics = trainer.model_update(buffer.sample(cfg.optim.batch_size))
        for _ in range(cfg.training.behaviour_updates_per_iter):
            ob, _, _, _, tb = buffer.sample(cfg.optim.batch_size)
            z0 = trainer.encoder(ob.to(device)).detach()
            bmetrics = trainer.behaviour_update(z0, tb.to(device))
        metrics |= bmetrics
        metrics["env_steps"] = env_steps

        # ---- zero-shot eval on held-out tasks ----
        iteration = env_steps // cfg.training.steps_per_iter
        if iteration % cfg.training.eval_every_iters == 0:
            tr = eval_zero_shot(trainer, family, split["train"][:4], device, cfg.seed)
            ip = eval_zero_shot(trainer, family, split["interp"], device, cfg.seed)
            ex = eval_zero_shot(trainer, family, split["extrap"], device, cfg.seed)
            metrics["eval/return"] = float(np.mean(list(tr.values())))   # train tasks
            metrics["eval/zeroshot_interp"] = float(np.mean(list(ip.values())))
            metrics["eval/zeroshot_extrap"] = float(np.mean(list(ex.values())))
            metrics["eval/generalization_gap"] = \
                metrics["eval/return"] - metrics["eval/zeroshot_interp"]
            metrics["eval/per_task"] = {str(k): v for d in (tr, ip, ex)
                                        for k, v in d.items()}

        run.log({k: v for k, v in metrics.items() if not isinstance(v, dict)})
        local_log.log(metrics)
        ckpt.maybe_save(trainer, env_steps)

    ckpt.save(trainer, env_steps, tag=f"step{trainer.step}")
    local_log.close()
    run.finish()


if __name__ == "__main__":
    main()
