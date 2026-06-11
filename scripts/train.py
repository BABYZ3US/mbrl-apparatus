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
import mbrl.envs  # noqa: F401 — side effect: registers TraceAtlas-v0 with gym
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
    local_run_name = f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}"
    local_log = MetricsLogger(cfg.logging.dir,
                              local_run_name,
                              meta={"group": cfg.experiment.name, "env": cfg.env.name,
                                    "seed": cfg.seed},
                              config=OmegaConf.to_container(cfg, resolve=True))
    trainer = Trainer(cfg, cfg.env.obs_dim, cfg.env.action_dim, device=device)
    buffer = ReplayBuffer(int(1e6), cfg.env.obs_dim, cfg.env.action_dim, cfg.seed)
    ckpt = CheckpointManager("checkpoints/" + run.name,
                             OmegaConf.to_container(cfg, resolve=True),
                             every=cfg.checkpoint.every,
                             keep_last=cfg.checkpoint.keep_last,
                             milestone_every=cfg.checkpoint.milestone_every,
                             wandb_run=run if cfg.checkpoint.push_wandb else None,
                             results_root=cfg.logging.dir,
                             run_name=f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}")
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
                    # W11: the SAME frames as a LOCAL artifact (GIF + manifest
                    # entry) so the Studio's Artifacts panel can browse + open it
                    from mbrl.eval import save_eval_media
                    save_eval_media(frames, Path(cfg.logging.dir),
                                    local_run_name, env_steps)
                except Exception as e:  # noqa: BLE001
                    # DISABLE, don't just silence: retrying a failed GL init
                    # re-enters corrupted GLFW static state and dies as a
                    # libc++abi ABORT (uncatchable) — the uniform 39k-step
                    # grid killer on the headless pod, 2026-06-11
                    video_enabled = False
                    if not video_warned:
                        print(f"[warn] eval video failed ({e!r}); video DISABLED "
                              "for this run — headless GL cannot recover")
                        video_warned = True
        # ---- M4: optional reward-surface + Hessian-spectrum export. OFF by
        # default; enable with `+viz.surface_every=N` (iterations). Writes a
        # pull.surface artifact (results/runs/<run>/surfaces/) and logs Hessian
        # eigenvalue summaries. Wrapped so viz can never kill a training run.
        _viz = cfg.get("viz", None)
        _surf_every = int(_viz.get("surface_every", 0)) if _viz else 0
        if _surf_every and iteration % _surf_every == 0:
            try:
                from mbrl.viz.surface_export import write_surface_json
                _run_name = f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}"
                _payload = trainer.reward_surface_payload(step=env_steps, run=_run_name)
                write_surface_json(_payload, cfg.logging.dir, _run_name, env_steps)
                _eigs = trainer.reward_hessian_eigs()
                if len(_eigs):
                    metrics["reward_hess/eig_max"] = float(_eigs[0])
                    metrics["reward_hess/eig_min"] = float(_eigs[-1])
                metrics["reward/curvature_budget"] = float(_payload.get("budget", 0.0))
            except Exception as _e:  # noqa: BLE001 — viz must never kill training
                print(f"[warn] surface export failed ({_e!r}); training continues")

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
