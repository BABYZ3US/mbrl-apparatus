"""Mode-B local CPU collector: parallel env interaction -> replay shards -> W&B artifacts.

Runs N worker processes, each stepping its own env with the latest policy
checkpoint (pulled from W&B), exporting buffer shards that the Colab GPU
trainer imports. Restart-tolerant; workers are independent.

Usage:
  python scripts/collect.py --run-path you/mbrl-curvature/<run_id> --workers 8 --steps 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def worker(worker_id: int, args) -> str:
    import gymnasium as gym
    import numpy as np
    import torch
    from mbrl.training.buffer import ReplayBuffer

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buf = ReplayBuffer(args.steps, obs_dim, act_dim, seed=args.seed + worker_id)

    policy = None
    if args.run_path:  # pull latest policy; else random collection
        from mbrl.utils.checkpoint import CheckpointManager
        import wandb
        api = wandb.Api()
        art = api.artifact(f"{args.run_path}:latest", type="checkpoint")
        ckpt_dir = Path(art.download())
        payload = torch.load(next(ckpt_dir.glob("*.pt")), weights_only=False,
                             map_location="cpu")
        # policy-only restore for collection
        from mbrl.models import Encoder, Policy
        k = payload["trainer"]["policy"]["net.0.weight"].shape[1]
        policy = Policy(k, act_dim)
        policy.load_state_dict(payload["trainer"]["policy"])
        encoder = Encoder(obs_dim, k)
        encoder.load_state_dict(payload["trainer"]["encoder"])

    obs, _ = env.reset(seed=args.seed + worker_id)
    for _ in range(args.steps):
        if policy is None:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                z = encoder(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = policy.sample(z)[0].squeeze(0).numpy()
        obs_next, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, obs_next)
        obs = obs_next
        if term or trunc:
            obs, _ = env.reset()

    out = Path(args.out) / f"shard_w{worker_id}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    buf.export_shard(out)
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-path", default=None, help="entity/project/run_id for policy ckpt")
    p.add_argument("--out", default="results/shards")
    p.add_argument("--upload", action="store_true", help="push shards as W&B artifact")
    args = p.parse_args()

    from joblib import Parallel, delayed
    paths = Parallel(n_jobs=args.workers)(
        delayed(worker)(i, args) for i in range(args.workers))
    print("shards:", paths)

    if args.upload:
        import wandb
        run = wandb.init(project="mbrl-curvature", job_type="collect")
        art = wandb.Artifact(f"replay-{args.env}", type="replay")
        for path in paths:
            art.add_file(path)
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
