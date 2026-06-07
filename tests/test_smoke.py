"""End-to-end smoke test: ~600 real Pendulum steps through the full loop
(collect -> model learning w/ penalty -> imagination -> behaviour update ->
checkpoint). Asserts finite losses and a penalty that the schedule is acting on.
CPU, < 1 minute. This is the gate before any Colab time is spent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.training import Trainer, ReplayBuffer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything

CFG = OmegaConf.create({
    "seed": 0,
    "model": {"latent_dim": 4, "hidden": 64, "depth": 2, "ema_decay": 0.99},
    "penalty": {"n_probes": 2, "penalize_dynamics": False,
                "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
    "smoothing": {"enabled": True, "sigma": 1.5},
    "imagination": {"horizon": 8, "gamma": 0.99},
    "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 64},
})


def test_full_loop_smoke(tmp_path):
    import gymnasium as gym
    seed_everything(0)
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    trainer = Trainer(CFG, obs_dim, act_dim, device="cpu")
    buffer = ReplayBuffer(10_000, obs_dim, act_dim)
    ckpt = CheckpointManager(tmp_path, OmegaConf.to_container(CFG), every=10)

    obs, _ = env.reset(seed=0)
    metrics_log = []
    for it in range(3):
        for _ in range(200):  # collect
            with torch.no_grad():
                z = trainer.encoder(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = trainer.policy.sample(z)[0].squeeze(0).numpy()
            obs_next, r, term, trunc, _ = env.step(a)
            buffer.add(obs, a, r, obs_next)
            obs = obs_next
            if term or trunc:
                obs, _ = env.reset()
        for _ in range(20):  # model learning
            m = trainer.model_update(buffer.sample(CFG.optim.batch_size))
            metrics_log.append(m)
        z0 = trainer.encoder(buffer.sample(CFG.optim.batch_size)[0]).detach()
        b = trainer.behaviour_update(z0)  # imagination + policy/value
        assert np.isfinite(b["loss/value"]) and np.isfinite(b["loss/policy"])
        assert np.isfinite(b["imagine/return_var"])

    # all losses finite, penalty positive and finite, lambda annealing downward
    assert all(np.isfinite(m["loss/total"]) for m in metrics_log)
    assert all(m["penalty/value"] >= 0 for m in metrics_log)
    assert metrics_log[-1]["penalty/lambda"] < metrics_log[0]["penalty/lambda"]

    # checkpoint round-trip mid-run
    path = ckpt.save(trainer, env_steps=600, tag=f"step{trainer.step}")
    assert path.exists()
    t2 = Trainer(CFG, obs_dim, act_dim, device="cpu")
    assert ckpt.resume(t2) == 600
    assert t2.step == trainer.step
