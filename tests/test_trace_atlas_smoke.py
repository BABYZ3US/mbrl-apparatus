"""Smoke: the curvature-MBRL Trainer runs on the trace-atlas reconstruction env.

A few collect / model-update / behaviour-update iterations on the REAL corpus
(val.jsonl); asserts finite losses, a non-negative penalty, and that reconstruction
reward is actually collected. Single-env loop (no AsyncVectorEnv), CPU. Marked slow.
"""
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.envs.trace_atlas import make_trace_atlas_env
from mbrl.training import ReplayBuffer, Trainer

CFG = OmegaConf.create({
    "seed": 0,
    "model": {"latent_dim": 8, "hidden": 64, "depth": 2, "ema_decay": 0.99},
    "penalty": {"n_probes": 2, "penalize_dynamics": False,
                "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
    "smoothing": {"enabled": True, "sigma": 1.5},
    "imagination": {"horizon": 5, "gamma": 0.99},
    "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 32},
})


def test_trainer_runs_on_trace_atlas():
    env = make_trace_atlas_env(None, embed_dim=32, seed=0)   # default corpus = val.jsonl
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    assert obs_dim == 8 and act_dim == 32

    trainer = Trainer(CFG, obs_dim, act_dim, device="cpu")
    buf = ReplayBuffer(5000, obs_dim, act_dim)

    obs, _ = env.reset(seed=0)
    rewards, n_correct = [], 0
    for _ in range(300):  # collect (every step is a 1-step reconstruction episode)
        with torch.no_grad():
            z = trainer.encoder(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            a = trainer.policy.sample(z)[0].squeeze(0).numpy()
        obs_next, r, term, trunc, info = env.step(a)
        buf.add(obs, a, r, obs_next)
        rewards.append(r)
        n_correct += int(info["correct"])
        obs, _ = env.reset()   # 1-step env: always terminal

    metrics = [trainer.model_update(buf.sample(CFG.optim.batch_size)) for _ in range(10)]
    z0 = trainer.encoder(buf.sample(CFG.optim.batch_size)[0]).detach()
    b = trainer.behaviour_update(z0)

    assert all(np.isfinite(m["loss/total"]) for m in metrics)
    assert all(m["penalty/value"] >= 0 for m in metrics)
    assert np.isfinite(b["loss/policy"]) and np.isfinite(b["loss/value"])
    assert np.isfinite(np.mean(rewards))   # reconstruction reward is being collected
    # (no learning assertion — this only proves the env trains in the loop without NaNs)
