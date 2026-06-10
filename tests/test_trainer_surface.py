"""Trainer hooks for the M4 reward-surface + Hessian export (training/loop.py).

Builds a real Trainer (CPU) and exercises reward_surface_payload /
reward_hessian_eigs / reward_concat_fn on BOTH the MLP and spectral reward paths.
Numerical correctness of the underlying math is covered by test_viz_surface_export;
this pins the WIRING (shapes, and the spectral-vs-MLP switch).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from mbrl.training import Trainer

MLP_CFG = OmegaConf.create({
    "seed": 0,
    "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
    "penalty": {"n_probes": 2, "penalize_dynamics": False,
                "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
    "smoothing": {"enabled": True, "sigma": 1.5},
    "imagination": {"horizon": 5, "gamma": 0.99},
    "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 32},
})


def _spectral_cfg():
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "symlog_reward": True, "reward_heads": 2, "latent_cap_mult": 1},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "spectral": {"enabled": True, "n_features": 64, "sigma_w": "auto",
                     "refit_every": 5, "cache_size": 256, "heads": 2,
                     "poly": {"degrees": [2], "coefs": [1.0], "shifts": [0]}},
        "smoothing": {"enabled": True, "sigma": 1.5},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    })


def _fake_batch(n=32, obs_dim=3, act_dim=1, seed=123):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_reward_input_dim_and_concat_fn_mlp():
    t = Trainer(MLP_CFG, obs_dim=3, action_dim=1, device="cpu")
    assert t.reward_input_dim() == 5             # k=4 + action=1 + task=0
    y = t.reward_concat_fn()(torch.zeros(7, 5))
    assert y.shape == (7,)                         # scalar per row


def test_reward_surface_payload_shape_mlp():
    t = Trainer(MLP_CFG, obs_dim=3, action_dim=1, device="cpu")
    p = t.reward_surface_payload(plane=(0, 1), n=9, extent=2.0, step=1000, run="r")
    assert len(p["z"]) == 9 and len(p["z"][0]) == 9
    assert len(p["curv"]) == 9
    assert p["plane"] == {"u": 0, "v": 1}
    assert p["budget"] >= 0.0 and p["step"] == 1000 and p["run"] == "r"


def test_reward_hessian_eigs_mlp():
    t = Trainer(MLP_CFG, obs_dim=3, action_dim=1, device="cpu")
    eigs = t.reward_hessian_eigs()
    assert len(eigs) == t.reward_input_dim()
    assert list(eigs) == sorted(eigs, reverse=True)


def test_spectral_before_refit_is_flat_zero_surface():
    t = Trainer(_spectral_cfg(), obs_dim=3, action_dim=1, device="cpu")
    assert t.spec_heads == []                      # no refit yet -> zeros fn
    p = t.reward_surface_payload(n=7)
    assert np.allclose(np.array(p["z"]), 0.0)
    assert p["budget"] == 0.0


def test_spectral_after_refit_surface_and_hessian_finite():
    t = Trainer(_spectral_cfg(), obs_dim=3, action_dim=1, device="cpu")
    for i in range(8):                             # trigger >=1 refit
        t.model_update(_fake_batch(seed=100 + i))
    assert t.spec_refits >= 1 and t.spec_heads
    p = t.reward_surface_payload(n=9)
    assert len(p["z"]) == 9 and np.isfinite(np.array(p["z"])).all()
    eigs = t.reward_hessian_eigs()
    assert len(eigs) == t.reward_input_dim() and np.isfinite(eigs).all()
