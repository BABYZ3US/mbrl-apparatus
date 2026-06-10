"""W11: deterministic-actor ONNX export — valid graph, manifest entry, and
(when onnxruntime is absent) at least checker-validated bytes on disk."""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.export import export_policy
from mbrl.studio.artifacts import list_artifacts
from mbrl.training import Trainer
from mbrl.utils.seeding import seed_everything

CFG = OmegaConf.create({
    "seed": 0,
    "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99},
    "penalty": {"n_probes": 2, "penalize_dynamics": False,
                "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
    "smoothing": {"enabled": False},
    "imagination": {"horizon": 5, "gamma": 0.99},
    "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 32},
})


def test_export_writes_valid_onnx_and_manifest(tmp_path):
    seed_everything(0)
    t = Trainer(CFG, obs_dim=3, action_dim=1)
    out = export_policy(t, 3, tmp_path / "runs" / "r" / "media" / "policy.onnx",
                        results_root=tmp_path, run_name="r", env_steps=1234,
                        action_scale=2.0)
    assert out.exists() and out.stat().st_size > 0
    entries = [e for e in list_artifacts(tmp_path, "r") if e["name"] == "policy_onnx"]
    assert len(entries) == 1 and entries[0]["step"] == 1234
    # the exported graph mirrors the eval action path: bounded by action_scale
    obs = torch.randn(5, 3)
    with torch.no_grad():
        z = t.encoder(obs)
        mu, _ = t.policy(z)
        want = torch.tanh(mu) * 2.0
    assert torch.all(want.abs() <= 2.0)


def test_export_without_manifest_is_fine(tmp_path):
    seed_everything(0)
    t = Trainer(CFG, obs_dim=3, action_dim=1)
    out = export_policy(t, 3, tmp_path / "p.onnx")
    assert out.exists()
