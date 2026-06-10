"""W11: the trained policy as a deployable ONNX artifact.

``export_policy`` wraps encoder + policy into ONE module computing the
DETERMINISTIC action from a raw observation (obs -> z -> tanh(mu) * scale —
the eval-time action path, no sampling so the graph is ONNX-clean), exports
it (obs batch dim dynamic), validates with onnx.checker, and upserts the
manifest entry ('policy_onnx', latest-wins like checkpoints).

In-ENGINE inference (the Studio's infer.load/run verbs) additionally needs an
ONNX runtime GDExtension on the Godot side — a PM install gate, same as Lua
was. This artifact is the deployment half that exists regardless.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..studio.artifacts import record_artifact


class _DeterministicActor(nn.Module):
    """obs -> deterministic action (the eval path, ONNX-traceable)."""

    def __init__(self, encoder: nn.Module, policy: nn.Module, action_scale: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        self.action_scale = float(action_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        mu, _log_std = self.policy(z)
        return torch.tanh(mu) * self.action_scale


def export_policy(trainer, obs_dim: int, out_path, *, results_root=None,
                  run_name: str | None = None, env_steps: int = 0,
                  action_scale: float = 1.0) -> Path:
    """Write <out_path>.onnx; optionally record the manifest entry when
    results_root + run_name are given. Returns the path."""
    actor = _DeterministicActor(trainer.encoder, trainer.policy, action_scale)
    was_training = actor.training
    actor.eval()                                # VAE encoders: mu path, no sampling
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, int(obs_dim))
    with torch.no_grad():
        torch.onnx.export(actor, dummy, str(out),
                          input_names=["obs"], output_names=["action"],
                          dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                          dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(str(out)))
    if was_training:
        actor.train()
    if results_root is not None and run_name:
        record_artifact(results_root, run_name, {
            "name": "policy_onnx", "type": "onnx", "step": int(env_steps),
            "path": str(out)})
    return out
