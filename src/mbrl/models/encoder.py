"""Encoder obs -> z in R^k, with EMA-stabilized target copy (framework 2.1).

k defaults small per the compact-latent lean (R17) — a soft prior to be tested
via the latent-dim sweep, not assumed.
"""
from __future__ import annotations

import copy

import torch
from torch import nn, Tensor


def mlp(sizes, act=nn.SiLU, out_act=None, init_std=0.02):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.normal_(lin.weight, std=init_std)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 4, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.net = mlp([obs_dim] + [hidden] * depth + [latent_dim])
        # normalized latent: keeps z scale stable for dynamics/reward/penalty
        self.net.append(nn.LayerNorm(latent_dim))
        self.latent_dim = latent_dim

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class EMAEncoder:
    """Exponential-moving-average copy used to encode targets (z'); stabilizes
    latent training (latent-MBRL experiment 2.3 required this)."""

    def __init__(self, encoder: Encoder, decay: float = 0.995):
        self.ema = copy.deepcopy(encoder).requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, encoder: Encoder):
        for p_ema, p in zip(self.ema.parameters(), encoder.parameters()):
            p_ema.lerp_(p, 1.0 - self.decay)

    @torch.no_grad()
    def __call__(self, obs: Tensor) -> Tensor:
        return self.ema(obs)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)
