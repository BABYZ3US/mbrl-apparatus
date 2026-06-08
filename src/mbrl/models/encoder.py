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


class VAEEncoder(nn.Module):
    """Run-10 encoder: q(z|obs) = N(mu, diag sigma^2) + MLP decoder.

    Why (vs the aux-grounded deterministic encoder): reconstruction + KL give
    the encoder a SELF-CONTAINED training signal (the 2026-06-08 HalfCheetah
    collapse is structurally impossible here), and the KL pull toward N(0, I)
    makes the latent's scale/shape near-stationary — the spectral basis
    (sigma*) stops chasing a moving coordinate system. Pre-registered
    criterion: sigma* drift / recalibration collapse vs the aux arm.

    forward(): rsample while training, mu at eval; the EMA target copy is
    forced deterministic (mu) via the `deterministic` flag — noisy dynamics
    targets would leak encoder noise into the NLL.

    The decoder is a swappable module: MLP for state observations (current).
    A transposed-conv decoder over the "normal map" (the 2 x k grid of
    (mu, sigma) parameters) is the PIXEL-task variant — deconvolution earns
    its keep only when the decoded target has spatial structure; imposing
    neighborhoods on a 17-dim state vector would be invented structure."""

    LOGVAR_RANGE = (-8.0, 4.0)

    def __init__(self, obs_dim: int, latent_dim: int = 4, hidden: int = 256,
                 depth: int = 2):
        super().__init__()
        self.trunk = mlp([obs_dim] + [hidden] * depth + [hidden])
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)
        for head in (self.mu_head, self.logvar_head):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)
        self.decoder = mlp([latent_dim] + [hidden] * depth + [obs_dim])
        self.latent_dim = latent_dim
        self.deterministic = False   # set True on the EMA target copy

    def moments(self, obs: Tensor) -> tuple:
        h = self.trunk(obs)
        return self.mu_head(h), self.logvar_head(h).clamp(*self.LOGVAR_RANGE)

    def forward(self, obs: Tensor) -> Tensor:
        mu, lv = self.moments(obs)
        if self.deterministic or not self.training:
            return mu
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def losses(self, obs: Tensor) -> tuple:
        """(recon MSE, KL to N(0,I), z sample) — one forward per update."""
        mu, lv = self.moments(obs)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) \
            if self.training else mu
        recon = torch.nn.functional.mse_loss(self.decoder(z), obs)
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).sum(-1).mean()
        return recon, kl, z


class EMAEncoder:
    """Exponential-moving-average copy used to encode targets (z'); stabilizes
    latent training (latent-MBRL experiment 2.3 required this)."""

    def __init__(self, encoder: Encoder, decay: float = 0.995):
        self.ema = copy.deepcopy(encoder).requires_grad_(False)
        if hasattr(self.ema, "deterministic"):   # VAE copy: mu targets only
            self.ema.deterministic = True
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
