"""Additive Gaussian observation noise — turns the encoder into a genuine noisy
channel for the latent-as-channel / information-bottleneck experiment
(docs/channel_capacity_formalization_2026-06-11.md).

On clean state-based obs the representation channel is not a bottleneck (z can
copy x); injecting obs noise forces the encoder to compress+denoise, which is the
regime where the rate–distortion / Wiener-filter story (R14) provably bites. The
noise is the channel's input noise: x̃ = x + σ·ε, ε~N(0,I). Each vector-env worker
gets its own seeded RNG so the noise stream is reproducible and decorrelated.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class GaussianObsNoise(gym.ObservationWrapper):
    """x̃ = x + σ·scale·ε. `relative` scales σ by a per-dim running |x| estimate
    (dimensionless σ across MuJoCo's heterogeneous obs dims); else raw units."""

    def __init__(self, env, sigma: float, seed: int = 0, relative: bool = True):
        super().__init__(env)
        self.sigma = float(sigma)
        self.relative = bool(relative)
        self._rng = np.random.default_rng(seed)
        self._scale = None   # per-dim EMA of |x| (relative mode)

    def observation(self, obs):
        if self.sigma <= 0.0:
            return obs
        obs = np.asarray(obs)
        if self.relative:
            a = np.abs(obs)
            self._scale = a if self._scale is None else 0.99 * self._scale + 0.01 * a
            scale = np.maximum(self._scale, 1e-3)
        else:
            scale = 1.0
        noisy = obs + self.sigma * scale * self._rng.standard_normal(obs.shape)
        return noisy.astype(obs.dtype, copy=False)
