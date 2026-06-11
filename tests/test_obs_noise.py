"""GaussianObsNoise channel wrapper: σ=0 passthrough; σ>0 perturbs, seeded-reproducible."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
gym = pytest.importorskip("gymnasium")
from mbrl.envs.obs_noise import GaussianObsNoise


def _env(sigma, seed=0, relative=True):
    return GaussianObsNoise(gym.make("Pendulum-v1"), sigma, seed=seed, relative=relative)


def test_sigma_zero_is_passthrough():
    e = _env(0.0)
    obs, _ = e.reset(seed=0)
    # observation() is the identity at sigma=0
    assert np.array_equal(e.observation(obs), obs)


def test_sigma_positive_perturbs_and_is_seeded():
    raw = gym.make("Pendulum-v1")
    obs, _ = raw.reset(seed=0)
    a = GaussianObsNoise(gym.make("Pendulum-v1"), 0.5, seed=11, relative=False)
    b = GaussianObsNoise(gym.make("Pendulum-v1"), 0.5, seed=11, relative=False)
    na, nb = a.observation(obs), b.observation(obs)
    assert not np.allclose(na, obs)              # actually perturbed
    assert np.allclose(na, nb)                   # same seed -> same noise (reproducible)
    c = GaussianObsNoise(gym.make("Pendulum-v1"), 0.5, seed=99, relative=False)
    assert not np.allclose(na, c.observation(obs))  # different seed -> different noise


def test_step_returns_noisy_obs():
    e = _env(0.5, seed=3)
    e.reset(seed=0)
    obs, r, term, trunc, info = e.step(e.action_space.sample())
    assert obs.shape == e.observation_space.shape and np.all(np.isfinite(obs))
