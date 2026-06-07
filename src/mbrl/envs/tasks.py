"""Multi-task families with scalar task parameters and train/held-out splits.

Design: dynamics are SHARED across tasks (same physics); only the reward varies
with tau. The world model is task-agnostic; reward/policy/value are
task-conditioned. Zero-shot generalization = evaluate on held-out tau values
never seen in training. Hypothesis under test: the H^2 penalty extended over
the tau coordinate forces smooth between-task interpolation, improving
zero-shot transfer — and the effect should survive ablation against lambda=0.

Families:
  - HalfCheetahVel: run at target velocity v*; reward = -|v_x - v*| - ctrl cost.
    (Classic meta-RL family; needs gymnasium[mujoco].)
  - PendulumTarget: hold target angle theta*; local smoke-test family, no MuJoCo.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


class TaskWrapper(gym.Wrapper):
    """Base: exposes .tau (np.ndarray, task_dim) and rewrites the reward."""
    task_dim = 1

    def __init__(self, env, tau: float):
        super().__init__(env)
        self.tau = np.array([tau], dtype=np.float32)

    def set_task(self, tau: float):
        self.tau = np.array([tau], dtype=np.float32)


class HalfCheetahVel(TaskWrapper):
    """Target-velocity HalfCheetah. tau = v* (m/s). Dense, smooth in tau."""

    def __init__(self, tau: float = 1.0, ctrl_cost: float = 0.05, **kw):
        super().__init__(gym.make("HalfCheetah-v5", **kw), tau)
        self.ctrl_cost = ctrl_cost

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        r = -abs(info["x_velocity"] - float(self.tau[0])) \
            - self.ctrl_cost * float(np.square(action).sum())
        return obs, r, term, trunc, info


class PendulumTarget(TaskWrapper):
    """Hold theta = tau (radians from upright). Local family, no MuJoCo."""

    def __init__(self, tau: float = 0.0, **kw):
        super().__init__(gym.make("Pendulum-v1", **kw), tau)

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        cos_t, sin_t, thdot = obs
        theta = np.arctan2(sin_t, cos_t)
        d = np.arctan2(np.sin(theta - self.tau[0]), np.cos(theta - self.tau[0]))
        r = -(d ** 2 + 0.1 * thdot ** 2 + 0.001 * float(np.square(action).sum()))
        return obs, float(r), term, trunc, info


FAMILIES = {"halfcheetah_vel": HalfCheetahVel, "pendulum_target": PendulumTarget}


def make_task_env(family: str, tau: float, **kw):
    return FAMILIES[family](tau=tau, **kw)


def task_split(family: str, n_train: int = 8, seed: int = 0):
    """Deterministic train/held-out task sets.

    Held-out covers BOTH interpolation (inside the training range) and
    extrapolation (outside) — report them separately; smoothness arguments
    only promise interpolation."""
    rng = np.random.default_rng(seed)
    if family == "halfcheetah_vel":
        lo, hi = 0.5, 3.0
        train = np.round(np.linspace(lo, hi, n_train), 3)
        interp = np.round((train[:-1] + train[1:]) / 2, 3)[:: max(1, (n_train - 1) // 4)]
        extrap = np.array([lo - 0.3, hi + 0.5], dtype=float)
    elif family == "pendulum_target":
        lo, hi = -1.0, 1.0  # radians from upright
        train = np.round(np.linspace(lo, hi, n_train), 3)
        interp = np.round((train[:-1] + train[1:]) / 2, 3)[:: max(1, (n_train - 1) // 4)]
        extrap = np.array([lo - 0.4, hi + 0.4], dtype=float)
    else:
        raise ValueError(f"unknown family {family}")
    rng.shuffle(train)
    return {"train": train.tolist(), "interp": interp.tolist(), "extrap": extrap.tolist()}
