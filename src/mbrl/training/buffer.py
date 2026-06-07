"""Replay buffer with shard export/import — the Mode-B transport unit.

Local CPU collectors export shards (torch .pt) that are uploaded as W&B
artifacts; the Colab GPU trainer imports them. Also serves Mode A in-process.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0,
                 task_dim: int = 0):
        self.capacity, self.idx, self.full = capacity, 0, False
        self.task_dim = task_dim
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, action_dim), np.float32)
        self.rew = np.zeros((capacity,), np.float32)
        self.obs_next = np.zeros((capacity, obs_dim), np.float32)
        self.tau = np.zeros((capacity, task_dim), np.float32) if task_dim else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, act, rew, obs_next, tau=None):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i], self.obs_next[i] = obs, act, rew, obs_next
        if self.task_dim:
            self.tau[i] = tau
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        ix = self.rng.integers(0, len(self), batch_size)
        fields = [self.obs, self.act, self.rew, self.obs_next]
        if self.task_dim:
            fields.append(self.tau)
        return tuple(torch.as_tensor(x[ix]) for x in fields)

    # ---- Mode-B shards ----
    def export_shard(self, path: str | Path):
        n = len(self)
        torch.save({"obs": self.obs[:n], "act": self.act[:n],
                    "rew": self.rew[:n], "obs_next": self.obs_next[:n]}, path)

    def import_shard(self, path: str | Path):
        d = torch.load(path, weights_only=False)
        for i in range(len(d["rew"])):
            self.add(d["obs"][i], d["act"][i], d["rew"][i], d["obs_next"][i])
