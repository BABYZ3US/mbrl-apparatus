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
        self.done = np.zeros((capacity,), dtype=bool)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, act, rew, obs_next, tau=None, done: bool = False):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i], self.obs_next[i] = obs, act, rew, obs_next
        if self.task_dim:
            self.tau[i] = tau
        self.done[i] = bool(done)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        ix = self.rng.integers(0, len(self), batch_size)
        fields = [self.obs, self.act, self.rew, self.obs_next]
        if self.task_dim:
            fields.append(self.tau)
        return tuple(torch.as_tensor(x[ix]) for x in fields)

    def _valid_window_starts(self, length: int) -> np.ndarray:
        """Start indices of length-row windows that are entirely contiguous in
        time and contain no interior episode boundary.

        Validity rule (no-wrap variant): a window occupies the inclusive row
        range [start, start + length - 1] and is valid iff
          (1) it lies wholly within [0, len) -> 0 <= start <= len - length;
          (2) when self.full it does not cross the circular write head self.idx
              (the slot about to be overwritten / the newest|oldest seam), i.e.
              self.idx is NOT in the interior range [start + 1, start + length - 1];
          (3) no row in [start, start + length - 2] is a done==True row
              (a window may END on an episode boundary but never SPAN one).
        """
        n = len(self)
        if length <= 0 or n < length:
            return np.empty((0,), dtype=np.intp)
        # Candidate non-wrapping starts: 0 .. n - length (inclusive).
        starts = np.arange(0, n - length + 1, dtype=np.intp)
        valid = np.ones(starts.shape, dtype=bool)
        # (2) Exclude windows whose interior crosses the circular write head.
        if self.full:
            # head falls inside (start+1 .. start+length-1)  <=>
            # start in (idx - length + 1 .. idx - 1)
            valid &= ~((starts >= self.idx - length + 1) & (starts <= self.idx - 1))
        # (3) Exclude windows with a done in any but the last row.
        # done_interior[s] is True if any of rows [s .. s+length-2] is done.
        if length >= 2:
            done = self.done[:n].astype(bool)
            # prefix[k] = number of done rows in done[0:k]
            prefix = np.concatenate(([0], np.cumsum(done)))
            # count of done in [start, start + length - 2] inclusive
            interior_done = prefix[starts + (length - 1)] - prefix[starts]
            valid &= interior_done == 0
        return starts[valid]

    def sample_windows(self, batch_size: int, length: int):
        """Sample `batch_size` consecutive-step windows of `length` rows each.

        Returns tensors shaped on the leading window axis:
          obs/obs_next [B, length, obs_dim], act [B, length, action_dim],
          rew [B, length], and (only if self.task_dim) tau [B, length, task_dim].
        Each window is `length` CONSECUTIVE stored rows that are contiguous in
        time and span no episode boundary (a done may appear only as the final
        row). See _valid_window_starts for the exact validity rule.
        """
        starts = self._valid_window_starts(length)
        if starts.size == 0:
            raise ValueError(
                "sample_windows: no valid windows of length "
                f"{length} (buffer len={len(self)}, full={self.full}); "
                "buffer too small or all spans broken by episode boundaries"
            )
        chosen = starts[self.rng.integers(0, starts.size, batch_size)]
        # Build [B, length] index matrix of consecutive rows per window.
        offsets = np.arange(length, dtype=np.intp)
        ix = chosen[:, None] + offsets[None, :]
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
