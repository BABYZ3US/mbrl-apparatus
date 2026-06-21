"""Replay buffer with shard export/import — the Mode-B transport unit.

Local CPU collectors export shards (torch .pt) that are uploaded as W&B
artifacts; the Colab GPU trainer imports them. Also serves Mode A in-process.

Two layouts:
  * num_lanes == 1 (default): a single flat circular buffer (legacy, byte-exact).
  * num_lanes  > 1: one contiguous ring PER env-lane, so consecutive-window
    sampling stays single-trajectory even when collecting from many parallel
    envs (the transformer-in-the-loop arm). Each lane gets capacity//num_lanes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _valid_starts_1d(done, n: int, write_idx: int, full: bool, length: int) -> np.ndarray:
    """Start indices of length-row windows in a 1-D ring of `n` valid rows.

    A window occupies inclusive [start, start+length-1] and is valid iff
      (1) it lies wholly within [0, n)  ->  0 <= start <= n - length;
      (2) when `full`, it does not cross the circular write head `write_idx`
          (write_idx not interior to [start+1, start+length-1]);
      (3) no row in [start, start+length-2] is a done==True row (a window may
          END on an episode boundary but never SPAN one).
    `done` is the (>= n) boolean done array for this ring (lane-local or flat).
    """
    if length <= 0 or n < length:
        return np.empty((0,), dtype=np.intp)
    starts = np.arange(0, n - length + 1, dtype=np.intp)
    valid = np.ones(starts.shape, dtype=bool)
    if full:
        valid &= ~((starts >= write_idx - length + 1) & (starts <= write_idx - 1))
    if length >= 2:
        d = done[:n].astype(bool)
        prefix = np.concatenate(([0], np.cumsum(d)))
        interior_done = prefix[starts + (length - 1)] - prefix[starts]
        valid &= interior_done == 0
    return starts[valid]


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0,
                 task_dim: int = 0, num_lanes: int = 1):
        self.num_lanes = max(1, int(num_lanes))
        self.task_dim = task_dim
        if self.num_lanes > 1:
            # per-lane layout: each env's trajectory is a contiguous ring, so a
            # window of consecutive rows is guaranteed single-trajectory.
            self.cap = max(1, int(capacity) // self.num_lanes)
            self.capacity = self.cap * self.num_lanes
            self.obs = np.zeros((self.num_lanes, self.cap, obs_dim), np.float32)
            self.act = np.zeros((self.num_lanes, self.cap, action_dim), np.float32)
            self.rew = np.zeros((self.num_lanes, self.cap), np.float32)
            self.obs_next = np.zeros((self.num_lanes, self.cap, obs_dim), np.float32)
            self.tau = (np.zeros((self.num_lanes, self.cap, task_dim), np.float32)
                        if task_dim else None)
            self.done = np.zeros((self.num_lanes, self.cap), dtype=bool)
            self.lidx = np.zeros(self.num_lanes, dtype=np.int64)   # per-lane write head
            self.lfull = np.zeros(self.num_lanes, dtype=bool)
        else:
            # flat circular buffer (legacy path; byte-exact).
            self.capacity, self.idx, self.full = capacity, 0, False
            self.obs = np.zeros((capacity, obs_dim), np.float32)
            self.act = np.zeros((capacity, action_dim), np.float32)
            self.rew = np.zeros((capacity,), np.float32)
            self.obs_next = np.zeros((capacity, obs_dim), np.float32)
            self.tau = np.zeros((capacity, task_dim), np.float32) if task_dim else None
            self.done = np.zeros((capacity,), dtype=bool)
        self.rng = np.random.default_rng(seed)

    def _lane_len(self, L: int) -> int:
        return self.cap if self.lfull[L] else int(self.lidx[L])

    def __len__(self):
        if self.num_lanes > 1:
            return int(sum(self._lane_len(L) for L in range(self.num_lanes)))
        return self.capacity if self.full else self.idx

    def add(self, obs, act, rew, obs_next, tau=None, done: bool = False, lane: int = 0):
        if self.num_lanes > 1:
            L = int(lane) % self.num_lanes
            i = int(self.lidx[L])
            self.obs[L, i], self.act[L, i] = obs, act
            self.rew[L, i], self.obs_next[L, i] = rew, obs_next
            if self.task_dim:
                self.tau[L, i] = tau
            self.done[L, i] = bool(done)
            self.lidx[L] = (i + 1) % self.cap
            self.lfull[L] = self.lfull[L] or self.lidx[L] == 0
            return
        i = self.idx
        self.obs[i], self.act[i], self.rew[i], self.obs_next[i] = obs, act, rew, obs_next
        if self.task_dim:
            self.tau[i] = tau
        self.done[i] = bool(done)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        if self.num_lanes > 1:
            # pooled iid over all filled (lane, row) cells
            lens = np.array([self._lane_len(L) for L in range(self.num_lanes)])
            tot = int(lens.sum())
            flat = self.rng.integers(0, tot, batch_size)
            csum = np.concatenate(([0], np.cumsum(lens)))
            lane_sel = np.searchsorted(csum, flat, side="right") - 1
            row_sel = flat - csum[lane_sel]
            fields = [self.obs, self.act, self.rew, self.obs_next]
            if self.task_dim:
                fields.append(self.tau)
            return tuple(torch.as_tensor(x[lane_sel, row_sel]) for x in fields)
        ix = self.rng.integers(0, len(self), batch_size)
        fields = [self.obs, self.act, self.rew, self.obs_next]
        if self.task_dim:
            fields.append(self.tau)
        return tuple(torch.as_tensor(x[ix]) for x in fields)

    def _valid_window_starts(self, length: int) -> np.ndarray:
        """Flat-layout (num_lanes==1) valid window starts. See _valid_starts_1d."""
        return _valid_starts_1d(self.done, len(self), self.idx, self.full, length)

    def sample_windows(self, batch_size: int, length: int):
        """Sample `batch_size` consecutive-step windows of `length` rows each, each
        single-trajectory (contiguous, spanning no episode boundary — a done may
        appear only as the final row). Returns tensors:
          obs/obs_next [B, length, obs_dim], act [B, length, action_dim],
          rew [B, length], tau [B, length, task_dim] (iff task_dim).
        With num_lanes>1, windows are drawn within a single env-lane.
        """
        if self.num_lanes > 1:
            lanes_arr, starts_arr = [], []
            for L in range(self.num_lanes):
                s = _valid_starts_1d(self.done[L], self._lane_len(L),
                                     int(self.lidx[L]), bool(self.lfull[L]), length)
                if s.size:
                    lanes_arr.append(np.full(s.shape, L, dtype=np.intp))
                    starts_arr.append(s)
            if not starts_arr:
                raise ValueError(
                    f"sample_windows: no valid windows of length {length} across "
                    f"{self.num_lanes} lanes (each lane too short or done-broken)")
            lanes_all = np.concatenate(lanes_arr)
            starts_all = np.concatenate(starts_arr)
            pick = self.rng.integers(0, starts_all.size, batch_size)
            lane_sel = lanes_all[pick]                       # [B]
            start_sel = starts_all[pick]                     # [B]
            offs = np.arange(length, dtype=np.intp)
            rows = start_sel[:, None] + offs[None, :]        # [B, length]
            lane_col = lane_sel[:, None]                     # [B, 1] -> broadcasts
            fields = [self.obs, self.act, self.rew, self.obs_next]
            if self.task_dim:
                fields.append(self.tau)
            return tuple(torch.as_tensor(x[lane_col, rows]) for x in fields)
        starts = self._valid_window_starts(length)
        if starts.size == 0:
            raise ValueError(
                "sample_windows: no valid windows of length "
                f"{length} (buffer len={len(self)}, full={self.full}); "
                "buffer too small or all spans broken by episode boundaries")
        chosen = starts[self.rng.integers(0, starts.size, batch_size)]
        offsets = np.arange(length, dtype=np.intp)
        ix = chosen[:, None] + offsets[None, :]
        fields = [self.obs, self.act, self.rew, self.obs_next]
        if self.task_dim:
            fields.append(self.tau)
        return tuple(torch.as_tensor(x[ix]) for x in fields)

    # ---- Mode-B shards (flat layout only) ----
    def export_shard(self, path: str | Path):
        n = len(self)
        torch.save({"obs": self.obs[:n], "act": self.act[:n],
                    "rew": self.rew[:n], "obs_next": self.obs_next[:n]}, path)

    def import_shard(self, path: str | Path):
        d = torch.load(path, weights_only=False)
        for i in range(len(d["rew"])):
            self.add(d["obs"][i], d["act"][i], d["rew"][i], d["obs_next"][i])
