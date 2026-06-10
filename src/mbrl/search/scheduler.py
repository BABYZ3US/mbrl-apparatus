"""W9 search-beyond-grids: the PURE scheduler core.

Three pieces, all deterministic and side-effect-free (the bridge server owns
launching/cancelling; the Studio drives ticks):

- ``sample_axes``: random-search sampling over typed axis distributions
  (choice / uniform / loguniform / int), seeded — the same (axes, n, seed)
  always proposes the same arms (resume-safe).
- ``decide_stops``: median-rule early stopping — at a comparison step, an arm
  whose metric is strictly below the median of all arms' values at that step
  stops. Conservative gates: no verdicts before ``min_arms`` have reported or
  before an arm has ``min_points`` points.
- ``next_actions``: the slot transition — given the search state, which queued
  arms to LAUNCH (respecting ``parallel``) and which running arms to STOP
  (from the median rule). The caller applies the actions and records statuses.

State shape (persisted by the server as JSON):
  {"name", "metric", "mode": "max"|"min", "parallel", "arms": [
      {"name", "overrides": {path: value}, "status":
       "queued"|"running"|"stopped"|"finished"|"failed"}]}
"""
from __future__ import annotations

import math
import random

_KINDS = ("choice", "uniform", "loguniform", "int")


def sample_axes(axes: list[dict], n: int, seed: int = 0) -> list[dict]:
    """``n`` override dicts {path: value}, deterministically from ``seed``.

    Axis forms: {"path", "kind": "choice", "values": [...]} |
    {"path", "kind": "uniform"|"loguniform", "low", "high"} |
    {"path", "kind": "int", "low", "high"} (inclusive bounds).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    for ax in axes:
        kind = str(ax.get("kind", ""))
        if kind not in _KINDS:
            raise ValueError(f"unknown axis kind '{kind}' (have {_KINDS})")
        if kind == "choice" and not ax.get("values"):
            raise ValueError(f"choice axis '{ax.get('path')}' has no values")
        if kind == "loguniform" and float(ax.get("low", 0)) <= 0:
            raise ValueError(f"loguniform axis '{ax.get('path')}' needs low > 0")
    out: list[dict] = []
    for _ in range(n):
        overrides: dict = {}
        for ax in axes:
            path, kind = str(ax["path"]), str(ax["kind"])
            if kind == "choice":
                overrides[path] = rng.choice(list(ax["values"]))
            elif kind == "uniform":
                overrides[path] = rng.uniform(float(ax["low"]), float(ax["high"]))
            elif kind == "loguniform":
                lo, hi = math.log(float(ax["low"])), math.log(float(ax["high"]))
                overrides[path] = math.exp(rng.uniform(lo, hi))
            else:  # int
                overrides[path] = rng.randint(int(ax["low"]), int(ax["high"]))
        out.append(overrides)
    return out


def _value_at(history: list, step: float) -> float | None:
    """The last value at or before ``step`` (None when nothing reported yet)."""
    best = None
    for s, v in history:
        if float(s) <= step:
            best = float(v)
    return best


def decide_stops(histories: dict, *, mode: str = "max", min_points: int = 3,
                 min_arms: int = 3) -> set[str]:
    """Median rule at the LATEST common step: arms strictly WORSE than the
    median of reporting arms stop. ``histories``: {arm: [(step, value), ...]}.
    """
    eligible = {a: h for a, h in histories.items() if len(h) >= min_points}
    if len(eligible) < min_arms:
        return set()
    compare_step = min(max(float(s) for s, _ in h) for h in eligible.values())
    vals = {a: _value_at(h, compare_step) for a, h in eligible.items()}
    vals = {a: v for a, v in vals.items() if v is not None}
    if len(vals) < min_arms:
        return set()
    ordered = sorted(vals.values())
    m = len(ordered)
    median = ordered[m // 2] if m % 2 else 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])
    if mode == "max":
        return {a for a, v in vals.items() if v < median}
    return {a for a, v in vals.items() if v > median}


def next_actions(state: dict, stops: set[str] | None = None) -> dict:
    """{"launch": [arm names], "stop": [arm names], "done": bool}.

    Launch fills free slots (parallel − running) from the queue in order;
    stop = the subset of ``stops`` that is actually running. done = nothing
    queued or running after the actions apply.
    """
    arms: list[dict] = list(state.get("arms", []))
    parallel = max(1, int(state.get("parallel", 1)))
    stops = stops or set()
    running = [a["name"] for a in arms if a.get("status") == "running"]
    stop = [a for a in running if a in stops]
    free = parallel - (len(running) - len(stop))
    launch = [a["name"] for a in arms if a.get("status") == "queued"][:max(0, free)]
    remaining = sum(1 for a in arms
                    if a.get("status") in ("queued", "running")
                    and a["name"] not in stop) - len(launch)
    # launched arms are still live; done only when nothing is left after this tick
    done = (remaining + len(launch)) == 0
    return {"launch": launch, "stop": stop, "done": done}
