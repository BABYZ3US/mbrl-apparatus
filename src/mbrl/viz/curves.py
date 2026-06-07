"""Training-curve figures: return vs env steps (per-seed + mean±CI), losses, lambda(t).

`make_figures.py` calls these against W&B API data; they also work on local
dicts of arrays. Style matches the parent project's figures.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

STYLE = {"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
         "axes.spines.top": False, "axes.spines.right": False}


def return_vs_steps(runs: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "",
                    ax=None, ci: float = 1.0):
    """runs: {label: (steps, returns[seeds, T])} — mean ± ci*sem band per label."""
    with plt.rc_context(STYLE):
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        for label, (steps, rets) in runs.items():
            mean = rets.mean(0)
            sem = rets.std(0) / np.sqrt(max(1, rets.shape[0]))
            ax.plot(steps, mean, label=f"{label} (n={rets.shape[0]})")
            ax.fill_between(steps, mean - ci * sem, mean + ci * sem, alpha=0.2)
        ax.set_xlabel("real env steps")  # always env steps, never grad steps
        ax.set_ylabel("episode return")
        ax.set_title(title)
        ax.legend()
        return ax


def lambda_and_penalty(steps, lam, pen, ax=None):
    with plt.rc_context(STYLE):
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, pen, label=r"$\|\nabla^2 \hat R\|_F^2$ (Hutchinson)")
        ax.set_yscale("log"); ax.set_xlabel("model updates")
        ax2 = ax.twinx()
        ax2.plot(steps, lam, color="tab:red", ls="--", label=r"$\lambda(t)$")
        ax.legend(loc="upper right")
        return ax


def horizon_variance(horizons, variances_by_lam: dict[float, np.ndarray], ax=None):
    """R15: imagined-return variance vs rollout horizon, one curve per lambda."""
    with plt.rc_context(STYLE):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))
        for lam, v in sorted(variances_by_lam.items()):
            ax.plot(horizons, v, marker="o", label=rf"$\lambda={lam:g}$")
        ax.set_yscale("log")
        ax.set_xlabel("imagination horizon H")
        ax.set_ylabel("imagined-return variance")
        ax.legend()
        return ax


def stone_rate(ns, errors, s: float, d: float, ax=None):
    """Validation item 7: log-log error vs n with predicted slope -2s/(2s+d)."""
    with plt.rc_context(STYLE):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4.5))
        ax.loglog(ns, errors, "o-", label="measured")
        slope = -2 * s / (2 * s + d)
        ref = errors[0] * (np.asarray(ns) / ns[0]) ** slope
        ax.loglog(ns, ref, "k--", label=rf"predicted $n^{{{slope:.2f}}}$")
        ax.set_xlabel("n samples"); ax.set_ylabel("test error")
        ax.legend()
        return ax
