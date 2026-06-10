"""crossval — deterministic K-fold cross-validation with a ridge probe.

K-fold CV is the battle-tested generalization diagnostic: fit on K-1 folds,
score on the held-out fold, report the spread. The default probe is closed-form
ridge regression (one linear solve per fold — no torch, no iterations, exactly
reproducible), scored by R^2. Use it to ask e.g. "do the spectral features
predict the target linearly, out of sample?" — the honest baseline any learned
model must beat.

Determinism: folds come from a seeded permutation; same (n, k, seed) -> same
folds, always (the resume-bitwise discipline applied to diagnostics).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def kfold_indices(n: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train_idx, val_idx)] x k — a seeded, balanced, disjoint partition."""
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n (got k=%d, n=%d)" % (k, n))
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        val = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((train, val))
    return out


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge with intercept: W = (Xc'Xc + aI)^-1 Xc'yc. y may be (n,) or (n, m)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y2 = y[:, None] if y.ndim == 1 else y
    xm, ym = X.mean(axis=0), y2.mean(axis=0)
    Xc, yc = X - xm, y2 - ym
    d = X.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)
    b = ym - xm @ W
    return W, b


def ridge_r2(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    """R^2 of the ridge prediction (multi-output: variance-weighted mean)."""
    y = np.asarray(y, dtype=np.float64)
    y2 = y[:, None] if y.ndim == 1 else y
    pred = np.asarray(X, dtype=np.float64) @ W + b
    ss_res = float(((y2 - pred) ** 2).sum())
    ss_tot = float(((y2 - y2.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def kfold_ridge(X: np.ndarray, y: np.ndarray, k: int = 5, alpha: float = 1.0,
                seed: int = 0) -> dict:
    """JSON-ready K-fold ridge-probe report: per-fold R^2 + mean/std."""
    scores = []
    for train, val in kfold_indices(len(X), k, seed):
        W, b = ridge_fit(np.asarray(X)[train], np.asarray(y)[train], alpha)
        scores.append(ridge_r2(np.asarray(X)[val], np.asarray(y)[val], W, b))
    return {
        "probe": "ridge", "alpha": float(alpha), "folds": int(k), "seed": int(seed),
        "r2_per_fold": [float(s) for s in scores],
        "r2_mean": float(np.mean(scores)),
        "r2_std": float(np.std(scores)),
    }


def kfold_score(X: np.ndarray, y: np.ndarray,
                fit: Callable[[np.ndarray, np.ndarray], object],
                score: Callable[[object, np.ndarray, np.ndarray], float],
                k: int = 5, seed: int = 0) -> list[float]:
    """Generic K-fold: any fit/score pair (the extension seam for torch probes)."""
    out = []
    for train, val in kfold_indices(len(X), k, seed):
        model = fit(np.asarray(X)[train], np.asarray(y)[train])
        out.append(float(score(model, np.asarray(X)[val], np.asarray(y)[val])))
    return out
