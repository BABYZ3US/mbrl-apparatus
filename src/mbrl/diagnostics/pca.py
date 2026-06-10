"""pca — principal component analysis via SVD (pure numpy).

The standard latent/data diagnostic: how many effective dimensions does a
representation use? The explained-variance spectrum (scree) pairs directly with
the spectral latent-cap rule (k = obs_dim): a latent whose variance concentrates
in fewer components than its width is over-provisioned; a flat spectrum says the
cap binds. Deterministic (SVD, no RNG); centered; components signed so each
row's largest-|.| entry is positive (a stable convention for tests + display).
"""
from __future__ import annotations

import numpy as np


class PCA:
    """Fit/transform/inverse_transform with explained-variance ratios."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None          # (k, d)
        self.explained_variance_: np.ndarray | None = None  # (k,)
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("PCA.fit wants X of shape (n>=2, d), got %s" % (X.shape,))
        n, d = X.shape
        k = min(self.n_components or d, d, n - 1)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD of the centered data: rows of Vt are the principal axes
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (s ** 2) / (n - 1)
        total = var.sum()
        comps = Vt[:k]
        # sign convention: each component's largest-|entry| is positive
        signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
        signs[signs == 0] = 1.0
        self.components_ = comps * signs[:, None]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / total if total > 0 else np.zeros(k)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_

    def _check(self) -> None:
        if self.components_ is None:
            raise RuntimeError("PCA not fitted")


def pca_diagnostics(X: np.ndarray, n_components: int | None = None) -> dict:
    """JSON-ready PCA summary: scree + cumulative + effective dimension.

    effective_dim = exp(entropy of the variance distribution) — the standard
    participation-ratio-style count of how many components really carry variance.
    """
    p = PCA(n_components).fit(X)
    evr = p.explained_variance_ratio_
    nz = evr[evr > 1e-12]
    eff = float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0
    return {
        "n_rows": int(np.asarray(X).shape[0]),
        "n_features": int(np.asarray(X).shape[1]),
        "explained_variance_ratio": [float(v) for v in evr],
        "cumulative": [float(v) for v in np.cumsum(evr)],
        "effective_dim": eff,
    }
