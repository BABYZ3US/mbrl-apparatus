"""Diagnostics: PCA (known-axes recovery) + K-fold CV (partition laws, probe power)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.diagnostics import PCA, pca_diagnostics, kfold_indices, kfold_ridge


# ---------------- PCA ----------------

def test_pca_recovers_known_principal_axis():
    """Anisotropic Gaussian stretched along a known direction -> PC1 ~ that axis."""
    rng = np.random.default_rng(0)
    axis = np.array([3.0, 4.0]) / 5.0
    t = rng.normal(size=(2000, 1)) * 10.0          # big variance along `axis`
    noise = rng.normal(size=(2000, 2)) * 0.1
    X = t * axis + noise
    p = PCA().fit(X)
    assert abs(float(np.dot(p.components_[0], axis))) > 0.999
    assert p.explained_variance_ratio_[0] > 0.99


def test_pca_ratios_sum_to_one_and_descend():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    p = PCA().fit(X)
    evr = p.explained_variance_ratio_
    assert abs(evr.sum() - 1.0) < 1e-9
    assert all(evr[i] >= evr[i + 1] - 1e-12 for i in range(len(evr) - 1))


def test_pca_roundtrip_full_rank_is_exact():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 4))
    p = PCA().fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X, atol=1e-9)


def test_pca_truncation_reduces_reconstruction_error_monotonically():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 5)) * np.array([4, 3, 2, 1, 0.5])
    errs = []
    for k in (1, 2, 3, 4, 5):
        p = PCA(k).fit(X)
        errs.append(float(((p.inverse_transform(p.transform(X)) - X) ** 2).mean()))
    assert all(errs[i] > errs[i + 1] - 1e-12 for i in range(4))
    assert errs[-1] < 1e-9                          # full rank: exact


def test_pca_diagnostics_payload_and_effective_dim():
    rng = np.random.default_rng(4)
    # ~2 strong directions out of 6 -> effective_dim between 1 and 3
    X = rng.normal(size=(500, 6)) * np.array([5, 5, 0.1, 0.1, 0.1, 0.1])
    d = pca_diagnostics(X)
    assert d["n_rows"] == 500 and d["n_features"] == 6
    assert abs(d["cumulative"][-1] - 1.0) < 1e-9
    assert 1.5 < d["effective_dim"] < 3.0


def test_pca_rejects_degenerate_input():
    with pytest.raises(ValueError):
        PCA().fit(np.zeros((1, 3)))


# ---------------- K-fold ----------------

def test_kfold_partitions_disjoint_balanced_deterministic():
    folds = kfold_indices(103, 5, seed=9)
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val.tolist()) == list(range(103))     # covers exactly once
    sizes = [len(v) for _, v in folds]
    assert max(sizes) - min(sizes) <= 1                     # balanced
    for train, val in folds:
        assert np.intersect1d(train, val).size == 0         # disjoint
        assert len(train) + len(val) == 103
    again = kfold_indices(103, 5, seed=9)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))


def test_kfold_rejects_bad_k():
    with pytest.raises(ValueError):
        kfold_indices(10, 1)
    with pytest.raises(ValueError):
        kfold_indices(3, 4)


def test_ridge_cv_finds_linear_signal():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 8))
    w = rng.normal(size=8)
    y = X @ w + 0.05 * rng.normal(size=400)
    rep = kfold_ridge(X, y, k=5, alpha=1e-3, seed=0)
    assert rep["r2_mean"] > 0.95
    assert rep["r2_std"] < 0.05
    assert len(rep["r2_per_fold"]) == 5


def test_ridge_cv_reports_no_signal_on_noise():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 8))
    y = rng.normal(size=300)                        # independent of X
    rep = kfold_ridge(X, y, k=5, alpha=1.0, seed=0)
    assert rep["r2_mean"] < 0.1                     # honest: ~0 out of sample
