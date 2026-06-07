"""Bridge experiment penalty matrices: exactness, scale matching, positivity.

Load-bearing checks for scripts/bridge_experiment.py:
(1) the lap2_positive Gram form equals the autograd per-sample (Delta R)^2
    (torch.func.hessian trace, no RFF shortcuts) — the penalty is EXACT;
(2) all three penalty matrices share the expectation scale diag(|w|^4)
    (so any benchmark ordering is attributable to per-sample structure);
(3) lap2_positive is PSD; lap2_indefinite is genuinely indefinite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import torch

from mbrl.models.spectral import SpectralReward
from bridge_experiment import penalty_matrix


def _setup(d=3, M=64, N=200, seed=0):
    sr = SpectralReward(in_dim=d, n_features=M, sigma_w=1.0, seed=seed)
    sr.c = torch.randn(M, generator=torch.Generator().manual_seed(seed + 1))
    X = torch.randn(N, d, generator=torch.Generator().manual_seed(seed + 2))
    return sr, X


def test_lap2_positive_matches_autograd_laplacian():
    """c' (M*G) c / M == mean_n (Delta R(x_n))^2 computed by autograd."""
    sr, X = _setup()
    P = penalty_matrix("lap2_positive", sr, sr.features(X), seed=0)
    closed = float(sr.c @ P @ sr.c) / sr.M

    hess = torch.func.vmap(torch.func.hessian(lambda x: sr.predict(x[None])[0]))(X)
    lap = torch.einsum("nii->n", hess)
    autograd_val = float(lap.pow(2).mean())
    assert closed == pytest.approx(autograd_val, rel=1e-4), (closed, autograd_val)


def test_expectation_scale_matches_diag_w4():
    """E[M*G] = E[M*B] = diag(|w|^4): with many samples the diagonal of the
    Gram forms approaches the frobenius_diag arm's diagonal."""
    sr, _ = _setup(N=0)
    X = torch.randn(20000, sr.in_dim, generator=torch.Generator().manual_seed(5))
    Phi = sr.features(X)
    G = penalty_matrix("lap2_positive", sr, Phi, seed=0)
    B = penalty_matrix("lap2_indefinite", sr, Phi, seed=0)
    D = penalty_matrix("frobenius_diag", sr, Phi, seed=0)
    # diagonals agree within MC error (rel tolerance generous but meaningful)
    for Mtx in (G, B):
        ratio = torch.diag(Mtx) / torch.diag(D)
        assert float(ratio.mean()) == pytest.approx(1.0, abs=0.1), float(ratio.mean())


def test_positivity_structure():
    sr, X = _setup(N=500)
    Phi = sr.features(X)
    G = penalty_matrix("lap2_positive", sr, Phi, seed=0)
    B = penalty_matrix("lap2_indefinite", sr, Phi, seed=0)
    assert float(torch.linalg.eigvalsh(G).min()) > -1e-4      # PSD
    assert float(torch.linalg.eigvalsh(B).min()) < -1e-3      # indefinite
