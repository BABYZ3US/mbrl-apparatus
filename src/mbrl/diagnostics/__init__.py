"""mbrl.diagnostics — model/data diagnostics (PCA, cross-validation).

Pure numpy — no torch, no yaml — safe to import inside the studio seal. Heavy
diagnostics run offline (scripts/diagnose.py) and write JSON artifacts under
results/diagnostics/, which the bridge serves via pull.diagnostics (the same
artifact -> reader -> verb -> panel pattern as the reward surfaces).
"""
from .pca import PCA, pca_diagnostics
from .crossval import kfold_indices, kfold_ridge, ridge_fit, ridge_r2

__all__ = ["PCA", "pca_diagnostics", "kfold_indices", "kfold_ridge",
           "ridge_fit", "ridge_r2"]
