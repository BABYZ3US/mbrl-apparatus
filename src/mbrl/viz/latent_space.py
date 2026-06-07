"""Latent-space inspection: PCA/UMAP embedding colored by reward; encoder-metric
curvature diagnostic hook (R16 — `encoder_curvature.py` lineage)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def embed_latents(encoder, obs: torch.Tensor, rewards: np.ndarray, method: str = "pca"):
    z = encoder(obs).cpu().numpy()
    if z.shape[-1] <= 2:
        emb = z[:, :2]
    elif method == "umap":
        import umap
        emb = umap.UMAP(n_components=2).fit_transform(z)
    else:
        z = z - z.mean(0)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        emb = z @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=rewards, s=6, cmap="coolwarm")
    fig.colorbar(sc, label="reward")
    ax.set_title(f"latent space ({method}), k={z.shape[-1]}")
    return fig
