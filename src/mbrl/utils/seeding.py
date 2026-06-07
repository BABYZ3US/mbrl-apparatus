"""Deterministic seeding across torch / numpy / python / env."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(device, seed: int) -> torch.Generator:
    """Device-local Generator; falls back to CPU where device generators are
    unsupported (e.g. some MPS versions). rademacher_like handles the move."""
    try:
        g = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        g = torch.Generator()
    g.manual_seed(seed)
    return g
