"""Declarative layer list -> nn.Sequential (W7: the Studio's generic NN bricks).

The Studio compiles a wired NN-layer chain into ``model.encoder_net``:
``[{"kind": "conv2d", "out_channels": 32, ...}, {"kind": "activation", ...}]``.
This module is the torch-side consumer. Lazy modules carry the input-dim
bookkeeping (LazyLinear/LazyConv*/LazyBatchNorm*), so the list needs no
in-feature plumbing; parameters materialize on the first forward.

Honesty status: implemented + tested; the Trainer does NOT consume
``model.encoder_net`` yet (the validators on both sides warn). Wiring it into
the Encoder is its own receipted arm.
"""
from __future__ import annotations

import torch.nn as nn

_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
_BN = {1: nn.LazyBatchNorm1d, 2: nn.LazyBatchNorm2d, 3: nn.LazyBatchNorm3d}
_CONV = {"conv1d": nn.LazyConv1d, "conv2d": nn.LazyConv2d, "conv3d": nn.LazyConv3d}


def build_layer(layer: dict) -> nn.Module:
    kind = str(layer.get("kind", ""))
    if kind == "linear":
        return nn.LazyLinear(int(layer["out_features"]))
    if kind in _CONV:
        return _CONV[kind](int(layer["out_channels"]), int(layer.get("kernel", 3)),
                           stride=int(layer.get("stride", 1)),
                           padding=int(layer.get("padding", 0)))
    if kind == "layer_norm":
        return nn.LayerNorm(int(layer["dim"]))
    if kind == "batch_norm":
        dims = int(layer.get("dims", 1))
        if dims not in _BN:
            raise ValueError(f"batch_norm dims must be 1/2/3, got {dims}")
        return _BN[dims]()
    if kind == "activation":
        act = str(layer.get("act", "relu"))
        if act not in _ACTS:
            raise ValueError(f"unknown activation '{act}' (have {sorted(_ACTS)})")
        return _ACTS[act]()
    if kind == "dropout":
        return nn.Dropout(float(layer.get("p", 0.1)))
    if kind == "flatten":
        return nn.Flatten()
    raise ValueError(f"unknown layer kind '{kind}'")


def build_net(layers: list[dict]) -> nn.Sequential:
    """The whole chain. Empty lists are an authoring error, not a silent no-op."""
    if not layers:
        raise ValueError("encoder_net is empty — nothing to build")
    return nn.Sequential(*(build_layer(l) for l in layers))
