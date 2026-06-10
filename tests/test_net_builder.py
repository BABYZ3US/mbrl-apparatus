"""net_builder: the Studio's NN-brick vocabulary -> nn.Sequential (W7)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mbrl.models.net_builder import build_layer, build_net


def test_mlp_chain_shapes():
    net = build_net([
        {"kind": "linear", "out_features": 64},
        {"kind": "activation", "act": "gelu"},
        {"kind": "layer_norm", "dim": 64},
        {"kind": "dropout", "p": 0.1},
        {"kind": "linear", "out_features": 8},
    ])
    out = net(torch.randn(5, 17))            # lazy in-dims materialize here
    assert out.shape == (5, 8)


def test_conv2d_chain_shapes():
    net = build_net([
        {"kind": "conv2d", "out_channels": 16, "kernel": 3, "stride": 2, "padding": 1},
        {"kind": "batch_norm", "dims": 2},
        {"kind": "activation", "act": "relu"},
        {"kind": "flatten"},
        {"kind": "linear", "out_features": 10},
    ])
    out = net(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_conv3d_and_conv1d_build():
    assert build_layer({"kind": "conv3d", "out_channels": 4}) is not None
    net = build_net([{"kind": "conv1d", "out_channels": 8, "kernel": 5, "padding": 2}])
    assert net(torch.randn(2, 3, 100)).shape == (2, 8, 100)


def test_determinism_under_seed():
    layers = [{"kind": "linear", "out_features": 32}]
    x = torch.randn(3, 7)
    torch.manual_seed(0)
    a = build_net(layers)(x)
    torch.manual_seed(0)
    b = build_net(layers)(x)
    assert torch.equal(a, b)


def test_rejections_are_loud():
    with pytest.raises(ValueError, match="unknown layer kind"):
        build_layer({"kind": "transformer"})
    with pytest.raises(ValueError, match="unknown activation"):
        build_layer({"kind": "activation", "act": "swishish"})
    with pytest.raises(ValueError, match="dims must be"):
        build_layer({"kind": "batch_norm", "dims": 4})
    with pytest.raises(ValueError, match="empty"):
        build_net([])


def test_validator_warns_on_unconsumed_custom_encoder():
    from mbrl.studio.spec_validator import validate_spec
    spec = {"model": {"encoder": "custom",
                      "encoder_net": [{"kind": "linear", "out_features": 8}]}}
    warns = validate_spec(spec)
    assert any("not yet consumed" in w and "custom" in w for w in warns)
    warns2 = validate_spec({"model": {"encoder": "custom"}})
    assert any("encoder_net is empty" in w for w in warns2)
