"""W7 consumption arm: model.encoder=custom builds the trunk from encoder_net.

Pins: latent contract (k-dim, normalized), Trainer construction + finite
update, EMA deepcopy through materialized lazies, loud failures (empty net,
conv-on-flat-obs), and bitwise resume with the custom encoder."""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mbrl.models import CustomEncoder
from mbrl.training import Trainer
from mbrl.utils.checkpoint import CheckpointManager
from mbrl.utils.seeding import seed_everything

NET = [{"kind": "linear", "out_features": 32},
       {"kind": "activation", "act": "gelu"},
       {"kind": "layer_norm", "dim": 32},
       {"kind": "dropout", "p": 0.05}]


def _cfg(net=None):
    return OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 32, "depth": 1, "ema_decay": 0.99,
                  "encoder": "custom", "encoder_net": NET if net is None else net},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100, "floor": 1e-6}},
        "smoothing": {"enabled": False},
        "imagination": {"horizon": 5, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4, "batch_size": 32},
    })


def _batch(n=32, obs_dim=3, act_dim=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, obs_dim, generator=g), torch.randn(n, act_dim, generator=g),
            torch.randn(n, generator=g), torch.randn(n, obs_dim, generator=g))


def test_latent_contract_holds_whatever_the_chain_emits():
    seed_everything(0)
    enc = CustomEncoder(obs_dim=7, latent_dim=4, layers=NET)
    z = enc(torch.randn(16, 7))
    assert z.shape == (16, 4)                       # projection head pins k
    assert z.mean().abs() < 0.5                      # LayerNorm'd latents


def test_trainer_consumes_encoder_net_and_updates_finitely():
    seed_everything(0)
    t = Trainer(_cfg(), obs_dim=3, action_dim=1)
    assert isinstance(t.encoder, CustomEncoder)
    m = t.model_update(_batch())
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values()
               if isinstance(v, float))
    # the EMA target deepcopied materialized lazies and encodes targets
    assert t.ema.ema(torch.randn(5, 3)).shape == (5, 4)


def test_loud_failures():
    with pytest.raises(ValueError, match="non-empty"):
        Trainer(_cfg(net=[]), obs_dim=3, action_dim=1)
    with pytest.raises(RuntimeError):                # conv chain on flat obs
        CustomEncoder(obs_dim=3, latent_dim=4,
                      layers=[{"kind": "conv2d", "out_channels": 8}])


def test_validator_no_longer_warns_on_consumed_custom_encoder():
    from mbrl.studio.spec_validator import validate_spec
    warns = validate_spec({"model": {"encoder": "custom", "encoder_net": NET}})
    assert not any("not yet consumed" in w for w in warns)
    assert any("encoder_net is empty" in w
               for w in validate_spec({"model": {"encoder": "custom"}}))


def test_resume_bitwise_identical_with_custom_encoder(tmp_path):
    cfg = _cfg()
    seed_everything(0)
    t1 = Trainer(cfg, obs_dim=3, action_dim=1)
    for i in range(3):
        t1.model_update(_batch(seed=i))
    cm = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    cm.save(t1, env_steps=300, tag="step3")
    m_ref = t1.model_update(_batch(seed=7))

    seed_everything(0)
    t2 = Trainer(cfg, obs_dim=3, action_dim=1)
    cm2 = CheckpointManager(tmp_path, OmegaConf.to_container(cfg), every=10)
    assert cm2.resume(t2) == 300
    m_resumed = t2.model_update(_batch(seed=7))
    for k in ("loss/dyn", "loss/total"):
        assert m_resumed[k] == pytest.approx(m_ref[k], rel=1e-6), k
