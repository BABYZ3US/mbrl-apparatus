"""Every experiment yaml must COMPOSE against base and honor the house rules —
config rot caught at test time, not at launch (authored with campaign 2)."""
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_composes(name):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="base", overrides=[f"+experiment={name}"])
    assert cfg.experiment.name, name


def test_ensemble_spectral_is_the_champion_bridge():
    """The campaign-2 arm: spectral reward stack + affine ensemble, both house
    rules intact — and the Trainer actually BUILDS + UPDATES on it."""
    import torch
    from mbrl.training import Trainer
    from mbrl.models.ensemble import EnsembleAffineDynamics
    from mbrl.utils.seeding import seed_everything

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="base", overrides=[
            "+experiment=ensemble_spectral",
            # tiny shapes for the construction smoke; science values untouched
            "model.latent_dim=3", "model.hidden=32", "model.depth=1",
            "spectral.n_features=64",
        ])
    assert cfg.spectral.enabled and str(cfg.model.dynamics) == "affine"
    assert int(cfg.algo.dynamics_ensemble) == 5
    assert str(cfg.penalty.schedule.kind) == "cuberoot"
    assert float(cfg.penalty.schedule.floor) > 0          # never zero-touching
    assert int(cfg.model.latent_cap_mult) == 1            # spectral latent rule

    seed_everything(0)
    t = Trainer(cfg, obs_dim=3, action_dim=1)
    assert isinstance(t.dynamics, EnsembleAffineDynamics)
    assert t.spec_enabled                                  # the spectral path is live
    g = torch.Generator().manual_seed(0)
    batch = (torch.randn(32, 3, generator=g), torch.randn(32, 1, generator=g),
             torch.randn(32, generator=g), torch.randn(32, 3, generator=g))
    m = t.model_update(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values()
               if isinstance(v, float))
    assert "dyn/disagreement" in m                         # both stacks coexist
