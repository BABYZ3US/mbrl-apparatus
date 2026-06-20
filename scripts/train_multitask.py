"""Multi-task zero-shot generalization experiment (validation item 9).

THIN SHIM. The training logic now lives in `scripts/train.py` (`run_multitask` /
`run_training`) so there is ONE training driver. This file stays a runnable Hydra
entry point with `config_name="multitask"` because callers spawn it by filename:
`scripts/parallel_runs.py` and the `train_multitask.py` presets in
`configs/presets.yaml`. It flips the `multitask` switch and delegates.

Train one task-conditioned world model + policy on n_train tasks (tau values);
periodically evaluate zero-shot on held-out taus — interpolation (inside the
training range) and extrapolation (outside), reported separately. The science
ablation is the same command with penalty.schedule.lam0=0.

  python scripts/train_multitask.py                                   # Pendulum family, CPU
  python scripts/train_multitask.py env=halfcheetah_vel               # Colab
  python scripts/train_multitask.py penalty.schedule.lam0=0 seed=0    # ablation arm

Equivalent (single entry point): python scripts/train.py multitask=true ...
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))   # so `import train` works regardless of cwd

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# The shared driver. eval_zero_shot is re-exported for any caller that imported it
# from this module historically.
from train import run_training, eval_zero_shot  # noqa: F401


@hydra.main(config_path="../configs", config_name="multitask", version_base=None)
def main(cfg: DictConfig):
    # Force the multitask branch regardless of what `multitask.yaml` carries, then
    # hand off to the single shared driver in train.py. open_dict so we can add the
    # key even under Hydra's struct mode.
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.multitask = True
    run_training(cfg)


if __name__ == "__main__":
    main()
