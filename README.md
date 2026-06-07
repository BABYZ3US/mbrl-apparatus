# mbrl-curvature

Testing apparatus for curvature-regularized latent MBRL. Science:
`../mbrl_foundations_and_framework.md`. Operations: `PLAN.md`.

## Setup (local)

```bash
cd mbrl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"          # add ,mujoco] on machines with MuJoCo
wandb login                          # academic account on the .edu email
pytest                               # verify the penalty math before anything else
```

## Setup (Colab)

Open `notebooks/colab_launcher.ipynb` in Colab (Pro, A100 runtime), add your
`WANDB_API_KEY` to Colab Secrets, run all cells. Sessions are disposable:
checkpoints auto-push to W&B and `checkpoint.resume=auto` continues on relaunch.

## Daily commands

```bash
# dev run (Pendulum; device=auto -> cuda > cpu; MPS ruled out — penalty double
# backward fails on MPS as of torch 2.x; re-check with scripts/check_mps.py after upgrades)
python scripts/train.py

# experiment from the validation queue
python scripts/train.py +experiment=multienv env=walker2d seed=0

# local CPU science, no GPU needed (validation items 6-7)
python scripts/local_sweep.py --experiment stone --jobs 8
python scripts/local_sweep.py --experiment smoothness --jobs 8

# GPU-lean Colab plan (doses fixed from docs/original_findings_report.md — no sweeps):
# Phase 0 (regression gate, ~3 GPU-h): recipe on HalfCheetah, 3 seeds, expect ~+98±23.
#   python scripts/parallel_runs.py --preset colab_recipe --overrides env=halfcheetah --seeds 0 1 2 --jobs 3
# Phase 1 (breadth): per new env, recipe 3 seeds + control 1 seed:
#   python scripts/parallel_runs.py --preset colab_recipe --overrides env=walker2d --seeds 0 1 2 --jobs 3
#   python scripts/parallel_runs.py --preset colab_control --overrides env=walker2d --seeds 0 --jobs 1
# Optional head-to-head (sec.3 claim): clamped trace vs Frobenius, same dose:
#   python scripts/parallel_runs.py --preset colab_estimator --overrides env=halfcheetah --seeds 0 1 2 --jobs 3
# (tiny models: 3 processes share one A100 fine — env stepping is the bottleneck)

# parallel grids: one process per core, same W&B project, grouped by arm
python scripts/parallel_runs.py --preset multitask_ablation --seeds 0 1 --jobs 6
python scripts/parallel_runs.py --preset schedule_ablation --seeds 0 1 2 --jobs 7
python scripts/parallel_runs.py --script train.py --seeds 0 1 2 3 4 \
    --overrides experiment.name=baseline --jobs 5

# multi-task zero-shot generalization (item 9)
python scripts/train_multitask.py                                # PendulumTarget, local
python scripts/train_multitask.py env=halfcheetah_vel            # Colab
python scripts/train_multitask.py penalty.schedule.lam0=0        # ablation arm
python scripts/train_multitask.py penalty.include_task=false     # mechanism ablation

# Mode-B collection on local cores -> W&B artifact
python scripts/collect.py --env HalfCheetah-v5 --workers 8 --steps 50000 --upload

# regenerate figures — local (offline, from results/runs/ JSONL mirrors) or cloud
python scripts/make_figures.py                                   # local, all runs + sweeps
python scripts/make_figures.py --group multienv                  # local, one experiment
python scripts/make_figures.py --source wandb --project <entity>/mbrl-curvature --group multienv
python scripts/make_figures.py --source auto --project <entity>/mbrl-curvature  # local, then cloud
```

## Non-negotiables (from the founding doc)

- Penalty is **isotropic**, in latent coords, 2-probe unbiased Hutchinson (R4/R16).
- **Never** penalize the policy Hessian (R10). Dynamics term optional, off by default.
- Always plot against **real env steps**, never gradient steps.
- s=1 spectral penalties diverge (R2) — there is no config option for them on purpose.
