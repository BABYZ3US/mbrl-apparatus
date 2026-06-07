# mbrl-curvature

Testing apparatus for curvature-regularized latent MBRL. Science:
`../mbrl_foundations_and_framework.md`. Operations: `PLAN.md`.
Publishability contract: `docs/claims_ledger.md` (what is defensible at each
tier — read before stating any claim). Architecture diagram with the closed-form
equations: `docs/architecture.svg`. Related work for the spectral stack:
`docs/related_work_spectral.md`. Roadmap: `docs/improvement_plan.md`.

## Setup (local)

```bash
cd mbrl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"          # add ,mujoco] on machines with MuJoCo
wandb login                          # academic account on the .edu email
make test                            # verify the penalty math before anything else
```

## Setup (Colab)

Open `notebooks/colab_launcher.ipynb` in Colab (Pro, A100 runtime), add your
`WANDB_API_KEY` to Colab Secrets, run all cells. Sessions are disposable:
checkpoints auto-push to W&B and `checkpoint.resume=auto` continues on relaunch.
Checkpoint lineages are scoped by config hash — a config change starts fresh
instead of resuming stale weights.

## Setup (any cloud, provider-agnostic)

```bash
pip install "skypilot[aws,gcp,lambda,runpod]"
sky jobs launch sky.yaml --env WANDB_API_KEY \
    --env EXP=spectral_auto --env ENV_NAME=halfcheetah --env SEED=0
```

Managed spot with auto-recovery; W&B is the rendezvous (PLAN.md Mode B).

## The spectral reward stack (closed-form path)

`spectral.enabled=true` replaces the MLP reward fit + Hutchinson penalty with an
RFF ridge ensemble whose H² penalty is exact (`src/mbrl/models/spectral.py`).
Four bandwidth modes, in increasing order of autonomy:

| `spectral.sigma_w` | behavior | status (ledger) |
|---|---|---|
| scalar (e.g. `0.5`) | single bandwidth | baseline |
| list (`[0.25,0.5,1,2]`) | sigma ladder: feature blocks per bandwidth | run 3: +33.7% supervised w/ poly |
| `auto` | SNR-calibrated ladder: rungs at measured sigma* × cal_mults | run 5 champion: +48.3% supervised |
| `learned` | per-block log-scales trained by reward-fit gradient | untested |

Band penalty: `spectral.poly` (lambda polynomial, validated) or
`spectral.weights_mode=snr` (explicit Wiener weights — measurement-grade,
lost the supervised head-to-head; see ledger run 4).

**Spectral rules (hard-won, see ledger):** never pair the spectral path with
step anneals or zero-touching schedules (closed-form refits have no inertia —
λ≈0 means the next refit is an unregularized interpolator); cap latent at
1× obs_dim (`model.latent_cap_mult=1`).

Dynamics: `model.dynamics=affine` (deterministic, default) or `gaussian`
(state probability transitions; mean stays affine-in-action to preserve R15).

## Daily commands

```bash
make test            # full suite
make bench           # supervised spectral benchmark (chunked, resumable)
make bridge          # bridge experiment cells (ledger runs 1-2)
make recipe          # recipe head-to-head incl. calibrated ladders (runs 3-5)
make dashboard       # results/dashboard.html from local runs
make figures         # regenerate figures from results/runs/ JSONL mirrors

# dev run (Pendulum; device=auto -> cuda > cpu; MPS ruled out — penalty double
# backward fails on MPS; re-gate with scripts/check_mps.py after torch upgrades)
python scripts/train.py

# experiment configs compose with +experiment= (note the plus)
python scripts/train.py +experiment=spectral_auto env=halfcheetah seed=0
python scripts/train.py +experiment=multienv env=walker2d seed=0

# THE spectral RL validation (5 arms: learned/auto/fixed ladder/single/MLP anchor)
python scripts/parallel_runs.py --preset colab_spectral --overrides env=halfcheetah --seeds 0 1 2

# original-recipe arms (fixed doses from docs/original_findings_report.md)
python scripts/parallel_runs.py --preset colab_recipe --overrides env=halfcheetah --seeds 0 1 2 --jobs 3
python scripts/parallel_runs.py --preset colab_control --overrides env=walker2d --seeds 0 --jobs 1
python scripts/parallel_runs.py --preset colab_estimator --overrides env=halfcheetah --seeds 0 1 2 --jobs 3

# local CPU science, no GPU needed (validation items 6-7)
python scripts/local_sweep.py --experiment stone --jobs 8
python scripts/local_sweep.py --experiment smoothness --jobs 8

# parallel grids: one process per core, same W&B project, grouped by arm
python scripts/parallel_runs.py --preset multitask_ablation --seeds 0 1 --jobs 6
python scripts/parallel_runs.py --preset schedule_ablation --seeds 0 1 2 --jobs 7

# multi-task zero-shot generalization (item 9)
python scripts/train_multitask.py                                # PendulumTarget, local
python scripts/train_multitask.py env=halfcheetah_vel            # Colab
python scripts/train_multitask.py penalty.schedule.lam0=0        # ablation arm

# Mode-B collection on local cores -> W&B artifact
python scripts/collect.py --env HalfCheetah-v5 --workers 8 --steps 50000 --upload

# figures from cloud runs
python scripts/make_figures.py --source wandb --project <entity>/mbrl-curvature --group multienv
```

## Non-negotiables (from the founding doc)

- Penalty is **isotropic**, in latent coords, 2-probe unbiased Hutchinson (R4/R16).
- **Never** penalize the policy Hessian (R10). Dynamics term optional, off by default.
- Always plot against **real env steps**, never gradient steps.
- s=1 spectral penalties diverge (R2) — there is no config option for them on purpose.
- Spectral path: smooth floored λ decay only; latent capped at 1× obs_dim.

## Repo map

```
configs/            base.yaml + experiment/ (+experiment=NAME) + env/
src/mbrl/models/    encoder, dynamics (affine|gaussian), spectral (RFF stack), policy
src/mbrl/training/  loop.py (Trainer), buffer, smoothing
src/mbrl/regularization/  hutchinson, schedule, transversality
scripts/            train, train_multitask, parallel_runs (presets), collect,
                    spectral_benchmark, bridge_experiment, make_dashboard, make_figures
docs/               claims_ledger (CONTRACT), original_findings_report (PRIMARY RECORD),
                    architecture.svg, related_work_spectral, improvement_plan
tests/              61 tests; checkpoint resume is bitwise (probe RNG included)
```
