# CLAUDE.md — mbrl-curvature

Curvature-regularized latent MBRL testing apparatus. Science contract:
`../mbrl_foundations_and_framework.md` (the founding doc — read it before changing any
math) plus `docs/claims_ledger.md` (user's tiered inventory of what is proved /
confirmed / NOT confirmed — the publishability contract; multi-kernel benefit is
explicitly unvalidated, never claim it). Ops plan: `PLAN.md`. Result tags like R4/R10
refer to the founding doc's ledger.

## Hard rules (from the founding doc — do not "improve" these away)

- The curvature penalty is **isotropic** (R16): never weight Hessian eigendirections,
  never anisotropize. It is applied **in latent coords** on detached `(z, a)`.
- **Never penalize the policy Hessian** (R10). There is deliberately no config option
  for it. Dynamics term is optional and off by default.
- Use the **unbiased 2-probe** Hutchinson estimator (R4/R5); the 1-probe squared trace
  is biased and underperforms — `laplacian_trace_penalty` rejects `n_probes < 2` on purpose.
- No s=1 (Jacobian/spectral) penalties — they diverge (R2). Don't add one.
- Penalty math runs in **fp32** even under bf16 autocast (`hutchinson.py` enforces this).
- Plot/log sample efficiency against **real env steps**, never gradient steps.

## Layout

- `src/mbrl/regularization/` — the heart: `hutchinson.py` (HVP penalty),
  `schedule.py` (λ(t), cuberoot anneal R12), `transversality.py` (angle diagnostic R8)
- `src/mbrl/models/` — encoder (+EMA), affine-in-action dynamics (∂²T/∂a²=0, R15),
  reward (the regularization target), policy/value
- `src/mbrl/training/` — `loop.py` (Trainer; model + behaviour updates), `returns.py`
  (Dreamer λ-returns — behaviour learning is λ-returns through the model with an EMA
  target value net, NOT SAC; chosen deliberately for cost), `buffer.py` (replay +
  Mode-B shards + optional task storage), `smoothing.py` (DreamSmooth)
- `src/mbrl/envs/tasks.py` — multi-task families (HalfCheetahVel, PendulumTarget):
  shared dynamics, task-conditioned reward/policy/value via τ. Zero-shot eval splits
  held-out τ into interpolation vs extrapolation — report separately, smoothness only
  promises interpolation. `penalty.include_task=true` puts τ in the Hessian coords
  (smooth between-task interpolation — the generalization mechanism under test);
  its ablations are include_task=false and lam0=0. Entry: `scripts/train_multitask.py`.
- `src/mbrl/utils/checkpoint.py` — atomic, RNG-complete resume; W&B artifact push.
  The Colab-disconnect defense — keep resume bitwise-exact. Checkpoints are scoped
  by config hash (`checkpoints/<run>/<hash>/`): a config change starts a fresh
  lineage under resume=auto (never crashes, old lineages preserved); only an
  explicit checkpoint path raises on hash mismatch.
- `scripts/` — `train.py` (Hydra entry), `collect.py` (local CPU collectors),
  `local_sweep.py` (synthetic experiments, items 6–7), `make_figures.py`
- `configs/` — Hydra: `base.yaml` + `env/` + `experiment/` (one per validation item)
- `notebooks/colab_launcher.ipynb` — GPU side (Colab Pro)

- `scripts/parallel_runs.py` — local grid launcher: one process per core, each run
  pinned to single-threaded math libs (OMP/MKL=1 — process-level parallelism, don't
  remove that), unique `hydra.run.dir` per run (two hydra runs starting in the same
  second otherwise collide). Arms get distinct `experiment.name` => distinct W&B
  group => the dashboard and make_figures aggregate seeds per arm automatically.

## Commands

```bash
pip install -e ".[dev]"            # setup (add ,mujoco on MuJoCo machines)
pytest                             # ALWAYS run before and after touching src/
python scripts/train.py            # dev run: Pendulum, CPU, ~no deps beyond core
python scripts/train.py +experiment=multienv env=walker2d seed=0
python scripts/local_sweep.py --experiment stone --jobs 8   # no GPU needed
```

## Testing conventions

- `tests/test_hvp.py` checks the estimator against **analytic Hessians** (quadratics).
  Any change to `hutchinson.py` must keep these passing unmodified.
- `tests/test_checkpoint.py::test_resume_bitwise_identical` is the resume guarantee:
  if you add any RNG or stateful component to `Trainer`, add it to
  `state_dict`/`load_state_dict` or this test will (correctly) fail. This already
  caught the Hutchinson probe generator once.
- `tests/test_smoke.py` is the gate before spending Colab time: full loop on
  Pendulum, CPU, <1 min.

## Environment notes

- Compute split: GPU (Colab Pro) = model/behaviour learning; local CPU (joblib) =
  env collection, synthetic experiments, figures. Mode B syncs via W&B artifacts
  (project `mbrl-curvature`), never direct networking.
- Local Mac (M2): **MPS does not work for this project** — the penalty's double
  backward fails on MPS (verified on the user's M2 via `scripts/check_mps.py`).
  `device=auto` therefore resolves cuda > cpu only; local runs are CPU. Do not
  re-enable MPS in auto-detection. If a future torch release fixes double-backward
  coverage, the path back is: `check_mps.py` passes -> user opts in with
  `device=mps` explicitly. Never enable PYTORCH_ENABLE_MPS_FALLBACK for real runs.
  The Hutchinson probe generator may be CPU-side under MPS (`make_generator`
  fallback) — intentional, keeps probe streams reproducible across backends.
- W&B run naming: group = experiment name, name = `{experiment}-{env}-s{seed}`,
  one run per (experiment, env, seed).
- Every run also writes a local JSONL mirror (`results/runs/<name>/metrics.jsonl`,
  via `utils/metrics_logger.py`) of the same keys it sends to W&B. Figures regenerate
  from either source (`make_figures.py --source local|wandb|auto`). If you add a
  logged metric that figures need, add it to KEYS in `scripts/make_figures.py`.
- `torch>=2.4` assumed; plain `torch.autograd.grad` double-backward (no torch.func
  dependency in the penalty — keep it that way for Colab version drift).
- Hydra writes job dirs under `outputs/` (gitignored); checkpoints under
  `checkpoints/` (gitignored; canonical copies are W&B artifacts).

## When changing experiment configs

Each `configs/experiment/*.yaml` maps 1:1 to a validation item in PLAN.md §6 / founding
doc Part 5. If you add an experiment, add the config, a PLAN.md row, and a W&B sweep
grid comment in the YAML.
