# MBRL Testing Apparatus — Project Plan

> **SUPERSEDED IN PART (2026-06-08):** the compute split below (Colab Pro GPU
> + local CPU) is retired. Training/inference are cloud-only via the sealed
> image (`docs/remote_execution.md`); local = tests, the supervised nightly
> loop, and W&B log/artifact analysis. Sections on W&B rendezvous,
> checkpointing, and Mode-B artifact flow remain accurate.

Operational plan for the curvature-regularized MBRL project. The science lives in
`../mbrl_foundations_and_framework.md`; this document covers the machinery: compute split,
tooling, logging, snapshots, visualization, and the experiment pipeline.

---

## 1. Compute architecture: local CPU + Colab GPU

The workload splits cleanly along the MBRL loop's natural seam:

**GPU-bound (Colab Pro, A100/L4).** Model learning — encoder, latent dynamics, reward model,
policy/value updates, and the Hutchinson HVP penalty (2 extra backward passes per batch, R4).
Imagined rollouts in latent space are batched tensor ops → GPU. This is `scripts/train.py`,
launched from `notebooks/colab_launcher.ipynb`.

**Local GPU (Apple M2, MPS) — ruled out for now.** Tested on the M2 via
`scripts/check_mps.py`: the penalty's double backward fails on MPS, so local training
is CPU-only (`device=auto` resolves cuda > cpu). This is fine — local runs are tests,
synthetic experiments, and Pendulum-class prototyping; all heavy GPU work is Colab's
job anyway. If a future torch release fixes MPS double-backward, `check_mps.py` is the
gate to opt back in with explicit `device=mps`.

**CPU-bound (local, parallelized).** Real-environment interaction (MuJoCo is CPU-only and
single-threaded per env — the classic parallelization target), seed-level parallelism for
cheap environments (Pendulum sweeps run entirely on local CPU), offline analysis of logged
runs, figure generation, and the synthetic experiments (Stone-rate curves, smoothness sweeps —
validation items 6–7 are small-model studies that don't need GPU). This is
`scripts/collect.py`, `scripts/local_sweep.py`, `scripts/analyze.py`, using
`multiprocessing` / `joblib` with one env instance per process.

**Two operating modes.**

- *Mode A — all-in-Colab (default for MuJoCo).* Colab runs the full loop; env stepping uses
  a `gymnasium.vector.AsyncVectorEnv` across the Colab VM's CPUs while the GPU trains. The
  simplest correct setup; latency between collection and training is zero.
- *Mode B — split pipeline (for heavy sweeps).* Local machine runs many CPU workers doing
  env interaction / synthetic experiments / analysis; Colab does GPU-heavy training. The two
  sides never talk directly — they rendezvous through **W&B artifacts** (and optionally Google
  Drive): local workers upload replay-buffer shards / sweep results as artifacts; Colab pulls
  them, trains, pushes checkpoints back as artifacts; local pulls checkpoints for analysis
  and plotting. Asynchronous, restart-tolerant, no networking between machines required.

**Colab Pro specifics.** Sessions still die; design for it: checkpoint every N model updates
*and* on SIGTERM; auto-resume from the latest W&B checkpoint artifact on relaunch; runs are
idempotent given (config, seed, resume step). Prefer A100 for the 5-seed MuJoCo suite, L4/T4
fine for Pendulum-class work. Mixed precision (bf16) on by default for model learning —
but the HVP penalty computes in fp32 (second derivatives are noise-sensitive; this is a
config flag, validated in the smoke test).

## 2. Stack

- **PyTorch ≥ 2.4** — `torch.func` (`jvp`/`vjp`/`grad`) gives clean Hessian-vector products
  for the Hutchinson estimator; `torch.compile` for the training step.
- **Gymnasium + MuJoCo** (`gymnasium[mujoco]`) — HalfCheetah-v5, Walker2d, Ant, Humanoid,
  Pendulum. **D4RL successor: Minari** for the offline-RL stress test (validation item 2).
- **Hydra + OmegaConf** — config composition; every experiment in Part 5 is a config, not a
  code branch. `configs/experiment/*.yaml` override `configs/base.yaml`.
- **Weights & Biases** — logging, sweeps, artifacts (see §3). Academic account on the .edu email.
- **joblib / multiprocessing** — local CPU parallelism.
- **matplotlib + seaborn / plotly** — static figures + interactive inspection (see §5).
- **pytest** — unit tests for the math-critical pieces (see §7).
- Pinned in `pyproject.toml`; Colab installs via `pip install -e .` from the repo. Same
  package, both machines — no Colab-only code paths.

## 3. Logging: W&B layout

- **Project** `mbrl-curvature`; one run per (experiment, env, seed); group = experiment name,
  job_type = {train, collect, analyze}.
- **Per-step metrics:** L_fit components, penalty value, λ(t), gradient variance, HVP probe
  variance (probes logged separately to support validation item 3), imagined-return mean/var
  vs horizon (R15), transversality angle α between reward and dynamics Hessians (R8),
  episode return, env steps (the sample-efficiency x-axis — always log real env steps, not
  gradient steps).
- **Artifacts:** checkpoints (`model-{run}-{step}`), replay-buffer shards, figure bundles.
  Artifacts are the Mode-B transport layer.
- **Sweeps:** W&B sweeps drive the ablation grids (probe count, λ schedule, latent dim k,
  penalty target R/R+T). Local CPU agents and Colab GPU agents can join the same sweep —
  this is how the hybrid parallelism actually gets used day-to-day.
- TensorBoard event files also written locally (`results/tb/`) as a free fallback; W&B picks
  them up via `sync_tensorboard=True`.

## 4. Snapshots

`src/mbrl/utils/checkpoint.py` — a single `CheckpointManager`:

- Atomic save (write tmp, rename) of: all module state_dicts (encoder + EMA copy, dynamics,
  reward, policy, value), optimizers, λ-schedule state, replay-buffer cursor, RNG states
  (torch/numpy/python), env step count, config hash.
- Cadence: every `ckpt_every` updates + on exit signal; keep last 3 + best-by-eval-return +
  permanent milestones every 100k env steps.
- Each save optionally pushed as a W&B artifact (size-gated; buffers stored separately).
- `--resume auto` fetches the newest artifact for the run id and restores exactly —
  the core defense against Colab disconnects.
- Config hash check on resume: refuse to resume a checkpoint under a different config.

## 5. Visualization

`src/mbrl/viz/` produces both live dashboards (W&B panels, defined once in
`viz/wandb_panels.py` as a saved workspace) and publication figures (matplotlib, consistent
with the existing project figure style):

- **Training:** return vs env steps (per-seed + mean±CI), loss components, λ(t) overlay.
- **Curvature diagnostics:** penalty value over training, Hutchinson probe variance vs N,
  reward-surface slices ∂²R̂ along random latent 2-planes (the "spikiness" picture, before
  vs after regularization) — `viz/reward_surface.py`.
- **Theory checks:** transversality-angle trajectory (R8), imagined-return variance vs
  horizon curves (R15), Stone-rate log-log error-vs-n plots with predicted slope
  −2s/(2s+d) (item 7), U-curves for λ* (R14).
- **Model graphs:** torchview/torchinfo architecture diagrams; latent-space PCA/UMAP
  embeddings colored by reward — `viz/latent_space.py`.
- `scripts/make_figures.py` regenerates every figure deterministically into
  `results/figures/` from **either source**: local JSONL mirrors that every run writes
  (`results/runs/<name>/metrics.jsonl`, offline-capable) or the W&B API
  (`--source local|wandb|auto`). Same render path either way.

## 6. Experiment pipeline (Part 5 of the founding doc → runnable units)

| # | Experiment | Where | Config |
|---|---|---|---|
| 1 | Multi-env replication, 5 seeds × {Walker2d, Ant, Humanoid, HalfCheetahDense} | Colab GPU (Mode A), seeds parallel via sweep | `experiment/multienv.yaml` |
| 2 | Offline RL (Minari replay datasets) | Colab GPU | `experiment/offline.yaml` |
| 3 | Probe-count sweep N ∈ {1,2,4,8} | Pendulum: local CPU; HalfCheetah: Colab | `experiment/probes.yaml` |
| 4 | Multi-kernel ablation R / R+T / R+T+π + α measurement | Colab GPU | `experiment/multikernel.yaml` |
| 5 | Latent-dim sweep k ∈ {2,4,8,16,32} | Colab GPU sweep | `experiment/latentdim.yaml` |
| 6 | Generic-vs-critical smoothness sweep (synthetic) | **local CPU** (small nets) | `experiment/smoothness.yaml` |
| 7 | Stone-rate curves (synthetic) | **local CPU** | `experiment/stone.yaml` |
| 8 | λ-schedule ablation: (t₀/(t₀+t))^⅓ vs step vs cosine vs constant | Pendulum local, HalfCheetah Colab | `experiment/schedule.yaml` |
| 9 | Multi-task zero-shot generalization: task-conditioned reward, train on N tasks, eval held-out τ (interp + extrap separately); penalty over (z,a,τ) vs (z,a) vs λ=0 | PendulumTarget local, HalfCheetahVel Colab | `multitask.yaml` (own entry point `train_multitask.py`) |
| 10 | **Gap-closing transversality test** (docs/claims_ledger.md): competent-policy data + curved (Gaussian-bump) reward; arms none/R/R+T; ≥5 seeds; success = predicted 6–25% R+T benefit AND benefit correlates with α | **local CPU** | `scripts/transversality_test.py` |

Priority order: 1 (the breadth gap) → 3 (cheap, informs everything) → 8 → 4 → 5 → 2 → 6/7.
Items 6–7 run entirely locally and can proceed in parallel with everything else from day one.

## 7. Correctness guards (tests before experiments)

The penalty math is the whole project; it gets unit tests against analytic ground truth:

- `test_hvp.py` — Hutchinson estimate vs exact Frobenius norm on quadratics with known
  Hessian; unbiasedness over many probes; the 2-probe vs 1-probe-biased distinction (R5).
- `test_null_lagrangian.py` — Frobenius vs Laplacian-trace forms agree in expectation.
- `test_schedule.py` — λ(t) profile values; anneal floor.
- `test_checkpoint.py` — save → restore → bitwise-identical next training step (RNG included).
- `test_affine_dynamics.py` — ∂²T/∂a² = 0 exactly for the affine-in-action class.
- Smoke test: 200 steps of the full loop on Pendulum, CPU, asserts finite losses and a
  decreasing penalty.

## 8. Repo layout

```
mbrl/
├── PLAN.md                     # this file
├── pyproject.toml
├── configs/
│   ├── base.yaml               # model sizes, penalty, schedule, logging defaults
│   ├── env/                    # pendulum.yaml, halfcheetah.yaml, ...
│   └── experiment/             # one per row of §6
├── src/mbrl/
│   ├── models/                 # encoder.py, dynamics.py (affine-in-action), reward.py, policy.py
│   ├── regularization/         # hutchinson.py (HVP penalty), schedule.py, transversality.py
│   ├── training/               # loop.py, imagination.py, buffer.py, smoothing.py (DreamSmooth)
│   ├── utils/                  # checkpoint.py, seeding.py, wandb_utils.py
│   └── viz/                    # curves.py, reward_surface.py, latent_space.py, wandb_panels.py
├── scripts/
│   ├── train.py                # Hydra entry point (GPU or CPU)
│   ├── collect.py              # parallel env interaction (local CPU)
│   ├── local_sweep.py          # joblib runner for synthetic/Pendulum grids
│   ├── analyze.py / make_figures.py
├── notebooks/colab_launcher.ipynb
├── tests/
├── checkpoints/                # local, gitignored; canonical copies in W&B
└── results/figures/
```

## 9. Build order

1. Package skeleton + configs + pyproject *(done with this scaffold)*
2. `regularization/` (the HVP penalty — the heart) + its tests
3. Models + training loop on Pendulum, CPU-only, end-to-end
4. CheckpointManager + W&B wiring + Colab launcher
5. Local parallel runners; synthetic experiments 6–7 (immediate science, zero GPU)
6. MuJoCo on Colab; replicate the HalfCheetah −165→+98 recipe (the regression test for
   everything above)
7. Then the Part-5 queue in priority order.
