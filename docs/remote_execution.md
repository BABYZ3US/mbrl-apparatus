# Remote-first execution — the sealed-environment transition

Drafted 2026-06-08. Goal: ALL training runs execute remotely in one sealed,
backend-agnostic environment; the local machine becomes a thin client (tests,
analysis, the nightly supervised loop). No more Colab-clone drift, no more
"which checkout is this batch running?" — both of which cost batches this week.

## 1. Principles (the seal)

1. **One image, one dependency set.** `requirements-core.txt` is the COMPLETE
   runtime — 7 pinned packages (torch, numpy, gymnasium[mujoco], hydra-core,
   omegaconf, wandb, joblib). Analysis deps (matplotlib, tensorboard, tqdm,
   viz extras) are dev-only and must never be imported by training code; CI
   enforcement: `make seal-check` greps training entrypoints for non-core
   imports.
2. **State never lives where code runs.** Checkpoints push to W&B artifacts
   (already wired, config-hash lineages); metrics mirror to W&B + JSONL;
   supervised results are sha-scoped. A worker can be preempted, destroyed,
   or swapped across providers and the run resumes bitwise from the artifact
   lineage — environment PERSISTENCY is a property of the artifact store,
   not the machine.
3. **Backend-agnostic = container + SkyPilot.** The image runs identically
   under `docker run` on any GPU box, `sky jobs launch` on 20+ providers, or
   Kubernetes. Colab is RETIRED (2026-06-08): it cannot run the sealed image
   and its clone drift burned two batches (06-07, 06-08). Training and
   inference are cloud-only; local = tests, the supervised nightly loop, and
   pulling W&B logs/artifacts for analysis.
4. **Provenance in the image.** `.git` ships inside the container so
   sha-scoped results and `config_hash` lineages work unchanged remotely.

## 2. The pieces (this commit)

- `requirements-core.txt` — the sealed runtime export, GENERATED from uv.lock
  by `make lock` (pyproject = manifest, uv.lock = exact cross-platform
  resolution constrained to the ledger-validated versions, .python-version
  pins the interpreter at 3.11 — runtime mismatch is structurally closed).
- `Dockerfile` — python:3.11-slim + core deps + repo; headless MuJoCo
  (MUJOCO_GL=osmesa); non-root; entrypoint `train.py checkpoint.resume=auto`.
- `sky-docker.yaml` — SkyPilot task pinned to the image (vs sky.yaml's
  workdir-sync mode, kept for dev).
- Makefile: `make image` (build, tagged with the git sha), `make seal-check`.

## 3. Workflow after the transition

```
code change -> commit -> make image && docker push <registry>/mbrl:<sha>
            -> sky jobs launch sky-docker.yaml --env IMAGE_TAG=<sha> \
                   --env WANDB_API_KEY --env EXP=champion --env ENV_NAME=halfcheetah
            -> watch W&B; make status reads the same artifacts locally
```

Local keeps: `make test` (fast suite), the supervised harness + nightly
research cycle (CPU, no GPU deps beyond core), figures/dashboard from JSONL
mirrors. Everything else is remote.

## 4. Long-term harness: PufferLib evaluation (so we stop rolling our own)

What we currently hand-roll: env vectorization (`collect_vectorized` +
gymnasium AsyncVectorEnv), the replay buffer, preset fan-out, and per-env
plumbing. [PufferLib](https://arxiv.org/abs/2406.12905) is the strongest
candidate to absorb the ENV SIDE: an emulation layer that flattens
gym/gymnasium/PettingZoo envs into uniform tensors, vectorization measured
at millions of steps/sec, and the Ocean suite (20+ envs — including
memory-task candidates the multimodal plan's Phase 3 needs).

**Adoption boundary (firm):** PufferLib for emulation + vectorization +
environment zoo ONLY. PuffeRL (its PPO trainer) is model-free and replaces
nothing we value — the Trainer (world model, spectral head, penalty
machinery, ledger-validated recipes) stays ours. The integration surface is
exactly the env seam documented in pipeline.md §2 (reset/step/spaces), which
is also the Godot seam — one adapter interface, three backends (gymnasium,
pufferlib, godot-bridge).

**Pre-registered adoption gates (run when integration is attempted):**
- **Gate 0 — continuous actions.** The PufferLib paper/docs list continuous
  action spaces as unsupported-at-the-time; our entire suite is continuous
  control. Verify against the CURRENT release before writing any code. If
  still missing: adoption deferred, re-check quarterly, no partial adoption.
- **Gate 1 — parity.** champion on HalfCheetah, pufferlib vectorization vs
  gymnasium AsyncVectorEnv, 3 seeds: returns within seed noise AND
  steps/sec >= 2x. Anything less is churn for its own sake.
- **Gate 2 — determinism.** Bitwise checkpoint-resume still holds through
  the pufferlib path (the buffer's Markov assumptions + seeding survive its
  emulation layer). The smoke + checkpoint tests must pass unmodified.
- Dependency cost: pufferlib joins requirements-core.txt ONLY if gates pass
  (the seal stays minimal; an env zoo we don't use yet is not a core dep).

Alternatives considered: EnvPool (C++ vectorization, very fast, narrower env
coverage, also needs the continuous/MuJoCo check), pure gymnasium
AsyncVectorEnv (status quo — zero new deps, known-slow at scale). Decision
deferred to the gates; the adapter interface is the hedge either way.

## 5. What this does NOT change

The science loop (pre-registration, ledger, sha-scoped results), the
checkpoint format, the config system, or the nightly research cycle. This is
a logistics layer: the same experiments, the same evidence rules, machines
that no longer matter.
