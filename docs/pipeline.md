# Pipeline — how everything works together (and what a Godot port touches)

Written 2026-06-08, post improvement-plan upgrades (commit-scoped results,
preset registry, recal-on-drift, learned-sigma anchor, always-on SNR
diagnostics, config validator). Companion to `docs/architecture.svg` (the
picture) and `docs/claims_ledger.md` (the evidence). This doc is the
operational contract: read it before porting anything.

---

## 1. The loop, end to end

```
 Godot/gymnasium env ──(obs, a, r, obs′)──► ReplayBuffer
        ▲                                        │ sample batch
        │ a = π(encode(obs))                     ▼
        │                                 Trainer.model_update
        │                                  ├─ encoder: z = LayerNorm(MLP(obs)), targets via EMA copy
        │                                  ├─ dynamics: affine μ = z+f(z)+G(z)a  (MSE)
        │                                  │            or gaussian N(μ, σ²(z)) (NLL)
        │                                  ├─ reward path, one of:
        │                                  │   MLP head + Hutchinson/clamped-trace penalty × λ(t)
        │                                  │   SPECTRAL: cache (x=[z⊥,a,τ], y=symlog r) → every
        │                                  │     refit_every updates: closed-form ridge per head
        │                                  │     ĉ = (ΦᵀΦ + diag(θ) + εI)⁻¹Φᵀy; penalty EXACT
        │                                  └─ auto-dose (λ0 from measured fit/penalty ratio)
        │                                 Trainer.behaviour_update
        │                                  ├─ imagine H steps (H certified by penalty EMA ≤ 25)
        │                                  │   z′ ~ dynamics (rsample if gaussian), r̂ = symexp∘clamp∘R
        │                                  ├─ λ-returns → policy ↑, value ↓
        └──────────────────────────────────┘
 CheckpointManager (every N updates + SIGTERM): bitwise state incl. RNGs,
 spectral basis (W/b/c, log_s, calibrated ladder), λ-schedule step; lineage
 dir = results of config_hash(cfg) — config change ⇒ fresh lineage.
 W&B: metrics + checkpoint artifacts (the cross-machine rendezvous).
```

Cadences that matter: collection and training interleave per `training.*`
(steps_per_iter env steps, then model_updates_per_iter updates); the spectral
refit fires when the cache first holds ≥ n_features rows, then every
`refit_every` updates; `sigma_w=auto` calibrates at the FIRST refit (and
re-probes every `recal_every` refits if enabled, rebuilding only on >
`recal_drift`× movement); the learned-sigma gradient step runs every
model_update after the first refit.

## 2. Component contracts (what a port must honor)

**Environment.** The Trainer needs exactly: `reset() → obs`,
`step(a) → (obs′, r, terminated, truncated)`, float32 Box spaces, and episode
boundaries. Nothing else. Vectorization is optional (`collect_vectorized`).
This is the seam where Godot plugs in (§5).

**Encoder/dynamics/policy/value.** Plain MLPs; the only structural promises:
LayerNorm'd latent (scale stability), EMA encoder for targets, dynamics mean
affine in action (R15 — both modes), policy Hessian NEVER penalized (R10).

**Spectral head.** State = three tensors per head: `W (M×d)`, `b (M)`,
`c (M)`, plus `log_s` + `W_base` when learned. Prediction is
`R(x) = Σ c_j √(2/M) cos(w_j·x + b_j)` — portable to any language in ~10
lines; no torch needed for INFERENCE (see §5). All training-side complexity
(calibration, band weights, refits) only ever changes those tensors.

**Penalty/schedule.** λ(t) counts MODEL UPDATES, not env steps. Spectral rules
(validator-enforced warnings): smooth floored decay only, latent cap 1×.

**Targets and scaling.** Rewards are fit in symlog space; imagination applies
a data-driven symexp clamp (bound = margin × running max |symlog r|). The
supervised harnesses standardize inputs with frozen REF_MU/REF_SD. ANY export
must carry its input-standardization constants and the symlog convention with
it — silent scale drift is the classic port killer.

**Checkpoint.** `state_dict` is everything (modules, optimizers, RNG states,
spectral basis, schedule step, symlog bound). Resume is bitwise and
config-hash-scoped. A port that changes config semantics gets a fresh lineage
automatically — that is a feature.

## 3. Configuration reference (post-upgrades)

All composition is Hydra: `configs/base.yaml` + `env/<name>.yaml` +
`+experiment=<name>` (files in `configs/experiment/`, `# @package _global_`
header, note the plus) + CLI overrides (last wins).

| Block | Key knobs | Notes |
|---|---|---|
| model | `dynamics: affine\|gaussian`, `latent_dim`, `latent_cap_mult` (4 MLP / 1 spectral) | gaussian = NLL + rsample imagination |
| penalty | `form: laplacian_trace\|frobenius`, `clamp_trace`, `n_probes`, `auto_dose.*` | MLP path only when spectral enabled (reward side) |
| penalty.schedule | `kind`, `lam0`, `floor`, kind-specific params | spectral: cuberoot + floor>0 (validator warns otherwise) |
| spectral | `enabled`, `n_features`, `heads`, `refit_every`, `cache_size` | |
| spectral.sigma_w | scalar / list / `auto` / `learned` | auto: `cal_mults`, `recal_every`, `recal_drift`; learned: `init_ladder`, `sigma_lr`, `sigma_wd` |
| spectral.poly | `degrees`, `coefs`, `shifts` | the validated band penalty (high-clamp = [1,3]/[0.1,10]) |
| spectral (snr) | `weights_mode: poly\|snr`, `snr_bands`, `snr_ema` | snr = Wiener weights (lost run 4; diagnostics always logged) |
| imagination | `horizon` (cap), `gamma`, `lambda_`, `entropy_coef` | horizon adapts via penalty EMA |
| checkpoint | `every`, `resume: auto`, `push_wandb` | hash-scoped lineages |
| training | `total_env_steps`, `steps_per_iter`, `model_updates_per_iter` | plots vs env steps, always |

Presets: `configs/presets.yaml` (data, not code) → `parallel_runs.py --preset`.
Current science preset: `colab_spectral` (5 arms). Experiment status:
`make status`. Number/prose consistency: `make ledger-check`. Results are
sha-scoped under `results/{bridge,bench}/<short-sha>/`.

## 4. Experiment workflow (the loop you actually run)

1. `make test` (fast set) — the penalty math is the foundation; verify first.
2. Supervised iteration: `make recipe` / `make bridge` — closed-form, seconds
   per cell, resumable, sha-scoped. Record outcomes in the ledger WITH
   falsifiers; `make ledger-check` keeps prose honest.
3. RL validation: `make spectral-rl` locally (Pendulum-class) or the same
   preset on Colab/SkyPilot for MuJoCo. W&B groups by arm; `make dashboard`
   for the offline mirror; `make status` for "what's missing".
4. Ledger adjudication: pre-registered criteria decide; claims move tiers.

## 5. Porting to Godot — recommended architecture

Three viable shapes, in increasing effort; (A)+(B) together are the sweet
spot for a long-term plugin. Avoid (C).

**(A) Godot as the environment (training stays Python).** The proven pattern
(cf. godot-rl-agents): a small Godot addon exposes the game as an RL env over
a local socket; the Python side wraps it in the 4-method contract from §2.
- Godot side (`addons/mbrl_bridge/`): an autoload node that, per physics
  tick when driven externally: applies the received action, steps the scene
  with fixed `physics_delta`, gathers an observation vector + reward, replies.
  JSON-over-TCP is fine at Pendulum-scale; switch to a length-prefixed binary
  frame if obs get large. Determinism: fixed physics tick, seeded RNG, no
  frame-rate coupling — the buffer assumes Markov transitions.
- Python side: a `GodotEnv` gymnasium adapter (~100 lines) registered in
  `configs/env/godot_<game>.yaml` (obs_dim, action_dim, action bounds,
  reward scale → auto-dose handles λ0). EVERYTHING else in the pipeline is
  untouched — this is the whole point of the env seam.

**(B) Inference export (the trained agent runs in-engine, no Python).**
- Policy/encoder/value: export to ONNX (`torch.onnx.export`), run with
  onnxruntime via GDExtension — standard, battle-tested.
- Spectral reward head: do NOT bother with ONNX — it is three arrays and a
  cosine. Serialize `{W, b, c, input_mu, input_sd, symlog: true}` to a
  resource file; a 15-line GDScript/C# function (or compute shader for
  batches) reproduces R(x) exactly. Useful in-engine for reward shaping
  display, debugging, or curiosity-style bonuses.
- Ship the standardization constants and symlog flag INSIDE the export
  artifact (§2's scale-drift warning).

**(C) Full training in-engine: don't.** The penalty needs double-backward
(`torch.func` HVPs) and the spectral path needs dense linear solves; neither
has a sane Godot-native equivalent, and the apparatus's value (Hydra
configs, W&B lineages, ledger discipline) lives Python-side. Keep training
out-of-process forever; the plugin is the env + the exported artifacts.

**Suggested plugin layout**

```
addons/mbrl_bridge/
  bridge.gd            # autoload: socket server, step/reset protocol
  obs_builder.gd       # scene -> float32 obs vector (versioned!)
  reward.gd            # in-engine reward fn (mirror of the Python one)
  spectral_head.gd     # cos-features inference from exported arrays
  onnx_agent.gdns      # GDExtension onnxruntime wrapper for pi/V
  exports/             # policy.onnx, spectral_head.tres, norm_constants.tres
```

**Port gotchas, learned the hard way here:** version the observation layout
(an obs_builder change is a config change — new checkpoint lineage, new
REF constants); keep reward definitions in ONE place mirrored by codegen or
test (the Python reward and `reward.gd` will drift otherwise — add a parity
test that replays a recorded episode through both); Godot physics at
variable timestep breaks the Markov assumption — fixed tick only; and the
λ-schedule counts model updates, so changing the collect:train ratio changes
the effective anneal — recheck doses per game like we re-dose per env.

## 6. Current evidence state (so the port doesn't cargo-cult)

Validated supervised: ladder×poly (+33.7%), SNR-calibrated ladder (+48.3%).
Pending RL: the 5-arm `colab_spectral` readout decides whether any of this
matters in the loop. Not supported: pointwise-positivity penalty (run 1),
Wiener weights as penalty (run 4), σ=1 crossing hypothesis (measured 0.207),
angle-as-cause (run 2b, −0.12 at fixed bandwidth). Untested: learned σ,
gaussian dynamics. A Godot port should start from the SIMPLEST validated
config (fixed ladder + high-clamp poly + cuberoot floor) and let `auto`
prove itself per-game via the logged sigma_star.
