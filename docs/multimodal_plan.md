# Multi-modal architecture plan — normal-map tokens end to end

Drafted 2026-06-08. Status: DESIGN — phases gate on pre-registered
adjudications per house method; nothing here is a claim. Builds strictly on
ledger-validated components; every borrowed lesson is cited by run number.

## 0. The core idea, stated once

Every modality is encoded to, predicted as, and decoded from the same
currency: **normal maps** — grids of Gaussian parameter tokens (μ, σ).

- A state vector becomes ONE (μ, σ) token (the run-10 VAE head).
- An image becomes a SPATIAL GRID of (μ, σ) tokens (a conv-VAE posterior over
  patches — literally a normal map in both senses).
- The world model is a transformer that consumes the fused token set at
  times t−H..t plus actions and emits the next normal map per modality
  (run-11 design, generalized from one token to a token set).
- Decoding inverts per modality: MLP for state tokens; **transposed-conv
  decoder for the image grid** — this is where the deconvolution idea
  becomes correct, because the tokens finally have spatial structure
  (decision recorded in VAEEncoder docstring: deconv on 17-dim state would
  be invented structure; on a patch grid it is the native operation).
- Reward stays the validated closed-form spectral head, reading the fused
  latent. Closed-form refits, calibrated ladder, high-clamp polynomial —
  the champion stack (runs 3/5) unchanged in role.

One currency everywhere means: one uncertainty semantics (every token
carries its own σ — calibration is measurable per modality the way
dyn/calib_corr already works), one fusion mechanism (attention over tokens,
no hand-built fusion MLPs), and one place to apply the house rules.

## 1. Architecture

```
state s_t ──► state-VAE ──► 1 token  (μ,σ) ┐
image o_t ──► conv-VAE  ──► P tokens (μ,σ) ├─► [modality + position embeds]
(audio/touch later: 1D-conv VAE → tokens)  ┘          │
                                            ┌─────────▼──────────┐
 a_{t-H..t} ──► action embeds ─────────────►│  causal transformer │
                                            │  over time × tokens │
                                            └─────────┬──────────┘
                              next normal map per modality (μ̂,σ̂)_{t+1}
                                   │                        │
                       state MLP decoder          image deconv decoder
                                   │                        │
                              ŝ_{t+1}                   ô_{t+1}
        fused latent (token means, pooled) ──► spectral reward head (champion)
                                          ──► policy / value (λ-returns)
```

**Tokenization.** Image 64×64 → conv encoder → 8×8 patch posteriors → P=64
tokens of dim d_tok (16–32). State → 1 token. Each token is (μ, σ): the
σ-channel is data, not decoration — attention sees uncertainty, and the
dynamics NLL trains against it (the run-9 finding that bought 4–6 orders of
imagined-variance suppression generalizes per-token).

**Transformer world model.** Causal over time, full attention within a
timestep's token set (so modalities fuse), context H=8–16. Output heads
emit (μ̂, σ̂) per token — NLL loss per modality, σ̂ floor per the gaussian
dynamics LOGVAR_RANGE convention. Imagination rolls out rsamples
(reparameterized, gradients flow — the validated run-9 mechanism).

**Latent budget (the 1×-cap lesson, runs at scale).** The spectral head
reads a POOLED fused latent, not all P+1 tokens: a learned pooling token
(CLS-style) with output dim capped at obs-equivalent size (≤ 32). The
1×-cap rule (ledger 2026-06-07) was about the closed-form fit
over-resolving wide latents; the pooled read is how the cap survives
multi-modality without strangling the tokens themselves.

**Reward head.** Champion spectral stack on the pooled latent. σ*
calibration runs on pooled-latent data; the VAE's KL standardization
(run-10 hypothesis) matters MORE here — if run 10 confirms stationarity,
the multi-modal latent inherits it; if not, recal-on-drift is already wired.

## 2. House rules, translated to multi-modal

1. **Encoder grounding (collapse rule, 2026-06-08):** every modality's
   encoder has a self-contained loss (its reconstruction). The fused/pooled
   path additionally keeps encoder_aux until proven unnecessary — the
   collapse was silent and cost a batch; assume it generalizes.
2. **Spectral rules:** smooth floored λ schedules only; pooled-latent cap;
   z_std + per-band SNR logged per modality.
3. **Per-component statistics need orthogonal/incremental frames (runs 4,
   8, 12B — three confirmations):** any per-token or per-band selection,
   masking, or shrinkage must be measured on residuals or in an
   orthogonalized basis. This WILL come up in token pruning; design for it.
4. **One change per experiment, matched budgets, pre-register before
   results.** The phase gates below are that rule at roadmap scale.
5. **KL balancing across modalities:** free-bits per modality (else the
   image KL swamps the state KL ~64:1 by token count). New rule candidate —
   to be validated in Phase 1, not assumed.

## 3. Phases with gates (each gate = a pre-registered run)

**Phase 0 — prerequisites (running now).** Run 10 (VAE encoder) must
adjudicate SHIP: the multi-modal plan stands on VAE grounding + latent
stationarity. Run 11 (transformer over single (μ,σ) tokens, state-only)
must beat or match gaussian-affine with calibrated σ and pass the
memory-task criterion. **If run 11 fails on Markov tasks AND the memory
task, the transformer core is rejected and this plan reverts to per-modality
VAEs + gaussian-affine fusion** — multi-modal does not require a
transformer; it requires shared currency.

**Phase 1 — pixel modality, single-mode (run ~13).** Conv-VAE normal-map
tokens + deconv decoder on a pixel-only task (DMC cheetah-vision or
Pendulum-pixels). Arms: pixel champion vs Dreamer-style baseline reproduced
in-house. Gate: within 20% of baseline return at matched steps AND
reconstruction sane AND pooled-latent spectral head fits reward (fit MSE
comparable to state-based). This phase also builds the deconv decoder slot.

**Phase 2 — fusion (run ~14).** State + pixels jointly on a task where
both carry signal (e.g., vision + proprioception locomotion). Arms:
fused vs pixels-only vs state-only (the ablation IS the point). Gate:
fused ≥ max(single-modality arms) on return; per-modality calibration
holds; masked-modality eval (drop pixels at test time) degrades gracefully
— the σ-channel should EXPAND on the missing modality's tokens, which is
the normal-map currency paying rent.

**Phase 3 — the payoff regime (run ~15).** Memory + occlusion tasks
(DMC with occlusions / Memory Maze-class): where Markov fails and
single modalities are insufficient. This is the predicted home turf of
the transformer + multi-modal combination (run-11 lesson: without a
memory-dependent task the transformer is overhead). Gate: beats the
Phase-2 champion by the standard bar on ≥2 such tasks.

**Phase 4 — engineering consolidation.** ONNX export of the full token
pipeline (Godot plan §5 extends naturally: tokens are the bridge wire
format), W&B per-modality dashboards, multimodal presets, pipeline.md v2.

## 4. Compute plan

Phases 1–2 are A100-class (Colab Pro or sky.yaml — conv VAEs and the
transformer end the CPU-parity era; the M2 keeps the supervised harness and
nightly research loop). Budget estimate: Phase 1 ~10 GPU-h (3 seeds × 2
arms), Phase 2 ~20 GPU-h (3×3), Phase 3 ~40 GPU-h. Checkpoint/resume and
config-hash lineages carry over unchanged; spot-safe via sky jobs.

## 5. Risks, ranked

1. **Sample efficiency at 200K-step budgets** (transformers + conv VAEs are
   data-hungry; IRIS/TWM needed careful regularization). Mitigation: small
   models (≤4 layers, d_model ≤ 256), token dropout, and the curvature
   penalty story — H² on the pooled read is cheap and already exact.
2. **KL imbalance / posterior collapse per modality.** Mitigation: free
   bits per modality; vae/recon + vae/kl logged per modality from day one
   (the collapse lesson: instrument BEFORE the failure).
3. **The pooled read becomes a new bottleneck** (the cap fighting the
   tokens). Mitigation: pooled-dim sweep is a pre-registered Phase-2
   sub-arm, not a tuning knob.
4. **Scope.** Each phase is one preset, one gate. The plan dies gracefully
   at any gate, leaving shipped components (deconv decoder slot, per-token
   σ machinery, masked-modality eval harness) usable by the next idea.

## 6. What this plan does NOT assume

Run 10's stationarity hypothesis (pending), run 11's transformer value
(pending), the colab_spectral RL validation (pending). If all three land
negative, the multi-modal currency (per-modality normal-map VAEs + shared
σ semantics + masked-modality eval) still stands on its own with
gaussian-affine fusion — the plan degrades to a smaller, still-coherent
architecture rather than collapsing.
