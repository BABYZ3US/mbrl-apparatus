# Pre-registration — Phase 1: self-bounding operator loss vs. cf4 clipping

**Status: pre-registered.** Criteria fixed BEFORE any cell runs. Do not edit after the first
seed launches (append an amendment block instead). Companion to
`operator_world_model_test_map.md` (Phase 1) and `operator_world_model_candidate_algorithms.md`.

## Question
Does a **self-bounding operator loss remove the need for the cf4 reward/return/value clips** —
matching what clipping buys (no divergence) from the loss geometry itself — on the dual operator
latent, at the normal latent? Framed as a **reliability + divergence** claim, **not** an
absolute-return claim (the mlp-recipe anchor regression is unresolved → absolute HalfCheetah
return is void; we compare arms relatively).

## What exists vs. what must be built (read first)
- **Config-ready (Tier 1, launch today):** the cf4 clips are live flags
  (`imagination.reward_clip`, `imagination.return_clip`, `optim.value_clip`,
  `optim.skip_nonfinite`); the **Euclidean Stein/Lyapunov consistency** regularizer is live as
  `model.dual_latent.lyap_weight` (`‖G − A_d G A_dᵀ − Q̂‖²`, `loop.py:1664`). All reward/dyn/value
  losses are currently `F.mse_loss`.
- **Needs a build (Tier 2):** the per-mode **operator cross-entropy** `Σ_i φ(ν_i)`,
  `φ(ν)=ν−log ν−1`, `ν=eig(L⁻¹GL⁻ᵀ)`, `Σ̂=A G Aᵀ+Q` — the information-geometry loss of
  derivations §6, which is *not* the same as the Euclidean `lyap_weight` residual. It is **not
  yet a trainable objective**; Tier 2 is gated on the build spec at the end.

---

## Tier 1 — Stein-consistency self-bounding (LAUNCH TODAY, config-only)

**Hypothesis (T1).** Turning on the existing Stein/Lyapunov consistency (`lyap_weight=0.3`, the
A7 lever) lets us run with the cf4 clips OFF without more divergence than the clipped baseline,
and with reliability ≥ the *unregularized, unclipped* baseline. I.e. the energy-consistency term
substitutes for the external clamps.

**Design — three arms on the cf22 canonical dual-latent base** (HalfCheetah-v5, twin operator
mode, the validated d=16 normal latent; everything else byte-identical to the cf22 base used in
Ablation-1). The ONLY differences:

| arm | `dual_latent.lyap_weight` | cf4 clips | role |
|---|---|---|---|
| **T1-A0** | `0.0` | OFF (`reward_clip=0, return_clip=0, value_clip=0, skip_nonfinite=false`) | unregularized + unclipped (current default) |
| **T1-Ac** | `0.0` | ON (`value_clip=100`, `reward_clip=RC`, `return_clip=RR`, `skip_nonfinite=true`) | clipped reference (what clipping buys) |
| **T1-A1** | `0.3` | OFF | self-bounding via consistency, no clips |

`value_clip=100` mirrors the live `actor_clip=100`. `RC, RR` are fixed by a **pre-registered
rule** (not a peeked value): run T1-A0 seed 0 for 20k steps, set `RC = 2 × p99(|imagined reward|)`
and `RR = 2 × p99(|imagined λ-return|)` over that window, then freeze for all T1-Ac seeds. Record
the two numbers in the amendment block before launching T1-Ac.

## Tier 2 — the φ(ν) operator cross-entropy loss (GATED ON THE BUILD BELOW)

**Hypothesis (T2).** The information-geometry per-mode loss `Σφ(ν)` self-bounds *better* than the
Euclidean consistency term (its `tr(Σ̂⁻¹G)→∞` barrier is intrinsic), so **T2-A2** (φ-loss, clips
OFF) matches T1-Ac reliability with ≤ T1-A1 divergence.

| arm | reward/dyn loss | clips | role |
|---|---|---|---|
| **T2-A2** | `operator_xent` (`Σφ(ν)`) | OFF | the real closed-form self-bounding loss |

---

## Seeds & compute (both tiers)
- **8 seeds per arm (0–7).** 500K env steps, deterministic eval on. (8v8 is the power floor; see
  below.) HalfCheetah-v5, cf22 base.
- Runner: `scripts/run_phase1_operator_loss.sh` (to be written, mirroring
  `run_followup_a6a7.sh`); or launch from the Studio (`submit.spec` per arm). W&B group prefix
  **`p1-loss-`** (kept distinct from `abl1-`, `fu-a6a7-`).
- Eval episodes = the abl1 value (3); permitted pre-registered amendment: raise to 10 (apply to
  ALL arms) for more reliable `final`.

## Primary metric & decision rule
Per seed, on `eval/return_det`: `final` = mean of last 3 evals; `fired ≡ final > 200` (the abl1
threshold, fixed in advance). Per arm, with `k` = #fired of 8:
- **Divergence score** `Dv` = total `stab/nonfinite_skips` + count of `latent/gram_cond`
  excursions above `10×` its first-10k-step median (a fixed, per-seed-relative threshold).
- **Clip-need signal** (T1-Ac only): mean `stab/reward_clip_frac`, `stab/return_clip_frac`, and
  fraction of steps with `stab/value_grad_norm > 100` — i.e. *how often clipping actually bound*.

**SHIP "drop the clips, keep the self-bounding loss"** (T1-A1, and T2-A2 if built) iff:
1. `k(A1) ≥ k(Ac)` AND `k(A1) − k(A0) ≥ 3` (reliability: at least matches the clipped reference
   and clearly beats the unregularized-unclipped baseline), AND
2. `Dv(A1) ≤ Dv(A0)` (no worse divergence than the unclipped baseline), AND
3. T1-Ac's clip-need signal is **non-trivial** (clips actually bound in Ac) — otherwise the test
   is moot (nothing to remove) and the verdict is INCONCLUSIVE-by-construction.

**NOT SUPPORTED (falsifier):** `k(A1) < k(Ac)` OR `Dv(A1) > Dv(A0)` — the consistency/φ loss does
not substitute for clipping; cf4 clips stay.

**INCONCLUSIVE:** anything between → report the 3×8 (or 4×8 with T2) contingency table + the
per-arm `Dv` and clip-need, do not ship.

## Verification overlay (not a perf arm)
On any arm where `detpos_weight>0` holds `det(op_p)≈1` (the conservative manifold), run the
**2-adic exact-value check** on the operator loss value (Hensel/Dixon lifting + rational
reconstruction) to certify it bit-exactly. This is the ONLY role of the 2-adic machinery
(`ℚ₂` is unordered → it cannot drive minimization); it does not gate the SHIP decision.

## Power note
8v8 binary is modest power (6/8 vs 2/8 ≈ one-sided Barnard p≈0.06) — hence the ≥3-seed margin and
the built-in INCONCLUSIVE band. Report Barnard's one-sided p descriptively, not as a gate.

## Falsification summary
If T1-A1 fails to match T1-Ac reliability or diverges more than T1-A0, the "self-bounding loss
removes clipping" hypothesis is refuted at the consistency-term level, and the cf4 clips are
retained as load-bearing — **a publishable negative** that also tells us the Euclidean Stein term
is too weak (motivating Tier 2's φ-loss, or retiring the idea if T2 also fails).

---

## Build spec for Tier 2 (the `operator_xent` loss) — implement before T2 launches
1. Add `model.operator_loss: mse | operator_xent` (default `mse`, byte-exact legacy).
2. When `operator_xent`, replace the reward/dynamics `F.mse_loss` (`loop.py:976/1093/1579/1585`)
   with the per-mode Stein loss: form `Σ̂ = A G Aᵀ + Q`; Cholesky `Σ̂ = LLᵀ`; symmetric
   `M = L⁻¹ G L⁻ᵀ`; `ν = eigvalsh(M)`; `loss = Σ_i (ν_i − log ν_i − 1)`. Use the analytic gradient
   `φ'(ν)=1−1/ν` (never the matrix inverse); guard `ν>0` (Cholesky ridge `+εI`, ε = `logdet_eps`).
3. It must be **default-off byte-exact** (gate on the new flag), add no `self.gen` draws, and log
   `loss/operator_xent`, `spectral/fit_r2` (already shipped), and the per-mode `φ(ν)` spread for
   the viz.
4. Pre-register a tiny numerical-parity test (the derivations' `{5/4,4/5,1}→D=1/20` exact case)
   before training. This build is itself a behavior-changing change → land it test-first like the
   cf4 fix, then run T2.

## To launch
Tier 1 is config-only and anchor-immune — launchable now (write `run_phase1_operator_loss.sh` or
submit the three arms from the Studio). Tier 2 waits on the build spec above. Recommended: run
Tier 1 in parallel with Phase 2 (the duty-cycle prereg), since both are cheap and independent.
