# Ablation-1 (`abl1`) data analysis — operator-spectrum / Lyapunov levers

**Date:** 2026-06-16 · **Env:** HalfCheetah-v5 · **Source:** W&B
`pandelak-boston-college/mbrl-curvature`, 45 runs, pulled to `results/abl1_cache/`.
**Primary metric:** `eval/return_det` (deterministic mean-policy return — the
pre-registered adjudication metric in `run_ablation1.sh`). Final = mean of the last 3
evals per seed; "peak" = max eval over training. `eval/return_det` itself averages 3
deterministic episodes (`scripts/train.py:44`). Figure: [`./ablation1_summary.png`](./ablation1_summary.png)
(local cache of pulled histories: `results/abl1_cache/`, gitignored).

> This is the measurement pass. The campaign's stated verdict question — *which lever
> moves `eval/return_det` vs A0, what to explore next* — is answered below. The
> ledger-ready adjudication block is in [`abl1_adjudication.md`](./abl1_adjudication.md)
> (kept as a separate doc to fold into `claims_ledger.md` rather than editing that file
> while it carries other uncommitted edits), and the registered follow-up it calls for is
> [`abl1_followup_a6a7_prereg.md`](./abl1_followup_a6a7_prereg.md).

## TL;DR

1. **The data is bimodal (fire vs collapse) with n=2 seeds on most arms, so arm *means*
   are not interpretable** — each "winning" mean (A1, A4, A5, A19) is one seed that fired
   to 1300–3400 paired with one flat seed (SD ±570–995). Report fire-fraction and peaks,
   not means.
2. **A7lyap (svband + Stein consistency, d=16) is the one real signal:** the *only* arm
   where **both** seeds fired (+486 ± 172 final; peaks 597 / 996), and it ran the full
   500K. Its control A6antifreeze (svband only) fired 1/2. A6→A7 differ by exactly the
   Stein term (`lyap_weight=0.3`) — the cleanest causal contrast in the campaign, and it
   points to the consistency term as a *stabilizer of firing*, not a return booster per se.
3. **The radius-pinning program (A9–A13, A18, A19) is mechanistically refuted:**
   `op/radius_d` converges to **≈1.0 in every seed** regardless of penalty strength
   (w_svband=15), initialization (0.447 / 0.707 / 0.99), or anneal schedule. The band
   penalty cannot move |λ| off the unit circle. This is the campaign's only
   *fully-powered* finding (~25 seeds, no exceptions, independent of the noisy return).
4. **The env-dim² latent idea (A14 d=256, A15 d=289) failed computationally** — A15 logged
   0 evals, A14 1 eval (the O(d³) operator SVD is too expensive). The amortized variants
   A16/A17 (phased SVD, d=289) ran to 500K but never fired. Dead end.

## Per-arm result (vs A0 baseline; baseline final ≈ −126, but baseline itself fires once)

| arm | seeds | reached | fired/seeds | final det (mean±sd) | peak (max) | read |
|---|---|---|---|---|---|---|
| A0 baseline | 2 | 460K | 1/2 | −126 ± 82 | 312 | baseline *also* spikes (s1 312→revert) |
| A1 modelfit (hidden 384) | 2 | 450K | 1/2 | +782 ± 913 | 1729 | one seed fires+holds, but imag→∞ at end |
| A2 pconsist (×3) | 2 | 450K | 0/2 | −25 ± 6 | 29 | flat, low variance |
| A3 autoalpha | 2 | 425K | 0/2 | −111 ± 103 | 63 | flat |
| A4 dblvalue (twin Q) | 2 | 440K | 1/2 | +524 ± 569 | 1365 | one seed fires+holds |
| A5 latent32 | 2 | 360K | 1/2 | +545 ± 615 | 1263 | one seed fires+holds (cut short ~360K) |
| A6 svband only | 2 | **500K** | 1/2 | +211 ± 230 | 583 | svband control for A7 |
| **A7 svband + Stein** | 2 | **500K** | **2/2** | **+486 ± 172** | 996 | **only both-seed firer; tightest** |
| A8 detpos | 2 | 345K | 0/2 | −53 ± 32 | 5 | flat (cut short) |
| A9 init.707 | 2 | 310K | 0/2 | −222 ± 69 | −55 | worst; pin failed |
| A10 init.447 | 2 | 250K | 1/2 | +298 ± 339 | 709 | one fires (cut short ~250K) |
| A11 anneal | 2 | 160K | 0/2 | −127 ± 95 | 142 | terminated ~160K |
| A12 pin.447 ("critical") | 4 | 405K | 0/4 held | −191 ± 146 | 487 | fire→**revert** (drawdown to 788) |
| A13 pin.458 ("just-above") | 8 | 385K | ~0/8 held | −141 ± 108 | 400 | 1/8 transient fire, reverts; rest flat |
| A14 d256 | 2 | **5K** | dead | −47 | — | compute failure (1 eval) |
| A15 d289 | 2 | **0** | dead | — | — | compute failure (0 evals) |
| A16 svd d289 | 1 | 500K | 0/1 | −144 | 135 | ran but never fired |
| A17 phased d289 | 1 | 500K | 0/1 | −67 | 96 | ran but never fired |
| A18 pin.707 | 1 | 155K | 0/1 | −126 | −75 | terminated ~155K |
| A19 ride.99→.8 | 2 | 500K | 1/2 | +886 ± 996 | 3408 | one huge fire (3408→1882), one revert |

*"fired" = a seed whose peak `eval/return_det` exceeded 300.*

## What the numbers actually say

**Firing is a property of the apparatus, not cleanly of any single lever.** Even A0
baseline has a seed that peaks to 312 before reverting. So the single-seed fires in
A1/A4/A5/A19 (the big means) are not separable, at n=2, from the same stochastic firing
the baseline shows. The only arm that lifts firing from a coin-flip to *reliable* is **A7**
(2/2, and it's the only arm where the post-peak return *holds* on both seeds rather than
reverting). A7's edge over its own control A6 (2/2 vs 1/2, +486 vs +211) is the strongest
within-campaign evidence, and it isolates the **Stein consistency term** as the active
ingredient — consistent with the ledger's recorded "validated A7, det_m4=487".

**"Fire-then-revert" is the dominant trajectory shape.** Of the seeds that peaked >150,
more reverted (lost ≥70% of peak: A0-s1, A12-s2/s3, A13-s6, A19-s0) or only partially held
than cleanly held. The pinning/critical arms are the worst for this: A12 seeds snap to
peaks of 223–487 and then crash to −300 (drawdown up to 788). **This refutes the A12/A13
"sit on the critical separatrix → convergent basin" hypothesis** — at the matched ratio the
system snaps and reverts; nudging 0.20→0.21 (A13, 8 seeds) did *not* buy a convergence
fraction (≈0/8 held).

**The pin never bites (Panel B).** `op/radius_d` final values, against their pinned targets:
A10 (target 0.447) → 1.03; A12 (target 0.447, **w_svband=15**) → 1.00–1.02, min ever 0.83;
A18 (init 0.707 + w=15) → 0.955; A9 (init 0.707) → shot *above* 1.0 immediately; A19
(ride 0.99→0.80) → 1.00. Across ~25 seeds the dynamics operator's dominant singular value
collapses onto |λ|≈1 (marginal/critical edge) at any penalty weight, init, or anneal. The
"control the energy/entropy ratio by pinning |λ|²" mechanism (A9–A13, A18) **does not
actuate the quantity it targets** — so those arms' return results are uninterpretable as
tests of the ratio hypothesis; they're tests of "what happens when you fight the operator
and lose." A19's own design note anticipated this ("the data says |λ| WANTS the edge"); the
data confirms it decisively.

**Run-length confound (must be stated).** Arms reached very different env-step budgets
(A11/A18 ≈155–160K, A10 ≈250K, A9 ≈310K, A8 ≈345K vs A6/A7/A16/A17/A19 = full 500K). The
firing that distinguishes A6/A7 happens *late* — at matched 300K **every** arm is still
negative (A7 itself is −115 @300K). So the short arms (A8–A11, A18) may simply have been
killed before any firing window. A12/A13 are less excused (≈400K, near where A0/A6/A7 fired,
yet still reverted). Paired identical stop times (A9 both 310K, A11 both 160K, A8 both ~347K)
suggest per-wave wall-clock/crash cutoffs rather than convergence — the local `results/gridlogs/`
is empty (cloud run), so the termination cause can't be confirmed from metrics alone.

**Weak mechanistic correlate.** Pooled over all seeds, fired seeds have slightly higher
`latent/z_std` (0.78 vs 0.74) and *lower* `op/eff_rank_d` (17.0 vs 24.2) than flat seeds —
suggestive that firing coincides with a lower-rank, slightly more excited latent, but it's
underpowered and not a clean predictor.

## Recommendation — what to explore next

1. **Promote A7 (svband + Stein consistency, d=16) and re-run it at ≥6 seeds, full 500K.**
   It is the only arm with a real signal; n=2 cannot establish its fire-rate. Its proper
   control is A6 (svband only) at matched seeds — the A6→A7 (Stein term) contrast is the
   one publishable comparison the campaign gestures at.
2. **Retire the radius-pinning sub-program (A9–A13, A18, A19) as specified.** The lever is
   inert: |λ|→1 regardless. If the entropy-ratio idea is to continue it needs a *different*
   actuator (e.g. constraining the parametrization itself, or operating on a quantity the
   optimizer can't undo), not a stronger band penalty.
3. **Drop env-dim² latent (A14/A15/A16/A17).** Computationally infeasible at full rank and
   no return payoff when amortized.
4. **Fix the eval-variance / run-length protocol before any further adjudication.** With
   fire/revert bimodality, single-rollout `return_det`, and ragged budgets, the campaign
   cannot rank levers. Need: ≥5 seeds, multi-rollout deterministic eval, and a fixed env-step
   budget all arms actually reach (or report at a matched budget). A20excite (the planned
   phase-kick arm) never launched — it is the natural next test *if* the eval protocol is
   fixed first, since its whole premise (break the fire/collapse bimodality) is exactly what
   this analysis says is the core problem.
