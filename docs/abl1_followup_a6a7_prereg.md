# PRE-REGISTRATION — A6 vs A7 follow-up (Stein consistency as a firing stabilizer)

**Status: pre-registered. Criteria fixed BEFORE any cell runs. Do not edit after the first
seed launches** (per the project's pre-registration discipline; cf. runs 12B / 13 in
`claims_ledger.md`). Motivated by Ablation-1 finding (iii) — see
[`abl1_adjudication.md`](./abl1_adjudication.md) and [`analysis_ablation1.md`](./analysis_ablation1.md).

## Hypothesis

In Ablation-1, **A7lyap** (svband + Stein consistency) was the only arm where both seeds
reached and *held* a positive deterministic return; its sole-difference control **A6antifreeze**
(svband only) fired on 1/2 seeds. Hypothesis: the Stein consistency term
(`model.dual_latent.lyap_weight = 0.3`) **raises the rate at which a seed reaches and holds a
firing policy** on HalfCheetah, relative to svband alone. This is a reliability claim
(P(fire)), not an absolute-return claim — abl1 sits on the unresolved mlp-recipe anchor.

## Design — one change

The ONLY difference between the two arms is `model.dual_latent.lyap_weight`:

| arm | override on the cf22 canonical BASE | lyap_weight |
|---|---|---|
| A6 (control) | `w_svband=2.0 radius_max=0.9 radius_min=0.05` | 0.0 |
| A7 (treatment) | `w_svband=2.0 radius_max=0.9 radius_min=0.05` + `dual_latent.lyap_weight=0.3` | 0.3 |

Everything else is byte-identical to the `abl1` cf22 BASE (`run_ablation1.sh`). **Seeds: 0–7
(8 per arm, 16 runs).** Budget: **500K env steps**, deterministic eval on. Runner:
[`scripts/run_followup_a6a7.sh`](../scripts/run_followup_a6a7.sh) (self-contained; W&B group
prefix `fu-a6a7-` so it does NOT contaminate the original `abl1-` group; original A6/A7 s0/s1
runs may be reused as 2 of the 8 seeds since the config hash is identical — record whether you
do). Training is byte-identical to abl1 A6/A7 (a pure seed-scaling replication); the eval
episode count stays at the abl1 value (3, `train.py:44`) so the comparison is clean — see
"power note" for the optional eval-episode bump.

## Pre-registered metrics

Per seed, on `eval/return_det`:
- **`final`** = mean of the last 3 logged evals.
- **`peak`** = max eval over training.
- **`fired`** ≡ `final > 200`. (Threshold fixed in advance: the abl1 A0 baseline never had a
  `final` above +50 — its lone spike, peak 312, reverted to −44 — and every abl1 "fired+held"
  seed had `final` ≥ 314, so 200 cleanly separates a held gait from baseline/flat. It is NOT
  tuned to A6/A7.)
- **`held`** ≡ `final ≥ 0.6 × peak` (did a fire survive vs revert).
- Mechanism (descriptive, logged not gating): `op/radius_d` (expect ≈1.0 both arms — sanity
  that the svband floor, not pinning, is what differs), `latent/z_std`, `op/eff_rank_d`,
  `dyn/calib_corr` if present.

## Pre-registered decision rule (fixed)

Let `kA6`, `kA7` = number of `fired` seeds out of 8 in each arm.

- **SHIPS (Stein term confirmed as a firing stabilizer, folds into the champion config):**
  `kA7 ≥ 6/8` **AND** `kA7 − kA6 ≥ 3` **AND** A7's fired-seed `final` mean ≥ A6's.
- **NOT SUPPORTED (FALSIFIER):** `kA7 ≤ kA6`, OR `kA7 < 6/8`. Then the abl1 A7 result was a
  2-seed fluke; the Stein consistency term is recorded as non-load-bearing for firing and
  drops from the champion config pending a different rationale.
- **INCONCLUSIVE:** anything between — report the 2×8 contingency table and Barnard's exact
  one-sided p (descriptive only), do NOT ship, and state the power shortfall.

Secondary, reported but NOT gating: median `final` and `peak` per arm; `held` fraction among
fired seeds; whether `op/radius_d` ≈ 1.0 in both arms (confirms this is a Stein-vs-no-Stein
contrast at a fixed svband floor, not a hidden |λ| difference).

## Power note

8v8 binary outcomes is modest power: detecting 6/8 vs 2/8 is ~one-sided p≈0.06 (Barnard),
4/8 vs 1/8 ≈ p≈0.13 — hence the SHIP rule demands a ≥3-seed margin, not mere significance,
and an INCONCLUSIVE band is built in. **Optional pre-registered amendment (decide before
launch, not after):** if budget allows, raise the eval to 10 deterministic episodes by passing
`episodes=10` at `scripts/train.py:168` (the `eval/return_det` call) to cut per-seed eval
noise — this is the one permitted deviation from byte-identical abl1 eval, must be applied to
BOTH arms, and must be fixed before the first seed runs.

## Falsification summary

If A7 does not fire on ≥6/8 seeds with a ≥3-seed margin over A6, the headline abl1 positive
collapses and the operator-spectrum lever program retains only its NEGATIVE result (radius
pinning is inert). That outcome is as publishable as the positive — record it either way.
