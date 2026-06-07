# Claims ledger — what is defensible at each confidence level

User-supplied inventory (2026-06-06) delineating publishable results. This refines the
founding doc's status tags with results from the ORIGINAL experiments (the supervised
transversality test and the HalfCheetah recipe runs). The full original empirical
record is preserved in `original_findings_report.md` — including the thermodynamic-
consistency finding (sec.3: estimator NON-NEGATIVITY is the operative property; the
clamped Lap-2 trace beat Frobenius +41 vs −40) and the working λ scale (0.5 on
HalfCheetah, sec.7). Treat both documents as the publishability contract.

## Tier 1 — Proved (theory stands alone, verifiable from standard tools)

- The Hessian penalty on reward models IS Sobolev H² regularization; in the RKHS the
  Representer Theorem makes Hessian constraints ellipsoidal.
- Wiener-filter identity: the penalty implements optimal rate-distortion filtering with
  cutoff at SNR = 1.
- Biharmonic flow: gradient descent on the penalized loss is a singularity-free
  geometric flow.
- Transversality theorem (conditional): IF reward and dynamics Hessians have misaligned
  eigenvectors THEN the multi-kernel constraint reduces the covering number, sample
  complexity improvement √((d_eff − κ)/d_eff); regret bound via the simulation lemma.

## Tier 2 — Empirically confirmed

- **Transversality angle 60–71° on a real environment** (strongest empirical result):
  sin²(65°) ≈ 0.82 ⇒ 82% of the theoretical maximum multi-kernel benefit is available.
  The central assumption of the multi-kernel theory holds in practice.
- **DS + Hessian + residual-annealing recipe: −165 → +98 (+263) on HalfCheetah.**
  The residual schedule winning matches the Dudley's-integral prediction (linear decay
  with floor).

## Tier 3 — NOT confirmed (and must be stated as such)

- **Multi-kernel Hess(R+T) showed ~zero improvement over Hess(R)** in the supervised
  test: −0.5% to +1.6%, within noise. The transversality is real; the measured benefit
  is not. Mitigating context: theory predicts only ~6.5% at d_eff = 8 — below the
  statistical power of 3 seeds with noisy rewards, AND the random-policy data made the
  reward nearly linear (trivially fit with or without penalty). Not dead; not validated.

## Defensible headline statement

The Hessian penalty on reward models in MBRL implements optimal Sobolev regularization;
in the RKHS regime it defines ellipsoidal constraint sets whose covering numbers
determine sample complexity; λ implements a Wiener filter whose optimal cutoff matches
the policy-gradient SNR; empirically: +263 return on HalfCheetah with a simple recipe;
reward/dynamics Hessians measured 60–71° misaligned, confirming the transversality
condition under which multi-task smoothness constraints provably reduce sample
complexity.

## Flagged conjecture (beyond current evidence — label clearly if stated)

Multi-kernel Hessian constraints triangulate the latent representation via intersection
of Sobolev balls (holographic encoding at critical dimension d = 4); the variational
principle is a biharmonic field theory with EL equation λ∇⁴R + (R − r) = 0.

## The schedule hypothesis (user, 2026-06-06 — drives the schedule ablation)

Stated form: high λ early produces a smooth, general mapping of the reward manifold
(smooth early gradients); λ should then EASE over time so policy and reward can
exploit the environment's real structure; and **λ must never be exactly zero or the
MLPs collapse**. Decomposed into falsifiable parts, tested by `schedule_ablation`:

- (a) smooth easing > abrupt release: cuberoot / sin2chirp / cosine vs step.
  (Note (a) agrees with R12's theory profile; it CONTRADICTS the original §7 result
  where step-to-zero won — adjudicate empirically.)
- (b) floor > 0 matters: `sched-step` (floor 1e-5) vs `sched-step-zero` (exact 0).
  Collapse prediction: step-zero degrades late training.

## The gap-closing experiment (now experiment 10, `scripts/transversality_test.py`)

Repeat the supervised transversality test with **trained-policy data** (narrow state
distribution) on a **genuinely curved/sparse reward** (the random-policy version made
generalization too easy). Success criteria, both required:

1. Multi-kernel Hess(R+T) shows the predicted ~6–25% sample-efficiency improvement
   over Hess(R), with seeds ≥ 5 for adequate power at a ~6% effect.
2. The improvement **correlates with the measured transversality angle** α across
   conditions.

If both hold: theory predicts transversality → measures it → predicts generalization
gain → measures it. A closed theory-experiment loop, publishable standalone.

**Run 01 (bump reward): null, diagnosed as task collapse, not refutation** — the bump's
curvature was rank-1 (d_eff ≈ 1), making R+T redundant by construction and α ≈ 88°
trivially orthogonal. See `results/analysis_transversality_01.md`. v2 uses a reward
with measured true d_eff ≈ 3.55 (predicted benefit ~15%); d_eff is now measured per
arm as a prediction — the penalty should push it down on its own (Wiener filtering IS
dimension reduction), and the benefit should track √((d_eff−κ)/d_eff) at the MEASURED
d_eff.
