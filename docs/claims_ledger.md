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

## The Weil-positivity identification (user, 2026-06-07 — conjecture tier)

The clamp's missing spectral analog (the open question from the spectral solver: the
max(est,0) coherence rectifier has no diagonal form in frequency space) is identified
by the user as **Weil positivity** — the explicit-formula positivity criterion. The
structural correspondence: both demand non-negativity of a quadratic functional tested
in a spectral dual domain (clamp: per-sample sign-coherence of the curvature quadratic
form; Weil: W(g⋆g̃) ≥ 0 over a test class). Status (updated 2026-06-07, per user): **the user asserts this is the real
identification, not an analogy — it is the object they have been working on in the
parent math project**, where the derivation lives. This crossing of the founding doc's
RH/Connes firewall is therefore intentional and user-authorized; the derivation is not
reproduced or verified in this repo. What is testable HERE, with no RH machinery: if
the clamp IS Weil positivity in the RFF dual, then a positivity-constrained spectral
solve (curvature-density positivity as the constraint class) should reproduce the
clamped-trace penalty's empirical advantage (+41 vs −40, report sec.3) in closed form
— same ordering, no Hutchinson, no clamp heuristic. That experiment is the bridge
between the two projects and the next natural build.

**Bridge run 1 (2026-06-07, `scripts/bridge_experiment.py`): NOT SUPPORTED — 0/9
cells.** Closed-form supervised proxy (competent-policy Pendulum, noisy train/val
targets σ=1, clean test; arms scale-matched in expectation to diag(|w|⁴), verified
in `tests/test_bridge.py`). Predicted ordering lap2_positive < frobenius_diag <
lap2_indefinite held in 0/9 cells; the positivity arm lost to the diagonal
Frobenius arm in 9/9 (n=8192 means: 0.107 vs 0.012 test MSE), robust to a λ sweep
spanning 1e-6..1e4. Diagnosis: the exact per-sample Gram penalty c'Gc constrains
curvature only on the span of the data-evaluated Laplacian vectors — data-null
frequency directions go unpenalized and absorb target noise; the isotropic
expectation penalty constrains every band. Pointwise positivity of the curvature
density, by itself, did NOT reproduce the clamp's advantage here. Caveats: (1)
supervised-MSE proxy for an RL-return ordering; (2) one constraint-class
construction tested (PSD Gram of the exact Laplacian density) — sign-coherence
cone constraints or hybrid diag+Gram forms are untested; (3) the clamp's
interaction with nonstationary RL training dynamics is not represented in a
single linear solve. The identification is not refuted, but its first concrete
prediction failed; any stated claim must say so.

**Bridge run 2 (2026-06-07, hybrid + angle sweep): wide-cut + sharp-transverse-cut
SUPPORTED, with a confound to resolve.** (a) Hybrid arm α·diag(|w|⁴)+(1−α)·M·G
beats diag-only in 7/9 cells, mean +6.5% relative test-MSE (gains concentrated at
n=512 — the scarce-data regime where constraint intersection should matter; n≥2048
gains within noise). (b) `--angle-sweep`: varying RFF bandwidth σ_w ∈ [0.5, 4]
moves the diag-vs-Gram Frobenius angle across 64.8–79.3° and the hybrid benefit
tracks it — Spearman(angle, benefit) = **+0.63** (n=20); at the widest angles
(σ_w=0.5, ~78–79°) benefit reaches +24–45%, at the narrowest (~65–70°) it vanishes
or goes negative. Matches the multi-kernel prediction that benefit grows with
constraint misalignment, and the measured angles overlap the 60–71° reward/dynamics
Hessian range. CONFOUND: angle is nearly a deterministic function of σ_w in this
design, so "angle causes benefit" is not separated from "bandwidth causes benefit";
within fixed σ_w the across-seed angle variation is too small to test. Deconfounding
needs an angle knob at fixed bandwidth (e.g., anisotropic W draws). Practical note:
σ_w=0.5 also has the best absolute MSE — low bandwidth + hybrid is the recommended
closed-form recipe pending the deconfound.

**Run 2b (2026-06-08, `--angle-deconfound`): angle causation NOT supported at fixed
bandwidth.** Anisotropic W draws (stretch along a random direction by γ ∈ [1,8],
mean |w|² held fixed) move the diag-vs-Gram angle only 72.9–79.2° and within that
range Spearman(angle, hybrid benefit) = **−0.12** (n=9) vs +0.63 confounded. The
benefit DOES vary with γ (anisotropy degrades it), just not through the angle.
Reading: low bandwidth causes both the wide angle and the hybrid benefit; the angle
is a correlate, not the lever. Caveat: anisotropy is a weak instrument here (~6° of
leverage vs run 2's ~15°), so this is "not supported", not "refuted". The practical
recipe (low σ + hybrid/ladder) is unaffected — only the causal story changes.

**Bridge run 3 (2026-06-07, `--recipe`): sigma parameterized over the transform ×
lambda polynomial is the winning recipe — user's suspicion confirmed.** Multi-scale
RFF frame (log-spaced σ ladder 0.25–2.0 across feature blocks) vs single σ=0.5
baseline, n∈{512, 2048}, 5 seeds: ladder alone +6.6% (6/10 wins); ladder + λ
polynomial **+33.7% (10/10)**; + Gram transverse cut **+36.8% (10/10)**. Mechanism:
the polynomial needs real band separation to act on — a single σ gives a narrow |w|
spread, the ladder gives the polynomial distinct bands to dose (winning shapes:
high-clamp, quartic+sextic — suppress high-σ blocks hard, spare low). The Gram cut
adds ~3pp, always at low α (sharp cut as minority partner). The interaction term,
not either ingredient alone, carries the effect. PORTED TO THE TRAINER (2026-06-07):
`spectral.sigma_w` accepts a list (sigma ladder over feature blocks; scalar path
bitwise-unchanged, ladder survives checkpoint resume via saved W);
`configs/experiment/spectral_ladder.yaml` = the run-3 recipe as a preset arm. RL
validation still pending: compare spectral_ladder vs single-sigma spectral control
vs sched-* arms on Pendulum, >= 3 seeds — the supervised +33.7% is NOT yet an RL
claim.

**Bridge run 4 (2026-06-08, explicit SNR / Wiener weights): hand-tuned polynomial
still wins; the sigma=1 hypothesis is contradicted as stated.** Implementation:
`snr_band_weights` (models/spectral.py) — split-half cross-fitted band SNR, Wiener
weights theta=(N/M)/SNR, cutoff at SNR=1 (the Tier-1 Wiener identity made
load-bearing); Trainer `spectral.weights_mode=snr` logs `spectral/sigma_at_snr1`.
Two findings: (1) the naive per-feature split-half estimate is BROKEN by feature
correlation — low-frequency signal leaks into high-band coefficients consistently
across halves, faking SNR >> 1 in dead bands (first attempt: -152% vs baseline);
the incremental-residual estimator (band SNR measured on the residual after lower
bands are fit) fixes the failure mode but the parameter-free Wiener arm still
trails the validation-swept polynomial: 5/10 wins, mean -23.8% vs single-sigma
baseline (vs ladder_poly +33.7%). Zero-hyperparameter vs 20-config-sweep is not a
matched comparison, but the claim "SNR weights beat hand tuning" is NOT supported.
(2) The measured SNR=1 crossing sits at sigma_eff = 0.207 +- 0.007 (10 cells,
remarkably stable) — NOT at sigma = 1. The user's hypothesis is falsified in this
setting; note the crossing is a property of data + noise level (NOISE_SIGMA=1
here), not a universal constant. The tight crossing DOES cohere with everything
else: signal lives below sigma ~ 0.25, which is why sigma_w=0.5 was the best
single bandwidth (run 2) and why high-clamp poly shapes win (run 3).

**Bridge run 5 (2026-06-08, SNR-calibrated ladder): new supervised champion.**
Recipe: measure sigma* (the SNR=1 crossing) on the training cache with a wide
probe basis, place ladder rungs at sigma* x mults, keep the validated lambda
polynomial as the penalty — SNR machinery for measurement, polynomial for
control. Head-to-head (10 cells): calibrated cal_low (mults 0.5/1/2/4) **+48.3%
vs single-sigma baseline, 10/10** — beats the hand ladder (+33.7%), the Gram
hybrid (+36.8%), and the higher placement (+38.8%). Trainer support:
`spectral.sigma_w: auto` (lazy calibration at first refit, sigma* logged as
spectral/sigma_star, ladder frozen + checkpointed, resume rebuilds the
calibrated basis exactly — covered by tests). Preset arm: spec-auto in
colab_spectral (`+experiment=spectral_auto`); spec-ladder is its fixed-ladder
control. RL validation pending, same criterion as run 3's.

**Learned bandwidths + probabilistic dynamics (user, 2026-06-08 — both
implemented, both untested).** (a) `spectral.sigma_w: learned` — no manual
placement or clamp: per-block log-scales trained by gradient on the reward fit
error through the cos features ("gradients flow through the scaled pipes");
closed-form c re-anchors on the moved basis each refit; scales logged as
spectral/sigma_scale_k, checkpointed with optimizer state. Risk to watch: the
gradient may drive scales toward overfit-friendly high bandwidths between
refits — same failure family as the schedule rule above; the smooth floored
schedule still applies to the poly weights. (b) `model.dynamics: gaussian` —
state probability transitions p(z'|z,a) = N(mu, diag sigma^2(z)), NLL-trained,
imagination rolls out rsamples. DESIGN GUARD: the mean stays affine in action
and the variance head is state-only, so R15's d^2/da^2 = 0 is preserved; a
full-MLP mean was deliberately not offered (it would reintroduce the
dynamics-curvature floor the affine choice removed). spec-learned arm added to
colab_spectral. Neither has supervised or RL evidence yet — conjecture tier
until the arms run.

**Bridge run 6 (2026-06-08, PRE-REGISTERED before results): the scattering
(rational) head.** Motivated by the math project's HP-candidate-11 note (the
Eisenstein scattering matrix is a RATIO whose pole structure carries the
spectrum — established Pavlov–Faddeev/Lax–Phillips mathematics; the numerical
"validation" there confirms a theorem, not a conjecture). Architecture
question: R = N/D (RationalSpectralReward, SK closed-form iterations, matched
M=512 and sweep budget vs the linear ladder+poly champion) on two target
families — smooth (rich_reward) and resonant (3 bounded spikes placed ON the
data manifold). Criteria fixed in scripts/scattering_head_test.py before any
cell ran: (i) rational wins resonant majority (overall AND near-spike MSE);
(ii) no tax on smooth; (iii) resonance-recovery contrast > 3 (the model's
1/|D| peaks land on the true spike centers). FALSIFIER: (i) fails ⇒ NOT
SUPPORTED regardless of (iii).

**RESULTS (2026-06-08): NOT SUPPORTED.** Three versions, all recorded:
v1 — SK collapsed to the degenerate attractor N≡0, D≡0 under target noise
(95.7% D-clamp rate); v2 — den_anchor=1 fix kills the collapse (0.01% clamp)
but crushes real resonances too; v3 — anchor swept on validation (rational
sweep 2x the linear arm's, noted): resonant 2/10 overall, **0/10 near-spike**
(0.491 vs 0.320), recovery contrast 1.2 (criterion >3), smooth 4/10 (no tax —
criterion ii holds). The head itself is sound — it beats linear on a PLANTED
strongly-rational target with pole recovery (unit test) — so the negative
result is about the task class: BOUNDED on-manifold spikes (A·eps/(eps+d²),
max 3) under sigma=1 noise are absorbed fine by the linear multi-scale frame;
the rational form's capacity doesn't pay and its denominator is one more thing
to estimate from noise. Possible follow-up (NEW pre-registration required, not
a retry): near-unbounded spikes / discontinuous goal bonuses, where linear
features provably ring. Structural echo for the parent project: a ratio's
power lies in representing ACTUAL poles; where the ground truth is bounded,
the scattering form is decoration — consistent with the HP-11 note's own
caveat (the identification concerns analytic structure, not ratios being
generically better approximators).

**Bridge runs 7–9 (2026-06-08, PRE-REGISTERED before results — the next
architecture iteration).**
- **Run 7 — sign-coherence cone (`--cone`):** the clamp is a CONSTRAINT, not a
  penalty; closest convex realization is ΔR(x_n) single-signed across the data
  (both signs tried, validation picks; active-set penalty iterations,
  violation fraction logged). Criterion: cone < frobenius_diag <
  lap2_indefinite in a majority of the 9 run-1-protocol cells. FALSIFIER: cone
  fails to beat frobenius_diag ⇒ both positivity-flavored closed forms
  (penalty AND constraint) are dead and the bridge prediction has no remaining
  closed-form candidate. **RESULTS (2026-06-08): NOT SUPPORTED — 0/9; cone
  loses to Frobenius by 5–20× in every cell.** Caveat: residual constraint
  violations 3–26% (penalty iteration stopped short of the exact QP), but
  tighter enforcement only constrains further — the direction is decisive.
  PER PRE-REGISTRATION: the Weil-positivity bridge prediction now has NO
  remaining closed-form candidate; it moves to the RL loop (clamped-trace vs
  positivity arms inside training) or retires. The supervised proxy program
  for the clamp is closed.
- **Run 8 — rational head, honest regime (`scattering_head_test.py
  --extreme`):** new targets where linear features provably ring — `goal`
  (discontinuous indicator bonus on a manifold ball) and `sharp` (near-pole
  spikes, eps=0.002, height 30). Criteria as run 6 (i)/(iii) on the new
  families; run-6 numbers stay frozen in their own results file. FALSIFIER:
  rational loses near-discontinuity too ⇒ the scattering form is retired for
  reward modeling at any sharpness, and the linear frame + symlog is declared
  sufficient. **RESULTS (2026-06-08): NOT SUPPORTED — goal 3/10 overall,
  sharp 0/10.** The informative twist: on sharp targets the recovery contrast
  is 6.5 (criterion >3 PASSES — the head correctly LOCATES the poles) while
  prediction MSE loses 2× with 17.6% denominator clamping. The scattering
  form finds resonances but its denominator is too noise-fragile to exploit
  them. RETIRED for reward modeling per the falsifier; kept as a possible
  resonance DETECTOR (measurement, not control — the same division of labor
  as runs 4/5). Linear multi-scale frame + symlog declared sufficient.
- **Run 9 — dynamics iteration (`--preset dynamics_ablation`, RL, Pendulum,
  3 seeds):** affine (control) vs gaussian (NLL + rsample imagination) vs
  full-MLP mean (DELIBERATE R15 break, ablation-only class). Criteria:
  (a) gaussian ≥ affine on final return with better dyn calibration
  (predicted σ vs realized error correlation > 0.5); (b) R15 binds: full-MLP
  shows the predicted imagined-return variance blowup / worse return at
  matched dose. FALSIFIER for (b): full-MLP matches affine ⇒ the
  zero-action-curvature design claim is decorative at Pendulum scale and R15
  needs requalification on MuJoCo. RESULTS: pending.

**Spectral scheduling rule (user, 2026-06-07, from the first RL attempt):** never
pair the spectral path with step anneals or zero-touching oscillations (sin2chirp
nulls, step release). The closed-form refit has no inertia: the instant lambda ~ 0,
the next refit IS the unregularized interpolator — observed as the reward fit going
"too good" (overfit to replay noise). Spectral arms use smooth floored decay only
(cuberoot, floor 1e-5; baked into spectral_ladder.yaml and the colab_spectral
preset guards). Companion rule: latent k capped at 1x obs_dim for spectral runs
(model.latent_cap_mult=1) — the closed-form fit over-resolves wide latents; the
4x cap (rule v2) stands for the MLP path.

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
