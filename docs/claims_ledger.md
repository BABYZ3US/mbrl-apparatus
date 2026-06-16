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
  needs requalification on MuJoCo.

  **Attempt 1 (2026-06-08): CONFOUNDED — no adjudication.** Arms ran at mixed
  budgets (launch predates the 20K preset trim): affine ~45K steps, gaussian
  100K/100K/20K, mlp 20K×3. Provisional observations, not verdicts: returns
  statistically flat (affine −1060±27, mlp −1123±116, gaussian −1153±178);
  gaussian's imagined-return variance FOUR orders lower (late-median 1.4e3 vs
  ~7e7; max 5.5e5 vs affine 2.4e15) — the mechanism without the return
  payoff; the R15 variance prediction looked inverted (affine worst max
  spike) but affine ran 2× longer, so uninterpretable. Calibration
  (criterion a) was UNINSTRUMENTED in attempt 1 — now logged every update as
  dyn/calib_corr, dyn/calib_ratio (~0.8 = well-calibrated Gaussian),
  dyn/pred_std. Attempt 2: matched 20K, same seeds, criteria unchanged.

  **Attempt 2 RESULTS (2026-06-08, seeds 3–6, all arms at 20K): (a) NOT met,
  (b) falsifier TRIGGERED, one effect replicated.** Returns: affine −1214±52,
  gaussian −1240±57, mlp −1193±72 (sem, n=4); pairwise winrates 0.44
  (gaussian vs affine) / 0.56 (mlp vs affine) — parity. (a) fails on the
  return half; its calibration half PASSES: dyn/calib_corr 0.55–0.58 > 0.5
  on every seed, calib_ratio 0.54–0.73 (mildly overconfident σ) — the
  Gaussian's variances are meaningful, not noise. (b) fails ⇒ per the
  pre-registered falsifier, R15's zero-action-curvature claim is DECORATIVE
  at Pendulum scale; requalification on MuJoCo at matched dose required
  before R15 is cited again as load-bearing. REPLICATED across both
  attempts: gaussian imagination suppresses late imagined-return variance by
  4–6 orders of magnitude (attempt 2 medians: 1.9e3 vs mlp 1.1e8 vs affine
  3.1e9) — better variance control than the affine constraint itself, which
  was R15's selling point — still without a return payoff. POWER CAVEAT: the
  20K Pendulum convention historically tops out ≈ −1000 (solved > −200);
  return criteria had ~no power — this convention ranks smoothness recipes
  but cannot adjudicate dynamics architectures. Follow-up (new
  pre-registration required): 60–100K or HalfCheetah GPU + a fixed-σ
  noise-injection arm to separate rsample-averaging from logvar-damping as
  the variance mechanism.

**Run 12 (2026-06-08, research-loop cycle 1 — approximation-theory
candidates): both NOT SUPPORTED; champion stands.** Candidate A, orthogonal
random features (Yu et al. 2016 — unbiased, lower variance, norms preserved
so ladder/poly untouched): 12/20 wins but mean +1.1% (bar: > +2%) with a
−21.0% worst cell (bar: > −20%) — a real but sub-threshold effect, the
variance reduction does not survive validation-sweep selection at M=512.
Candidate B, Donoho–Johnstone universal-threshold shrinkage: DROPPED PRE-RUN
with the failure mode pinned by a test — DJ requires an orthonormal basis;
in the correlated RFF design, cancellation pairs ring when one side is
zeroed (MSE 0.05 → 178 on the planted case). Third instance of the same
meta-lesson (runs 4, 6/8, 12B): per-component statistics demand an
orthogonalized or incremental measurement frame. Cycle-2 queue: leverage-
score feature sampling (Bach 2017 line), shrinkage in the Φ-SVD basis
(adaptive TSVD / Rosasco spectral filtering — the theoretically correct
form of B).

**Run 12B (2026-06-14, PRE-REGISTERED before results — research cycle 2,
the corrected candidate B): shrinkage in the Φ-SVD basis (adaptive TSVD /
Rosasco spectral filtering).** Candidate B (Donoho–Johnstone universal-
threshold shrinkage, `shrink_coefs`) was dropped pre-run because DJ near-
minimaxity requires an ORTHONORMAL basis and the correlated RFF basis rings
(tests/test_spectral.py pins MSE ×3000 — cancellation pairs). The cycle-2
fix, named in the queue above: do the shrinkage in the orthonormal basis the
problem actually provides — the economy SVD of the RFF design Φ = U S Vᵀ.
U is orthonormal, so β = Uᵀy carries iid target noise per component (the
exact DJ setting), and reconstruction uses a Rosasco spectral filter on the
singular values (here the Tikhonov filter s/(s²+λ), which damps small-s
directions by construction — stable at any conditioning; the penalty-
whitening route was tried and rejected for blowing up when poly weights → 0).
Web-searched first: Rosasco spectral-filtering family (MIT 9.520 class07;
Wikipedia "Regularization by spectral filtering" — Tikhonov/Landweber/TSVD
as filters g(s)); Donoho–Johnstone 1994 universal threshold τ = σ√(2 log n),
MAD/0.6745 noise estimate. Implementation: `spectral.svd_shrink_fit`
(kappa=0 ≡ scalar Tikhonov ridge, verified in tests/test_spectral.py;
kappa=1 = DJ universal threshold). Arms on the calibrated champion form
(cal_low ladder, run-6 smooth+resonant targets, n∈{512,2048}×5 seeds×2
targets = 20 cells): champion (poly-band ridge sweep, SHAPES×LAMS=12) vs
svd_shrink (SVD spectral filter, LAMS×KAPPAS=12 — MATCHED budget; kappas
0/1/2). Harness: `scripts/svd_shrink_test.py` (sha-scoped, chunked --budget,
resumable). PRE-REGISTERED CRITERIA (ledger default bar, fixed before any
cell ran): svd_shrink ships iff it beats champion in a MAJORITY of the 20
cells, mean relative test-MSE > +2%, worst cell > −20%. FALSIFIER: bar not
cleared ⇒ NOT SUPPORTED; the corrected candidate B is closed (4th
orthonormal-frame instance: runs 4, 6/8, 12, 12B) and cycle 2 moves to
leverage-score feature sampling.

**RESULTS (2026-06-14, results/bridge/c1b9144/): NOT SUPPORTED — and the DJ
shrinkage was never selected.** svd_shrink vs champion over 20 cells: wins
6/20, mean −69.4%, worst cell −945% — fails all three bars. The informative
twist: **kappa=0 (pure Tikhonov spectral filter, NO shrinkage) was the
validation pick in all 20/20 cells** — the DJ universal threshold (kappa=1)
and the conservative variant (kappa=2) never beat plain ridge on validation,
so the actual cycle-2 ingredient (orthonormal-basis shrinkage) contributed
nothing here. Diagnosis: smooth/resonant rewards are NOT sparse in the Φ-SVD
basis (energy spread across many singular directions), so soft-thresholding β
removes signal; the orthonormal frame fixes the *ringing* failure of
`shrink_coefs` (test pins it, ×3000 → bounded) but sparsity, the property DJ
needs, is absent. The arm therefore reduces to scalar Tikhonov ridge in the
SVD basis, which loses to the hand-tuned poly-band recipe exactly where the
recipe is strong: per-target — smooth 0/10, mean −140% (champion dominates);
resonant 6/10, mean +1.0%, and resonant n=2048 a clean 5/5, mean +10.8%
(worst +3%) — i.e. plain spectral filtering only ties/edges the recipe in the
data-rich resonant regime, never on smooth or scarce-data cells. Per the
pre-registered falsifier the corrected candidate B is CLOSED (4th
orthonormal-frame instance). Upshot, consistent with runs 6/8: the linear
multi-scale frame + the validated poly recipe is sufficient; principled
spectral filtering does not pay against it. Cycle 2 now moves to leverage-
score feature sampling (Bach 2017 line).

**Run 13 (2026-06-15, PRE-REGISTERED before results — research cycle 2, the
final queued candidate: leverage-score feature sampling.** The cycle-2 queue's
last item, after candidate A (ORF, run 12 — sub-threshold) and candidate B
(Φ-SVD shrinkage, run 12B — NOT SUPPORTED). Idea (Bach 2017, "On the
Equivalence between Kernel Quadrature Rules and Random Feature Expansions"): the
number of random features needed to match full-kernel performance is governed by
the kernel's LEVERAGE function; importance-sampling features from the leverage-
tilted distribution (vs iid from the base spectral measure) needs provably
fewer. Web-searched first: Bach 2017 (JMLR v18, leverage function in Fourier
space); Rudi & Rosasco 2017 (generalization of RFF learning, Ω(√n log n)
features, data-dependent sampling improves it); Rudi-Camoriano-Rosasco 2018
(fast empirical-leverage sampling). Implementation: `spectral.ridge_leverage_
scores` (per-feature ridge leverage l_j = diag(Φ(ΦΦᵀ+λI)⁻¹Φᵀ) via the N×N Gram
solve; push-through identity is the test oracle) and `spectral.leverage_sample`
(importance-sample N_FEATURES of a POOL_MULT× pool WITHOUT replacement ∝ l_j on
the LABEL-FREE training design, rebuild the head). ONE CHANGE vs champion: the
feature SET only — the calibrated cal_low ladder, poly-band penalty, closed-form
ridge, and the SHAPES×LAMS=12 validation sweep are byte-for-byte identical; only
the frequencies populating the basis differ (leverage-tilted vs iid). Arms on
the calibrated champion form (run-6 smooth+resonant targets, n∈{512,2048}×5
seeds×2 = 20 cells): champion (iid RFF, control) vs leverage. Harness:
`scripts/leverage_sample_test.py` (sha-scoped results/bridge/<sha>/, chunked
--budget, resumable). PRE-REGISTERED hyperparameters (fixed before results,
chosen on a LABEL-FREE dry run — no test MSE consulted): POOL_MULT=4 (pool=2048);
LAM_LEV=1.0 (central, clearly selective: leverage CV 1.0–1.8, d_eff 5.5–116
across the 4 regimes). RECORDED CONTEXT for honest interpretation: the reward's
effective dimension is SMALL — d_eff ≪ M=512 in every regime — so at the matched
M=512 budget the low-rank signal may already be over-covered by iid features;
leverage's theoretical edge is at M ≈ d_eff (a reduced-budget feature-efficiency
question this matched-budget test deliberately does NOT ask, to honor the
one-change/matched-budget rule). PRE-REGISTERED CRITERIA (ledger default bar,
fixed before any cell ran): leverage ships into the champion config iff it beats
champion in a MAJORITY of the 20 cells, mean relative test-MSE > +2%, AND worst
cell > −20%. FALSIFIER: bar not cleared ⇒ NOT SUPPORTED, recorded; the cycle-2
supervised queue is then EXHAUSTED (candidates A, B, and leverage all closed)
and the loop moves to a fresh literature pass / the RL-loop questions.

**RESULTS (2026-06-15, results/bridge/f8e9219/): NOT SUPPORTED — fails all
three bars.** leverage vs champion over 20 cells: wins 10/20 (NOT a majority),
mean −3.2%, worst cell −93.6% (smooth n=2048 s0: champion 0.0177 vs leverage
0.0343). Selection was genuinely leverage-tilted, NOT a flat-leverage null: per-
cell lev_cv 1.0–2.0, d_eff 5.5–116 (mean 12.5). The informative split is per-
target — smooth: wins 3/10, mean −16.0% (champion dominates; the worst cell
lives here); resonant: wins 7/10, mean +9.6%. Reading: the reward's effective
dimension is small (d_eff ≪ M=512 everywhere), so at the matched budget the iid
ladder already OVER-covers the low-rank smooth signal and concentrating the 512
features by leverage only removes the recipe's useful even band coverage (smooth
loses, including a near-2× worst cell); where structure is localized (resonant)
leverage's concentration helps and it edges the recipe — the SAME resonant-only
pattern as run 12B's spectral filter (resonant +1.0%, n=2048 +10.8%). On the
pre-registered bar it does not ship. Per the falsifier, the cycle-2 SUPERVISED
queue is now EXHAUSTED: candidates A (ORF, run 12 — sub-threshold), B (Φ-SVD
shrinkage, run 12B — NOT SUPPORTED), and leverage (run 13 — NOT SUPPORTED) are
all closed. Consistent meta-result across runs 6/8/12/12B/13: the linear multi-
scale frame + the validated poly recipe is sufficient at matched budget;
smarter feature selection (orthogonalization, coefficient/spectral shrinkage,
leverage placement) does not beat it. NOT pre-registered, hypothesis-tier
follow-up if the thread is revived: leverage's theoretical payoff is feature
EFFICIENCY (match accuracy at M ≈ d_eff ≈ 10, far below 512) — a reduced-budget
M-sweep, not a matched-M test, would be the fair question (new pre-registration
required). Cycle 2's actionable supervised work is done; the loop now moves to a
fresh literature pass and the still-open RL-loop items (run 10 vae arms; the
mlp-recipe anchor regression).

**Runs 10–11 (2026-06-08, PRE-REGISTERED — the VAE/transformer generation,
user-proposed).**
- **Run 10 — VAE encoder (`vae_ablation` preset: champ-vae vs champion-ctl,
  HalfCheetah 200K, ≥3 seeds).** Hypotheses: recon+KL grounding makes encoder
  collapse structurally impossible AND the KL pull toward N(0,I) makes the
  latent near-stationary, so the spectral basis stops chasing a moving
  coordinate system. Criteria: (a) return parity or better vs champion-ctl;
  (b) STATIONARITY: spectral/sigma_star drift and spectral/recal_rebuilds
  collapse vs control; (c) z_std healthy. Risk on record: reconstruction
  wants reward-irrelevant detail in the latent (against the 1×-cap lesson) —
  β=1e-3 + encoder_aux kept on for parity; β sweep only if (a) fails.
  FALSIFIER: return tax > seed noise with no stationarity gain ⇒ VAE retired
  for state-based tasks (revisit at pixels). RESULTS: pending.
- **Run 11 — transformer dynamics on (μ,σ) tokens (design locked, build
  gated on run 10's winner).** Tokens = the per-step normal map (μ_t, σ_t)
  plus action; causal transformer over the time sequence predicts the next
  normal map (decoder = MLP head over state-vector params; the
  transposed-conv "normal map" decoder is reserved for the PIXEL variant —
  deconvolution presumes spatial structure the 17-dim state lacks). Criteria
  to fix at build time: return, imagined-variance, calibration, R15 check at
  cheetah scale, and ≥1 memory-dependent task (the Markov-failure regime is
  the predicted payoff; without it the transformer is pure overhead at 200K
  steps). One change at a time: run 11 builds on run 10's winning encoder.

**gpu_spectral 6-arm RL validation — RESULTS (2026-06-12, adjudicated from
results/runs JSONL): NOT SUPPORTED, and the batch is apparatus-confounded by
the anchor.** Arms champion / spec-auto / spec-ladder / spec-single /
spec-learned / mlp-recipe, HalfCheetah-v5, 200K env steps, 3 seeds each; all
post-encoder_aux-fix (spectral/aux_loss logged, latent/z_std 0.50–0.93 — no
collapse). Pre-registered rule (improvement plan #1, runs 3/5): spec-auto ≥
spec-ladder > spec-single WITH the mlp-recipe anchor reproducing +98 ± 23.
Final returns (mean of last-3 evals per seed; mean ± sd over seeds):
spec-ladder −24.5 ± 410.0, spec-learned −117.6 ± 145.6, spec-single
−187.5 ± 178.7, mlp-recipe −188.9 ± 90.8, spec-auto −208.5 ± 96.9, champion
−309.5 ± 56.8. Adjudication: (i) spec-auto ≥ spec-ladder FAILS (−208.5 vs
−24.5; seedwise winrate 4/9 = parity) — the supervised +48.3% calibration
edge does not appear at this scale/power; (ii) spec-ladder > spec-single
holds on point estimate only (winrate 6/9; the ladder's sd 410 is one seed
at +443, the other two lose to single's mean); (iii) ANCHOR FAILED
decisively: mlp-recipe −188.9 ± 90.8 vs required +98 ± 23 — it lands in the
ORIGINAL BASELINE band (−165 ± 41) even though the λ schedule verifiably
executed (λ = 0.5 → 1e-5 step at the schedule midpoint, per JSONL) and
DreamSmooth was on (note: smoothing.sigma = 1.5 here vs σ_t = 1 in the
original — a candidate discrepancy). PER THE RULE, NO SOFTENING: the
supervised +48.3% does NOT become an RL claim. Because the apparatus
regression test failed, absolute comparisons to historical numbers are VOID
for this batch; the arm-vs-arm reads above stand only with the stated power
caveats. ROOT-CAUSE REQUIRED BEFORE RELAUNCH: why does this trainer's
recipe arm reproduce the original baseline instead of the original result
(suspects: smoothing σ 1.5 vs 1.0, eval protocol, Hutchinson probe count,
imagination/value config drift). Secondary, NOT pre-registered (hypothesis
tier only): champion — spec-auto + gaussian dynamics — was the WORST arm,
losing 7–9/9 seedwise to every other arm; since it differs from spec-auto
chiefly by the gaussian head, this connects to run 9's "mechanism without
return payoff" and makes the MuJoCo requalification urgent before gaussian
stays in the champion config. Imagined-return variance ordering (late
median): mlp-recipe ~3e2 ≪ spec arms ~1e11 ≪ champion ~2.6e15.

**Spectral encoder-collapse rule (2026-06-08, from the first HalfCheetah
batch):** in spectral mode the encoder's only gradient was the dynamics MSE
(reward cache and behaviour z0 are detached by design), whose trivial solution
is near-constant z. Observed: spec-auto loss/dyn → 2e-5 (1000× below the MLP
arm) with returns at random level — the dynamics became "perfect" by emptying
the latents. Pendulum never surfaced this; MuJoCo did. FIX (default on):
`spectral.encoder_aux` trains the otherwise-bypassed MLP reward head as an
encoder-grounding auxiliary loss (no Hutchinson on it; the spectral head still
produces all rewards). `latent/z_std` is now logged every update as the
collapse early-warning. CONSEQUENCE: all spectral-arm results from the
2026-06-08 HalfCheetah batch predate the fix and are void — the batch must be
relaunched (config hash change starts fresh lineages automatically).

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

## Research cycle 3 queue — refilled 2026-06-16 (research-note pass; cycle-2 supervised queue was exhausted)

Cycle-2's three supervised candidates are all closed: A (orthogonal random features, run 12 — sub-threshold), B (Φ-SVD shrinkage / adaptive TSVD, run 12B — NOT SUPPORTED), leverage-score feature sampling (run 13 — NOT SUPPORTED). Consistent meta-result across runs 6/8/12/12B/13: at MATCHED budget (M=512) the linear multi-scale frame + validated poly-band recipe is sufficient; smarter feature SELECTION (orthogonalization, coefficient/spectral shrinkage, leverage placement) does not beat it. The three candidates below are deliberately chosen NOT to re-test that exhausted axis — C3-1 changes the BUDGET axis (the fair question runs 12/13 each flagged but did not ask), C3-2 changes the SOLUTION-MOMENT/penalty axis (a structural-math translation), C3-3 is the rate-distortion band-allocation axis recorded WITH an explicit duplication-risk flag against run 4. All are supervised closed-form-harness candidates (CPU, seconds/cell; sha-scoped, chunked `--budget`, resumable, in the `scripts/orf_shrinkage_test.py` style). Each must be PRE-REGISTERED (criteria fixed and committed) BEFORE any cell runs. Web-searched and vetted 2026-06-16 (citations inline).

**C3-1 [TOP PRIORITY] — Reduced-budget feature-efficiency M-sweep (QMC + Gaussian-quadrature features).** The question runs 12 and 13 each flagged as the FAIR test and deliberately did NOT ask: not "does smarter selection beat iid at M=512" (settled — no) but "does a smarter feature CONSTRUCTION match the champion's M=512 accuracy at far FEWER features (M ≈ d_eff)?" — feature EFFICIENCY, the property these methods are actually theorized to deliver. d_eff is small everywhere (run 13: 5.5–116, mean 12.5 ≪ M=512), so there is large headroom to test. Two vetted constructions, both lower-discrepancy / sub-Monte-Carlo:
- Quasi-Monte Carlo feature maps — Yang, Sindhwani, Avron, Mahoney, ICML 2014 / JMLR 17 (2016): low-discrepancy (Halton/Sobol/digital-net) frequency points instead of iid draws; provably lower kernel-approximation error per feature for shift-invariant kernels.
- Gaussian quadrature for kernel features — Dao, De Sa, Ré, NeurIPS 2017 (arXiv:1709.02605): deterministic frequency nodes via Gaussian quadrature in the spectral domain; sub-O(ε⁻²) sample complexity in certain regimes; sparse-ANOVA variant for structured kernels.
DESIGN (one change = feature construction + M; the matched-M rule is DELIBERATELY and EXPLICITLY relaxed here — this IS the budget question, which run 13 said requires its own pre-registration): fixed reference = champion (iid RFF, M=512, calibrated cal_low ladder + poly-band ridge + SHAPES×LAMS=12 sweep, run-6 smooth+resonant targets, n∈{512,2048}×5 seeds×2 = 20 cells). Test arms = {QMC, quadrature, iid-control} each swept over M ∈ {≈d_eff, 2·d_eff, 4·d_eff, 8·d_eff, …, 512}. Per cell record the smallest M at which each construction's test-MSE comes within tol (e.g. +5% relative) of champion@512. PRE-REGISTERED CRITERION (default bar adapted to the budget axis): a construction SHIPS as the efficient default iff it reaches champion@512 test-MSE within tol at M ≤ 256 (≥2× feature savings) in a MAJORITY of the 20 cells AND beats the iid-RFF control's M-at-tol in that same majority (so the win is the construction, not merely the low-rank signal). FALSIFIER: no construction halves the feature budget at matched accuracy ⇒ NOT SUPPORTED, and the matched-budget sufficiency result (runs 6/8/12/12B/13) is the FINAL word on the feature axis — the supervised feature-program closes for good. Honest caveat to record at run time: d_eff ≪ 512 is exactly why this could pay (headroom) AND why it might not (iid already over-covers a low-rank signal, so even M ≈ 64 iid may already match — that null is itself the informative outcome, and it would localize the recipe's value to the PENALTY, not the basis).

**C3-2 [MEDIUM PRIORITY] — Second-moment (eff_rank / CV) penalty on the closed-form solution.** Structural translation (structural-math level only; NO claim import) of two NEW math-side notes: `Appendix_C_RL_derivations.md` §C.3 and `sigma_scaling_and_entropy_balance_2026-06-15.md`. Those prove (CAS) that under band-pinning the FIRST spectral moment σ²=⟨λ⟩ and the SECOND moment (via eff_rank/d = 1/(1+CV²)) are OVER-DETERMINED by a single band fill — a two-level fill cannot match both, hence "report both, or free a second spectral moment." Translation to the supervised closed-form head: the ridge λ and poly-band weights set the FIRST moment (per-band fitted energy); nothing in the champion explicitly controls the SECOND moment (the participation ratio / CV of the band-energy profile of the fitted coefficients). NEW axis = shape the solution's second moment, not the feature set and not the per-band first-moment weight. DESIGN (one change): add a single eff_rank/CV regularizer on the band-energy profile of the closed-form coefficients, weight swept on validation alongside the existing poly λ (the CV-weight replaces one swept poly DOF so the total 12-config sweep budget is held — matched budget). MUST respect the standing meta-lesson (runs 4, 12B): per-component statistics need an orthogonalized/incremental frame — compute the band-energy profile in the Φ-SVD orthonormal basis (run 12B's machinery), NOT the correlated raw-RFF basis. Champion comparison, run-6 targets, 20 cells. PRE-REGISTERED CRITERION: ships iff it beats champion on the default bar (majority of 20, mean rel test-MSE > +2%, worst cell > −20%). FALSIFIER: the CV knob does not clear the bar ⇒ NOT SUPPORTED; the first-moment band fill is sufficient for the supervised reward fit and C.3's over-determination has no supervised payoff (the second moment matters for the RL latent spectrum, not the supervised head).

**C3-3 [LOW PRIORITY — high duplication risk, recorded for completeness] — Reverse-water-filling band allocation (Gaussian rate-distortion).** Vetted: reverse water-filling for parallel Gaussian sources (Cover & Thomas, *Elements of Information Theory*) — per-band distortion D_i = min(σ_i², θ), water level θ set by a total rate/distortion budget; bands with variance below θ are dropped, the rest get rate ∝ ½log(σ_i²/θ). HONEST PRIOR (why this is LOW priority): run 4 already tested parameter-free Wiener/SNR band weights (cutoff at SNR=1) and LOST −23.8% to the hand-tuned poly. RWF differs ONLY by (a) an explicit rate/distortion budget and (b) a SINGLE swept water-level θ — a 1-parameter family, a fairer match to the 12-config poly sweep than run-4's zero-parameter Wiener. DESIGN (one change): replace the poly-band penalty with RWF band weights from the incremental-residual band-SNR estimator (run 4's correlation-corrected machinery, NOT the broken naive split-half), θ swept on validation; matched budget; champion comparison; 20 cells. PRE-REGISTERED CRITERION (STRICTER than default, because of the prior): ships iff it beats champion on the default bar AND beats run-4's parameter-free Wiener arm. FALSIFIER: fails to clear the run-4 Wiener bar ⇒ the rate-distortion band-ALLOCATION axis is closed for the supervised reward fit (the Wiener identity stays Tier-1 theory, not a supervised recipe), and no future night should re-propose SNR/Wiener/RWF band weighting without a genuinely new mechanism.

**PARKED structural observation (NOT a supervised-queue item; logged so it is not lost).** `stein_loss_triangularization_rational_2026-06-16.md` derives the Stein / LogDet (Itakura–Saito) operator loss D(G|Σ̂) = tr(Σ̂⁻¹G) − logdet(Σ̂⁻¹G) − d, separating over the generalized spectrum as Σ_i φ(ν_i), φ(ν)=ν−log ν−1 (convex, ≥0, zero at ν=1), from one triangular factorization. This is an RL-side / dynamics-operator loss in the math note (covariance matching, ν_i=1 at calibration), NOT a regression loss — it does NOT map cleanly onto the supervised reward fit (least-squares regression, not covariance matching), so it is parked OFF the supervised queue. Natural on-axis home: the operator-dynamics / gaussian-calibration instrumentation (cf. run 9's `dyn/calib_corr`; the in-progress `tests/test_operator_dynamics.py`), where a LogDet calibration term on the predicted-vs-realized dynamics covariance is the honest use — an RL-loop item, gated like the items below.

**Cross-reference — STILL-OPEN RL-loop items (NOT supervised-queue candidates; need cloud RL; listed so the queue is complete).** (1) The mlp-recipe anchor regression — top SCIENCE priority (improvement plan #1): diff the recipe arm's effective config vs original report §2 (smoothing.sigma 1.5 vs 1.0, eval protocol, Hutchinson probe count) before any spectral relaunch. (2) Run 10 vae_ablation — still no arms in results/runs; cannot be adjudicated until the user pulls the cloud artifacts. (3) Gaussian-dynamics MuJoCo requalification — run 9's "mechanism without return payoff" + the gpu_spectral finding that champion (= spec-auto + gaussian) was the WORST arm; R15 (zero action-curvature) was ruled DECORATIVE at Pendulum scale and needs MuJoCo requalification before gaussian stays in the champion config.
