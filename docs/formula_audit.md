# Formula audit — curvature-regularized MBRL

Read-only correctness + efficiency review of the formula-bearing code in `src/mbrl/`,
audited against the founding doc (`../../mbrl_foundations_and_framework.md`), the claims
ledger (`claims_ledger.md`), and the original findings report (`original_findings_report.md`).

Scope: the Hutchinson curvature penalty, the Laplacian-trace clamp, the spectral RFF ridge
path, return normalization, and all eigen/SVD/Cholesky-class linear algebra. The 56-test
suite was read in full to record what is already pinned.

**Verdict up front:** the load-bearing math is correct and the founding-doc hard rules
(penalize R/T never π; isotropic penalty; double-backward; clamp = max(·,0); RFF
convention; DreamerV3 return norm) are all honored. No correctness bugs in the core
formulas. Findings are dominated by **efficiency** items (one redundant graph build, the
normal-equations conditioning choice) and two **minor semantic edge cases** (odd probe
counts > 2; a non-stable rank-1 product estimator for n_probes=2 — by design). Details
below, each cited to `file:line`.

---

## 1. Curvature penalty — Hutchinson 2-probe HVP (`regularization/hutchinson.py`)

### (a) It is the dynamics/reward-map Hessian, NOT the policy Hessian — CORRECT (hard rule honored)

- `hvp_penalty` (`hutchinson.py:24`) is generic in `fn`; the *call site* decides the target.
- Reward target: `loop.py:449` passes `self.reward.on_concat` over `cat(z.detach(), a[, tau])`
  (`loop.py:439-442`). `RewardModel.on_concat` (`reward.py:60`) is the head-mean on the raw
  (symlog-space) reward output. Correct target (R1).
- Optional dynamics term: `loop.py:452-458`, `penalize_dynamics` gated, default **false**
  (`base.yaml:42`). Uses the deterministic mean of the dynamics (`loop.py:454`, `.mean` for
  the Gaussian model). Correct (R8/R9).
- The **policy is never penalized**: there is no `hvp_penalty`/`laplacian_trace_penalty` call
  anywhere over policy params or actions-as-curvature-coords. `behaviour_update` comments this
  explicitly (`loop.py:608`). Confirmed by grep across `src/`. **Hard rule R10 honored.**
- The penalty does not leak into the encoder: coords are `z.detach()` (`loop.py:439`). Pinned
  by `test_core.py::test_penalty_never_touches_encoder` (`test_core.py:96`) and
  `test_hvp.py::test_penalty_differentiable`.

### (b) E[vᵀHv] = tr(H) / E‖Hv‖² = ‖H‖²_F for Rademacher v — CORRECT

- `rademacher_like` (`hutchinson.py:15`) draws {0,1}→{-1,+1} via `randint(0,2).mul_(2).sub_(1)`.
  Proper ±1 Rademacher; `vᵢ²=1` exactly, so `E[vᵀHv]=tr(H)` and `E‖Hv‖²=Σ_i(Σ_j H_ij v_j)²`,
  whose expectation over independent `v` is `Σ_ij H_ij² = ‖H‖²_F`. Standard Hutchinson — correct.
- `hvp_penalty` accumulates `‖Hv‖²` per probe (`hutchinson.py:55`: `hv.pow(2).sum(-1).mean()`)
  and divides by `n_probes` (`:56`). This is **unbiased at any N** (each probe is an independent
  unbiased draw of `‖H‖²_F`); variance falls as 1/N. The docstring at `:38-39` makes the right
  distinction: squaring `‖Hv‖²` per-probe is unbiased, unlike squaring a single trace estimate.
- Verified analytically by `test_hvp.py::test_hutchinson_unbiased_frobenius` (rel err <5% at
  N=400 vs exact `(H²).sum()`) and `::test_two_probes_default_reasonable` (mean of 50 two-probe
  estimates within 10% — unbiasedness with variance as the cost). **Correct and tested.**

### (c) Double-backward (create_graph/retain_graph) — CORRECT and NOT redundantly recomputed

- First-order grad computed **once** (`hutchinson.py:47`) with `create_graph=True`, then reused
  across all probes (`grad * v`). HVPs taken with `retain_graph=True` (`:53`) so the shared
  first-order graph survives multiple probes. This is the efficient pattern — the forward + the
  expensive first backward are not repeated per probe. **No redundant backward passes.**
- `create_graph` propagates from the arg (`:30`, default True) so the penalty is differentiable
  into model params during training; set False in tests for speed (`test_hvp.py:25`).
- Same one-grad-then-reuse pattern in `laplacian_trace_penalty` (`:82`, `:87`). Correct.

### (d) Numerical stability — CORRECT (well-handled)

- The whole penalty is forced to **fp32 even under bf16 autocast** (`hutchinson.py:42`,
  `torch.autocast(..., enabled=False)`; config note `base.yaml:150` "penalty always fp32").
  Second derivatives are noise-sensitive; this is the right call and matches the module
  docstring (`:7`).
- Probe RNG runs on a CPU generator and moves to device (`hutchinson.py:18-21`), reproducible
  across MPS/CPU/CUDA; the generator state is checkpointed (`loop.py:668`) for bitwise resume.

### Efficiency note (penalty path) — MINOR, not a bug

- `hvp_penalty` is already optimal (one first backward, retained graph, vectorized per-probe
  reduction). No Python loop over the batch; the only loop is over `n_probes` (=2), which is
  irreducible. Good.

---

## 2. Laplacian-trace penalty + clamp = max(est,0) (`hutchinson.py:59`)

### Clamp correctness — CORRECT (matches the thermodynamic-consistency / Weil-positivity finding)

- The estimator forms two **independent** probe traces `vᵀHv` (`:90`), multiplies them
  (`est = t1*t2`, `:93`) — unbiased for `tr(H)²` because the probes are independent (E[t1·t2]
  = E[t1]·E[t2] = tr(H)²). Correct realization of the `(ΔR)²` form (R5).
- The clamp is **per-sample `max(est, 0)`** (`:94-95`, `est.clamp_min(0)`), applied to the
  sign-indefinite product before the batch mean. This is exactly the apparatus consequence in
  `original_findings_report.md:116` ("clamping = max(estimator, 0) on the sign-indefinite Lap-2
  product") and `claims_ledger` (clamped +41 vs Frobenius −40). **Applied correctly.**
- Schedule default: `clamp_trace: true` (`base.yaml:38`), threaded through the trainer
  (`loop.py:422-426`) via `cfg.penalty.get("clamp_trace", True)`. Default penalty `form` is
  `laplacian_trace` (`base.yaml:35`) — i.e. the clamped trace is the active default recipe.
  Consistent with the founding doc and ledger. **Schedule honored.**
- Pinned by `test_hvp.py::test_trace_clamp_nonnegativity` (`test_hvp.py:56`): clamped ≥ 0
  always; unclamped goes < 0 on a mixed-sign rotated spectrum; clamped is intentionally biased
  up while unclamped stays unbiased for `tr(H)²=0`. The test correctly *rotates* the Hessian
  (`:64-66`) so Rademacher probes don't trivially give `vᵀHv=tr(H)` — a well-constructed test.
- `test_hvp.py::test_null_lagrangian_trace_matches_frobenius_EL` confirms the unbiased (unclamped)
  product matches `tr(H)²` in mean (R5).

### Edge case A (MINOR, semantic — not a correctness bug): the n_probes=2 product is not non-negative pre-clamp and has high variance

- For the **default** `n_probes=2`, the estimator is a single product of two trace estimates.
  This is the *correct* unbiased `(ΔR)²` estimator and the clamp is what the findings selected,
  so this is by design — but note it is a rank-1 (2-probe) product, the highest-variance member
  of the family. The ledger's own bridge runs treat this as the intended object. No action
  needed; flagging only that variance here is structurally large (the clamp trades that for the
  sign constraint, which is the whole point).

### Edge case B (MINOR — wasted probe on odd n_probes > 2): the >2-probe aggregation drops a trailing probe

- For `n_probes > 2` the code pairs probes: `est = t0·t1` then the loop
  `for i in range(2, n_probes-1, 2)` adds `t_i·t_{i+1}` (`:96-98`), dividing by `n_probes//2`
  (`:99`). I traced this exhaustively:
  - N=2 → 1 pair, ÷1. Correct.
  - N=4 → pairs (0,1),(2,3), ÷2. Correct.
  - N=6 → (0,1),(2,3),(4,5), ÷3. Correct.
  - **N=3 or N=5 (odd) → the last probe is silently computed-then-dropped.** E.g. N=3 forms
    only (0,1) and divides by 1; probe #3 was sampled and HVP'd at full cost but never used.
- **Impact:** correctness is fine (the result is still an unbiased average of paired estimates);
  this is purely a small **inefficiency** for odd probe counts above 2 — one HVP wasted. The
  default N=2 and the swept even counts {2,4,8} (validation item 3) are unaffected. Could short-
  circuit by computing only `2*(n_probes//2)` probes, or assert even. Low priority.

---

## 3. Spectral path — RFF ridge reward heads (`models/spectral.py`)

### RFF feature map — CORRECT (sqrt(2/M) cos(Wx+b) convention)

- `features` (`spectral.py:408-413`): `sqrt(2.0/M) * cos(X @ Wᵀ + b)` with `W ~ N(0, σ²I)`
  (`:360-373`) and `b ~ U[0, 2π)` (`:374`). This is the standard Rahimi-Recht single-cosine RFF
  (the `sqrt(2/D)` convention with `b` absorbing the phase) — correct.
- Seed-deterministic CPU generator (`:359`), reproducible across backends; pinned by
  `test_spectral.py::test_seed_determinism_and_predict_shape`.

### Exact H² penalty constant — CORRECT and cross-validated against autograd

- Claimed constant: `E_x‖∇²R‖²_F = (1/M) Σ_j c_j² |w_j|⁴` (`:18`, `:451-455`,
  `hessian_frobenius_sq`). Derivation in the module docstring (`:9-28`) is sound: each feature's
  Hessian is rank-1 (`∇²φ_j = -√(2/M) cos(θ_j) w_j w_jᵀ`), `‖w wᵀ‖²_F = |w|⁴`, and the cross
  terms vanish in expectation over the uniform phases (`E[cos θ_j cos θ_k] = ½δ_jk`). Correct.
- `laplacian_trace_sq` (`:457-465`) returns the identical value — correct, because rank-1
  feature Hessians make `‖H_j‖²_F = (tr H_j)²` per feature (the null-Lagrangian holds
  *exactly in expectation* in this basis, stronger than the EL-only R5). Pinned by
  `test_spectral.py::test_laplacian_equals_frobenius_null_lagrangian` (`rel=0`).
- **The load-bearing cross-check**: `test_spectral.py::test_exact_penalty_matches_autograd_hutchinson`
  (`test_spectral.py:22`) Hutchinson-estimates `E_x‖∇²R‖²_F` with 64 probes on a 4096-row batch
  and requires agreement with the closed form within 10%. This mutually validates the two
  penalty implementations against the *same* estimator the trainer uses. **Strong evidence the
  constant is right.**

### Ridge solve — numerically CORRECT but uses NORMAL EQUATIONS (efficiency/stability tradeoff — flag)

- `fit` (`:422-448`) solves `c = (ΦᵀΦ + diag(weights) + 1e-8 I)⁻¹ Φᵀy` via
  `torch.linalg.solve(A, Φᵀy)` (`:446`) where `A = ΦᵀΦ + diag(weights+1e-8)` (`:445`).
- **This is the normal-equations form.** `torch.linalg.solve` uses LU (not Cholesky), so it does
  not exploit the SPD structure, and forming `ΦᵀΦ` **squares the condition number** vs operating
  on `Φ` directly (QR / `lstsq` on the augmented `[Φ; sqrt(diag)]` system). For an `M=512`,
  well-curvature-weighted system this is fine in practice and the `1e-8` floor + the `|w|⁴`
  ridge weights keep `A` PD, but:
  - **Stability flag (not a bug):** under a small `lam` (e.g. the `1e-9` "near-interpolator"
    used in tests, `test_spectral.py:151`, and the `floor=1e-5` schedule), `ΦᵀΦ` can be
    ill-conditioned when N < M or features are correlated (the ladder deliberately correlates
    bands). The `1e-8` absolute floor is small relative to typical `ΦᵀΦ` diagonal magnitudes, so
    conditioning rests mostly on the curvature weights being nonzero. A `torch.linalg.lstsq` on
    the stacked system, or `cholesky_solve` with adaptive jitter, would be more robust and is the
    standard recommendation for ridge. The repo's own ledger documents real conditioning-driven
    failures in *adjacent* code (run-4 leakage, run-12B DJ ringing), so the project is sensitive
    to this class of issue.
  - **Efficiency:** `linalg.solve` (LU) on an SPD matrix is ~2× the flops of `cholesky`. At
    M=512 and ~0.04s/refit (`loop.py:241` comment) this is negligible, but `cholesky` (with the
    existing jitter as the factorization fallback) would be both faster and more stable — a
    free win if touched.
- **Recommendation (low priority, not blocking):** prefer `cholesky_solve`/`lstsq` over
  `solve(ΦᵀΦ+…)` for the ridge. Correctness today is fine; this is robustness + a minor speedup.
- Same normal-equations pattern in the SNR band solve (`:279`), the rational head SK solve
  (`:151`), and `shrink_coefs` (`:207`). All `M`-or-smaller, all the same tradeoff.

### Polynomial per-band weights — CORRECT

- `poly_weights` (`:55-71`): `Σ_d coefs[d] · |w|^(2·degrees[d])`, applied to `|w|²` raised to
  `degrees[d]` (so degree 2 → `|w|⁴`, the pure H² weight). Correct and matches the docstring.
  Length mismatch raises (`:65`). Pinned by `test_spectral_trainer.py::test_poly_weights_hand_values_and_shifts`
  (hand values `{1,16}` for the quartic; `{5,56}` for mixed `2|w|²+3|w|⁴`).
- Per-band weights are correctly *applied as the ridge diagonal* in the refit
  (`loop.py:290` → `head.fit(X, y, weights=...)`), with per-degree time-shifts on the λ schedule
  (`loop.py:233-239`, `theta_d(t) = coefs[d]·lam(t+shifts[d])`). The shift logic is tested
  (`test_spectral_trainer.py:108-124`). Correct.

### Sigma ladder — CORRECT (block-wise scaling, scalar path bitwise-preserved)

- `SpectralReward.__init__` (`:344-373`): scalar `σ_w` scales all of `W` (`:361-362`); a list =
  ladder scales feature block `k` by `σ_w[k]` (`:363-372`), last block absorbing the remainder
  (`:371`). The scalar path applies the scale *after* the draw, so it is bitwise-identical to the
  pre-ladder code on the same seed. Pinned precisely by
  `test_spectral.py::test_sigma_ladder_blocks_and_scalar_equivalence` (`:88`) — checks block
  scaling, scalar equivalence (`half.W == 0.5*unit.W`), shared phase stream, and that the ladder
  genuinely widens the `|w|` spread. Validation (`:111`) covers empty ladder and `n_features <
  rungs`. **Correct and well-tested.**
- `w2 = W.pow(2).sum(-1)`, `w4 = w2²` precomputed once (`:395-396`) — correct, vectorized.

### SNR-calibrated sigma ladder (`sigma_w: auto`) — CORRECT (with the documented caveat)

- `snr_band_weights` (`:215-295`): per-band Wiener weight `θ_j = (N/M)/SNR_band(j)` (`:273`),
  matching the Tier-1 Wiener identity (cutoff at SNR=1, shrinkage `SNR/(1+SNR)`). The derivation
  uses `E[ΦᵀΦ] = (N/M) I` for near-orthogonal RFF (`:222-228`) — correct for the RFF
  normalization chosen.
- The **incremental/residual** SNR estimate (`:259-281`) is the right fix for the feature-
  correlation leakage failure (documented in the ledger as bridge run 4): bands processed
  low→high, each band's SNR measured on the residual after lower bands' Wiener-shrunk fits are
  subtracted (`:276-281`). Split halves separate target noise; residualization separates
  redundant signal. This is sound and the code comments correctly explain why the naive per-
  feature estimate is broken.
- `calibrate_sigma_ladder` (`:298-331`): probes a wide log-spaced σ basis, finds the SNR=1
  crossing `σ*` by log-linear interpolation (`:288-294`), places production rungs at `σ*·mults`.
  Lazy at first refit (`loop.py:249-257`), frozen + checkpointed (`loop.py:681`,
  `test_spectral_trainer.py::test_sigma_auto_calibrates_and_resumes`). The no-crossing fallback
  (`:324-327`) is conservative (geometric middle of live bands, or 1.0). **Correct.**
- Tested end-to-end: `test_spectral.py::test_snr_band_weights_wiener_behavior` (SNR decreases
  low→high band, weights inversely track SNR, Wiener fit beats near-unregularized on noisy
  targets) and the trainer smokes for `auto`/`snr`/`learned`/`ladder`.
- The clamp's spectral analog is **correctly NOT claimed**: `spectral.py:37-40` and `:464`
  explicitly state this module implements the *unclamped* penalty and that the `max(est,0)`
  rectifier has no diagonal form here. Consistent with the ledger's open-question status and the
  bridge-run negative results. Honest.

### Minor spectral notes

- `current_W` returns the fixed `W` when not learning scales (`:404-406`) — no spurious graph.
  When `learn_scales`, the differentiable scaled `W` is snapshotted to `self.W`/`w2`/`w4` at each
  refit (`:433-438`) so the penalty weights track the moved basis. Correct.
- The learned-scales gradient step (`loop.py:379-391`) updates only `log_s` on detached `(z,a)`
  with an L2 anchor toward init (`loop.py:386-388`, the documented overfit-drift guard). Correct.
- `RationalSpectralReward` (`:74-159`) is experimental (bridge runs 6/8, retired for reward
  modeling per the ledger) and **not wired into the trainer** (`:95`); its `predict` D-floor
  guard (`:116-119`) and SK iteration (`:145-153`) are internally consistent and unit-tested
  (`test_spectral.py:157`), but out of the production path. Not load-bearing.

---

## 4. Return normalization — DreamerV3 percentile/EMA (`training/returns.py`, `loop.py:597-605`)

### Lambda-returns formula — CORRECT

- `lambda_returns` (`returns.py:16-26`): backward recursion
  `R_t = r_t + γ[(1-λ)v_{t+1} + λR_{t+1}]`, `R_H = v_H`. This is the standard Dreamer λ-return.
  Pinned **exactly** by `test_returns_and_tasks.py::test_lambda_returns_hand_computed`
  (hand-checked R0=23.05, R1=29.0) and `::test_lambda_returns_limits` (λ=0 → 1-step TD; λ=1 →
  Monte-Carlo with terminal bootstrap). **Correct and tested to the arithmetic.**
- One backward pass over the horizon, no critic ensembles — matches the docstring's cheapness
  claim. The policy gradient flows through `rs` (rewards via dynamics+reward), bootstrap values
  `v_tgt` are correctly detached (`loop.py:591-594`, `torch.no_grad` + `value_target`). This is
  the intended Dreamer dynamics-backprop; correct.

### Percentile/EMA return scaling — CORRECT (matches DreamerV3)

- `behaviour_update` (`loop.py:597-605`): computes the 5th/95th percentile of the detached
  returns (`torch.quantile(..., 0.05/0.95)`, `:599-600`), EMA-updates a running scale
  `ret_scale = decay·ret_scale + (1-decay)·span` (`:604`), and divides the policy objective by
  `max(1.0, ret_scale)` (`:605`, `:610`). This is **precisely** the DreamerV3 return
  normalization: percentile range (DreamerV3 uses the 5–95% range), EMA-smoothed, with the
  `max(1, ·)` floor that prevents amplifying small returns. **Formula matches DreamerV3.**
- Symlog/symexp (`reward.py:16-24`) is the standard DreamerV3 squashing
  (`sign(x)·log1p(|x|)` / `sign(x)·expm1(|x|)`); roundtrip pinned by
  `test_new_features.py::test_symlog_roundtrip_and_shape`. The reward model trains in symlog
  space and imagination symexps back (`loop.py:346`, `:543-551`). Correct.
- **NaN hygiene** is thorough and correct: the span EMA is only updated on finite spans
  (`loop.py:603`); the symexp clamp is data-driven (`bound = margin·running-max|symlog(r)|`,
  `loop.py:550`) replacing a fixed ±20 that overflowed — pinned by
  `test_new_features.py::test_symexp_overflow_clamped_and_nan_hygiene` and
  `::test_data_driven_symexp_clamp_and_checkpoint_roundtrip`. The scale-invariance property is
  directly tested: `test_schedule_and_stability.py::test_return_normalization_bounds_policy_gradient`
  (1000× reward scale → policy grad grows far less than 1000×). **Correct, robust, tested.**

---

## 5. Eigen / SVD / Cholesky / QR — stability + vectorization

Inventory of all decomposition-class calls (grep `linalg`/`qr`/`cholesky`/`eig`/`svd` across
`src/`):

| Call | Location | Correct? | Stability / vectorization |
|---|---|---|---|
| `torch.linalg.solve` (ridge) | `spectral.py:446` | yes | **Normal equations** (see §3): squares cond #; mitigated by `1e-8` + curvature weights. Prefer `cholesky_solve`/`lstsq`. Fully vectorized (no batch loop). |
| `torch.linalg.solve` (SNR band) | `spectral.py:279` | yes | Per-band `Mb×Mb`, ridge floor present; same normal-eq tradeoff, tiny matrices. |
| `torch.linalg.solve` (SK rational) | `spectral.py:151` | yes | Experimental head, not in trainer; `theta+1e-8` floor + `den_anchor` Gram. |
| `torch.linalg.solve` (shrink split-half) | `spectral.py:207` | yes | Helper, not in trainer; documented-failure test pins the ringing. |
| `torch.linalg.qr` (ORF) | `spectral.py:179` | yes | Uniform orthogonal frame; norms preserved exactly (tested `test_orf_preserves_norms_and_orthogonalizes`). Block loop is over `d`-sized chunks (unavoidable, small). |
| `torch.linalg.qr` (test only) | `test_hvp.py:65` | yes | Rotates the test Hessian — good test hygiene. |
| `torch.quantile` (return scale) | `loop.py:599-600` | yes | Vectorized; finite-guarded. |
| `torch.quantile` (SNR band edges) | `spectral.py:248` | yes | Vectorized band edges. |

- **No eigendecomposition or SVD on the hot path** (the spectral fit is closed-form ridge, not
  SVD). `effective_dim` and `transversality_angle` compute spectral *summaries* (`tr(H²)`,
  `tr(H⁴)`, Frobenius inner products) via Hutchinson, **avoiding** any explicit eig/SVD — this is
  the right, scalable choice (`transversality.py:17-71`).
- **No Python loop over the batch** where torch ops suffice anywhere in the linear algebra. The
  only loops are over `n_probes`, `n_bands`, ladder rungs, and `d`-sized ORF chunks — all
  irreducible and small. Vectorization is good throughout.

### `effective_dim` / `transversality_angle` — CORRECT, with one efficiency leak

- `effective_dim` (`transversality.py:17-48`): participation ratio
  `d_eff = tr(H²)²/tr(H⁴) = (Σλ²)²/Σλ⁴`, with `tr(H²) = E‖Hv‖²` and `tr(H⁴) = E‖H²v‖²` via a
  nested HVP. The **per-sample-ratio-then-median** aggregation (`:40-48`) is correct and the
  comment (`:40-43`) correctly explains why pooling traces first is wrong under heterogeneous
  curvature. Pinned by `test_schedule_and_stability` analog
  `test_effective_dim_known_spectra` (uniform→6, rank-2→2, spiked→~1) and
  `test_effective_dim_heterogeneous_curvature` (PR stays ~2, never < 1). Flat-sample masking
  (`:44`) and `1e-30` guards present. **Correct.**
- `transversality_angle` (`:51-71`): `cos α = ⟨H_r, H_t⟩_F / (‖H_r‖_F‖H_t‖_F)` via
  `E_v[(H_r v)·(H_t v)]`. Correct Frobenius-inner-product-via-Hutchinson identity; clamps cos to
  `[-1,1]` before acos (`:71`). Pinned by `test_transversality_angle_known_cases` (parallel→<10°,
  orthogonal→~90°). **Correct.**
- **Efficiency leak (MINOR, not a bug):** both diagnostics build the inner HVP with
  `create_graph=True` (`transversality.py:35-36`, `:65-66`) but then immediately `.detach()` the
  result (`:37-39`, `:67-69`) and never backprop through it — these are eval-time diagnostics
  with no `.backward()`. `create_graph=True` here builds a second-order graph that is never used,
  wasting memory and time. `effective_dim`'s `:29` first grad and `:35` HVP only need
  `create_graph` for the *nested* HVP at `:37` — and `:37` already (correctly) omits it. So
  `:35`'s `create_graph=True` is needed (the `h2v` at `:37` differentiates `g·hv`); but `:36`
  in `transversality_angle` (the angle, single HVP, no nesting) does **not** need it.
  Net: `transversality_angle` (`:65-66`) can drop `create_graph` entirely; `effective_dim`'s
  `:35` is load-bearing for the nested derivative and should stay. Low priority (diagnostics,
  not the training hot path), but a clean memory/time win if touched.

---

## Cross-cutting: founding-doc hard rules — all honored

| Hard rule | Where enforced | Status |
|---|---|---|
| Penalize R (and optionally T), **never π** (R10) | `loop.py:449` (R), `:452-458` (T, default off), no π anywhere | HONORED |
| Penalty in **latent coords** (exp 2.3) | `cat(z.detach(), a)` (`loop.py:439-442`) | HONORED |
| **Isotropic** penalty (R16) | Hutchinson `‖Hv‖²` (no eigen-weighting); spectral `(1/M)Σc²|w|⁴` is rotation-symmetric | HONORED |
| **Unbiased 2-probe** Hutchinson (R4/R5) | `hvp_penalty` ÷N unbiased; N=2 default (`base.yaml:35`) | HONORED |
| Clamp = **max(est,0)** per-sample, default on (findings §3) | `hutchinson.py:94`; `clamp_trace: true` (`base.yaml:38`) | HONORED |
| Affine-in-action dynamics, `∂²T/∂a²=0` (R15) | `AffineDynamics` (`dynamics.py:21-23`); Gaussian mean stays affine, variance state-only (`dynamics.py:26-62`); MLP arm is a loudly-warned ablation (`dynamics.py:65`, `loop.py:60-62`) | HONORED |
| Spectral: smooth floored schedule only, latent cap 1× | warned at construction (`loop.py:207-220`); presets enforce (`spectral_auto.yaml`, `presets.yaml:23`) | HONORED |

The affine zero-action-curvature property is pinned exactly by
`test_core.py::test_affine_dynamics_zero_action_curvature` (second difference in `a` < 1e-5).

---

## Prioritized summary

**Correctness issues:** none found in the load-bearing formulas. The Hutchinson penalty (target,
unbiasedness, double-backward, fp32), the Laplacian clamp, the RFF map + exact penalty constant,
the polynomial/SNR band weights, the λ-returns, and the DreamerV3 percentile/EMA return
normalization are all correct and (with one exception below) directly tested against analytic
ground truth.

**Efficiency / robustness (in rough priority):**
1. **Spectral ridge uses normal equations** (`spectral.py:446`, and `:279/:151/:207`):
   `solve(ΦᵀΦ + diag, Φᵀy)` squares the condition number and uses LU on an SPD system. Correct
   today (floor + curvature weights keep it PD) but `cholesky_solve` or `lstsq` on the stacked
   system would be more stable and slightly faster — the better default for ridge, especially at
   small `lam`/`floor` where `ΦᵀΦ` can be ill-conditioned with correlated ladder bands.
2. **`transversality_angle` builds an unused second-order graph** (`transversality.py:65-66`,
   `create_graph=True` then immediate `.detach()`): drop `create_graph` for a free memory/time
   win. Diagnostic path only. (`effective_dim`'s `:35` is load-bearing — leave it.)
3. **`laplacian_trace_penalty` wastes one probe on odd `n_probes>2`** (`hutchinson.py:96-99`):
   the trailing probe is sampled + HVP'd but dropped. Result stays correct; just compute
   `2·(n_probes//2)` probes or assert even. Does not affect the N=2 default or the even sweep.

**Test coverage is strong (56 tests):** the penalty math, the clamp non-negativity, the
null-Lagrangian equivalence (both autograd and closed-form), the RFF↔autograd cross-check, the
ladder/SNR/auto/learned spectral paths, the λ-returns arithmetic and limits, the return-scale
invariance, symexp overflow hygiene, and bitwise checkpoint resume are all pinned. No untested
load-bearing formula was found except the normal-equations conditioning, which is exercised
indirectly (fits succeed) but not stress-tested for ill-conditioning.
