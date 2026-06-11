# Algorithm correctness review — 2026-06-11 (pre-campaign-2 gate)

PM request: before the next experiment, verify the algorithms are correctly
applied and reproduced. Method: each implementation read against its canonical
formulation; the test suite checked for whether it pins the *right identity*
(not just "runs"); gaps fixed in this pass. The RUNNING campaign's path was
audited first.

## The running arm's path (PLAN row 11) — VERDICT: correct

| Piece | Canon | Finding |
|---|---|---|
| λ-returns (`returns.py`) | Dreamer: `R^λ_t = r_t + γ[(1−λ)v_{t+1} + λR^λ_{t+1}]`, `R^λ_H = v_H` | exact, incl. the boundary (`last = values[-1]` seeds the recursion). No terminal masking — correct for fixed-horizon imagination (no terminals exist) |
| Ensemble members (`ensemble.py`) | PETS/MBPO deep ensemble, independent inits | members are affine-in-action (`z' = f(x) + B(x)a`) — deliberately NOT the PETS full-MLP, preserving the R15 hard rule (sum of affine maps is affine, pinned by the affinity-identity test) |
| Disagreement | std across members | per-dim std averaged over latent dims — a standard epistemic proxy; positivity pinned |
| Member training | deep-ensemble discipline | every member regresses to the EMA target (mean-only fitting would leave disagreement unregularized — pinned: all member params move) |
| Pessimism placement | MOPO: `r̃ = r − λ·u(s,a)` | applied per-step at `(z_t, a_t)` to the imagined reward BEFORE stacking → smoothing → λ-returns; behaviour-side only (the model fit never sees it); pinned: fixed-seed pessimistic returns sit strictly below neutral. Note: DreamSmooth smooths the penalized stream — a deliberate ordering (the penalty is part of the imagined reward signal) |
| Composition | — | reward-head pessimism (`imagination.pessimism`, ensemble of reward heads) and dynamics-disagreement pessimism (`algo.ensemble_pessimism`) are separate, additive knobs; the campaign varies only the latter |

## Battle-tested components (selectors)

| Piece | Canon | Finding |
|---|---|---|
| GAE (`gae_advantages`) | Schulman 2016: `A_t = Σ (γλ)^l δ_{t+l}` | exact; `returns = A + V` is the value target; λ→1 (MC−baseline) and λ→0 (TD residual) limits pinned; GAE's value target ≡ λ-return pinned (`test_returns_gae`) |
| Squashed Gaussian (`critics.py`, unwired) | SAC App. C: stable `log(1−tanh²u) = 2(log2 − u − softplus(−2u))` + scale Jacobian | **TWO DEFECTS FOUND + FIXED**: it used the ε-clamped `log(1−a²+1e−6)` (precision loss for \|u\|≳6) and OMITTED the `−d·log(action_scale)` Jacobian while claiming an exact log-prob. Now mirrors the WIRED policy (`policy.py`, which was already fully correct). New pins: the scale-Jacobian offset `d·log s` exactly; finiteness at saturated pre-activations |
| TwinQ | Fujimoto 2018 clipped double-Q | independent heads + `min_q` — correct; min ≤ both pinned |
| CEM (`cem.py`) | Chua 2018 / standard MPC | sample → score → top-K → refit, std floor, bound clamps, seeded generator (no global RNG); recovers a known optimum + determinism pinned |
| Custom encoder / net_builder | — | Lazy modules, latent contract enforced by a projection head (k-dim + LayerNorm); shape-rank typing refuses illegal chains at authoring; bitwise resume pinned |

## Search & analysis machinery

| Piece | Canon | Finding |
|---|---|---|
| Median stopping (`mbrl.search`) | median rule (Google Vizier) | compares at the LATEST COMMON step (a slow arm is never beaten by a fast arm's later points); strict losers only; conservative gates — all pinned |
| Random sampling | — | typed distributions, seeded (same seed → same arms; resume-safe); bounds pinned |
| PCA (`mbrl.diagnostics`) | SVD on centered data | sign convention fixed; known-axes recovery, ratio ordering, roundtrip exactness, monotone truncation error pinned |
| K-fold ridge CV | closed form | partition laws (disjoint/balanced/seeded), finds planted linear signal (R²>0.95), reports ≈0 on noise — the honest-baseline behavior |

## The heart (untouched, verified standing)

Hutchinson penalty: 2-probe unbiased estimator vs analytic Hessians on
quadratics (`test_hvp.py` unchanged this whole period); isotropic, fp32 under
autocast, no policy-Hessian option — all founding-doc hard rules intact.

## Conclusions

1. **The running campaign's math is correct** — no action needed on it.
2. Two real defects existed, both in the UNWIRED SAC policy copy (numerical
   form + missing scale Jacobian). Fixed + pinned before that selector ever
   trains anything. The wired policy was already exact.
3. Constant-offset caveat for the record: a missing `−d·log s` term shifts
   log-probs uniformly — zero effect on policy-parameter gradients, but it
   would have skewed entropy diagnostics and any future auto-temperature
   tuning. Worth having caught now.
