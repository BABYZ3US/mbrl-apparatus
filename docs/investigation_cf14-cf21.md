# Curvature-MBRL investigation cf14–cf21: loss closed form, paths, tests, results, spectral breakdown

**Apparatus:** mbrl-curvature (HalfCheetah-v5, 2×A100 RunPod). **Window:** 2026-06-13 → 2026-06-15.
**Question that started it:** reproduce and stabilize the band-alone curvature representation that
reached **+1344** eval (exceeding the unregularized baseline), then close the two gaps it left —
*late convergence* and *across-seed spread*.

---

## 1. The loss function — closed form (cf21)

Two objectives are optimized on the same replay: a **model/representation** objective and a
**behaviour/policy** objective (Dreamer-style λ-returns, *not* SAC). Notation: `z` latent, `a`
action, `r` reward, `G = (1/N)ZᵀZ` the latent Gram matrix with eigenvalues `{λ_i}`, `R̄` the eval
return EMA, `t` model-update step.

### 1.1 Model / representation objective

```
L_model = L_rew + L_dyn + Λ(t)·Ω_curv  +  w_band·B(G)  +  L_couple
```

- **L_rew** = Huber( symlog R̂(z,a), symlog r ) — reward regression. *The regularization target.*
- **L_dyn** = ‖T(z,a) − z′‖²  + operator-structure penalties  w_n‖N(T)‖ + w_s‖∂T‖ + w_ρ·ρ(T)
  (normal-defect, smoothness, spectral-radius; structure=normal, rank 0).
- **L_couple** = w_c·‖C(z_d, z_p)‖ — dual-twin coupling (`penalize_reward=true`, radius_p=0.02).

**Curvature penalty (isotropic, R16) — the unbiased 2-probe Hutchinson trace²:**
```
Ω_curv = (1/⌊n_p/2⌋) · Σ_pairs (vᵢᵀ H vᵢ)(vⱼᵀ H vⱼ),   H = ∇²_(z,a) R(z,a),   v ~ Rademacher
```
i.e. an unbiased estimator of `tr(H)² ≈ (ΔR)²` (R4/R5). Applied in latent coords on detached `(z,a)`;
**never** anisotropized, **never** on the policy Hessian (R10).

**Effective penalty weight (scheduled × gated × floored):**
```
Λ(t) = max( λ(t) · g_ret(R̄) · g_dis , λ_min )
  λ(t)     = λ₀ · ( t₀ / (t₀+t) )^(1/3)            cuberoot anneal (R12), λ₀=1e-3
  g_ret(R̄) = leaky_relu return gate ∈ [g_f, 1], RATCHETED   (§1.3)
  g_dis    = ensemble-disagreement gate            ( = 1 with one reward head )
  λ_min    = 1e-4                                   hard floor — never fully releases
```

**Spectral band (the only active frame term):**
```
B(G) = Σ_i relu(λ_i − c)²   +   Σ_i σ( β (f − λ_i) )           c=1, f=0.1, β=20
        └ ceiling wall ┘         └ sigmoid floor wall ┘
```
Penalizes *only* eigenvalues that escape `[f, c]`; the interior is free, so the **rank emerges**
rather than being demanded. (All other frame terms — `w_compress, w_logdet, w_ortho, w_rank2,
w_dissip, w_lyap, w_shell` — are **0** in cf21.)

### 1.2 Behaviour / policy objective (Dreamer λ-returns)

```
L_π = − E_τ[ A_t / s ]  −  c_ent(rf)·H[π]  +  w_align·A_z
  A_t      = λ-return advantage over imagined rollouts of horizon H_t (model T unrolled)
  s        = max(1, ret_scale)                     Dreamer-V3 percentile return-normalization
  H[π]     = − E[ log π(a|z) ]                      entropy estimate
  c_ent(rf)= c_ent⁰ · (1 − rf)                      reward-ANNEALED entropy bonus
  A_z      = ‖mean/std(imagined z) − mean/std(encoded z)‖²   imagination-latent alignment
  rf       = clip( (R̄ − mid)/scale , 0, 1 )        reward fraction (mid=0, scale=1000)
```
Actor grad-norm clipped at `γ_clip·( m + (1−m)(1−rf) )` (reward-adaptive clip, m=0.1).

**The policy distribution — and cf21's structural lever (the variance bound):**
```
π(a|z) = tanh 𝒩( μ(z), σ(z)² ),   log σ(z) ∈ [ ℓ(rf), 2 ]
   ℓ(rf) = ℓ_hi + (ℓ_lo − ℓ_hi)·rf            ℓ_hi = −1 (σ≥0.37 explore),  ℓ_lo = −4 (σ≥0.018 commit)
```
A **hard** lower clamp on `log σ`, driven by reward: σ has a guaranteed minimum, so the policy
**cannot collapse to a deterministic point mass**. Because `act()`/eval sample stochastically, this
keeps *collection* exploratory; the reward-relaxation recovers near-deterministic peak return.

### 1.3 Adaptive schedules (the "ratchet" family)

```
Return gate (leaky_relu):  release(frac) = leak + (1−leak)·2(frac−½)   for frac ≥ ½   (sharp release above mid)
                                         = leak·2·frac                  for frac < ½   (rigid hold below mid)
                           g_ret = g_f + (1−g_f)(1 − release);  RATCHET: once R̄>mid, lock running-min(g_ret)
Horizon (ratcheted):       H_t = max(H_{t−1}, Ĥ_t) once Ĥ ≥ 15, else Ĥ_t;  clamp [15, 25]
Value:                     critic regresses to the λ-returns (symlog), EMA target net
```

**The one-line story of cf21:** the loss is the validated band-alone curvature objective (`B(G)` +
`Λ(t)·Ω_curv`), wrapped in reward-coupled *ratchets* on every release valve (λ-gate, horizon,
entropy bonus, grad-clip) and floored by a **hard variance bound** `ℓ(rf)` on the policy.

---

## 2. The investigation (chronology)

| run  | lever under test | outcome |
|------|------------------|---------|
| cf14 | double-wall (0.1 logdet + 0.99 shell) → redirected to **band-alone** | band-alone **+1244** (peak); double-wall weak |
| cf15 | + nuclear-norm **compression** + quadratic **return-gate** | capped **+458** — compression+gate **HURT** |
| cf16/17 | band-alone reproduction, ceiling sweep | **cf17-ceil1.0-s0 +1344** (exceeds baseline); **ceil 1.0 ≫ 0.99** |
| cf18 | floor-wall **shape** sweep {relu1, sigmoid, gelu, leaky} | **sigmoid** converges fastest then stabilizes; relu1 close |
| cf19 | combined adaptive levers + **relu** entropy floor (6 seeds) | **10/12 entropy-collapsed**, 0 climbed @125k → obstruction found |
| cf20 | **sigmoid** entropy floor + **leaky_relu** gate (6 seeds) | mixed: s1 **+705** but whipsawing (705→−154), 4/6 flat |
| cf21 | reward-adaptive **hard log_std floor** (variance bound) | *in flight* — entropy bounded −6…+3 early (no −1e33 free-fall) |

The arc: **find** the win (band-alone, cf14–17) → **stress** it across seeds (cf18–19) → **diagnose**
the obstruction (cf19) → **mis-attribute then correct** it (cf20) → **fix it structurally** (cf21).

---

## 3. Paths explored (levers, and what each tested)

- **Spectral shaping.** Two-sided band `B(G)` (bound `[f,c]`, free interior) vs. rank-demand (shell),
  vs. compression (nuclear norm `Σ√(λ−f)`), vs. log-det volume barrier. **Band-alone won;**
  compression + gate actively hurt.
- **Floor-wall shape** (CAS-motivated, §5): relu², relu1, softplus, sigmoid, gelu, leaky_relu — the
  function `Φ(f−λ)` enforcing the floor. relu²'s lift *vanishes* at the floor; relu1/softplus/sigmoid
  *bind*.
- **Return-gate shape**: quadratic / cuberoot / sigmoid / bump / **leaky_relu** (threshold) — the curve
  relaxing `λ` as return climbs, plus a **ratchet** (lock once return crosses mid).
- **Horizon ratchet**: monotone non-decreasing imagination horizon once H≥15.
- **Seed-robustness stack**: near-zero policy init, reward-annealed entropy bonus, reward-adaptive
  grad-clip, **soft entropy floor** (relu/sigmoid) → **hard variance bound** (log_std floor).
- **Theory probes**: CAS (sympy) for the band's spectral fixed point; time-evolving PCA / Gram /
  covariance of the latent; a power-law / self-organized-criticality reframe of effective rank.

---

## 4. Tests run

All `src/` changes gated on `pytest` (CPU, no MuJoCo/W&B). Last green run before cf21 launch: **66
passed**. Relevant suites:

- **`test_rank2_frame.py`** — band floor-wall shapes: each `Φ` lifts/binds correctly (relu², relu1,
  softplus, sigmoid, gelu, leaky); ceiling wall; `relu2` byte-identical to the original two-sided band.
- **`test_return_gate.py`** — gate shapes & anchors: sigmoid/quadratic/cuberoot/bump, the
  **leaky_relu threshold** (full λ below mid → ~0.91 held at mid → floor above), slew-limit, and the
  **λ-gate ratchet** (no re-tighten).
- **`test_new_features.py`** — horizon ratchet (locks running-max, checkpointed); reward-adapt
  (`_reward_frac`, `_policy_reg` three knobs track reward, off-is-identity); **entropy-floor shapes**
  (sigmoid lift peaks at target & bounded; relu constant lift); **cf21 log_std floor** (rf→floor map,
  hard clamp of policy output, legacy-off identity); near-zero policy init.
- **`test_checkpoint.py::test_resume_bitwise_identical`** — the resume guarantee. cf21's variance
  bound recomputes `ℓ(rf)` deterministically from the checkpointed `ret_ema`, so resume stays exact.
- **`test_hvp.py`** — Hutchinson estimator vs. analytic Hessians (unchanged; the R4/R5 contract).
- **`test_smoke.py`** — full loop on Pendulum < 1 min (the pre-GPU gate).

---

## 5. Results

### 5.1 Validated findings (hold across runs)
1. **Band-alone is the win.** Bounding every Gram eigenvalue in `[0.1, 1]` with a free interior
   reaches **+1344** (> baseline). Adding compression or a quadratic gate **caps** it (+458).
2. **Ceiling 1.0 ≫ 0.99.** The hard energy ceiling at 1.0 beats the 0.99 "leave 1% in the tail" form.
3. **`cond(G)` is DECOUPLED from eval** (§6) — the spectral condition number does *not* gate the climb.
4. **The obstruction is a policy-side collapse, not a spectral one.** A *tanh-mean-saturation* entropy
   collapse: σ→0 and μ saturates tanh → log-prob blows up → `policy/entropy` → −1e9…−1e33;
   `skip_nonfinite` limps on with a dead, deterministic policy.
5. **Soft entropy floors cannot fix it.** Penalizing the entropy *estimate* (a downstream quantity)
   loses the gradient fight — relu (cf19: 10/12 collapsed) and sigmoid (cf20: still whipsawing).
6. **The blowups are partly a red herring.** They are transient and `skip_nonfinite`-absorbed; cf20
   s0 reached **+252 *with* ent_min −2.5e33**. The real, unsolved problem is the **seed spread**
   (1–2 of 6 climb).
7. **The fix is structural** (cf21): a *hard* lower clamp on `log σ` bounds entropy below by
   construction — no gradient battle — and keeps collection exploratory. Early signal confirms the
   mechanism: entropy bounded at −6…+3 instead of the −1e33 free-fall.

### 5.2 A correction we made mid-investigation
At 120k, cf20 was called "6/6 dead" off an `ent_min < −10` detector and a kill was recommended. By
250k it was 1/6 *sustained* climb (+319) and 1/6 climbed-then-faded — the detector conflates a
**transient** blowup with death. **Lesson (recurring): do not call death before the late climb window
(~280k+); judge by the eval trajectory, not `ent_min`.** Same error class as the cf14 premature kill
(+506 → +1244).

---

## 6. PCA / spectral breakdown

**Method.** Time-evolving PCA of the latent: log the full Gram spectrum `{λ_i}` per eval, track the
covariance/Gram condition number `cond(G)=λ_max/λ_min`, effective rank, and their correlation with eval.

**Key results.**
- **`cond(G)` ⟂ eval return** (corr ≈ 0.02). The **+1344** winner ran at *high* cond; a dead arm had
  the *same* effective rank. ⇒ the spectral state is not the bottleneck; cf18's cond-cleanup is theory
  hygiene, not a peak-mover. This is what pivoted the work to the *seed-spread / policy* axis.
- **Equilibrium effective rank ≈ 12.8 / 16 ≈ 0.80.** Falls straight out of the band geometry: with
  `f/c = 0.1` and ~12 modes pinned at the ceiling, the participation ratio
  `(Σλ)²/Σλ² = 12.4²/12.04 ≈ 12.8`. The "0.80 ≈ Pareto" coincidence is a *box* artifact, not a
  scale-free law (see the power-law note below).

**CAS closed forms** (`scripts/cas_spectral_optimum.py`, sympy) for the band's spectral fixed point
under a downward drift `g` per mode, band weight `w_b`:
```
active eigenvalue   λ*  = c + g/(2 w_b)
condition number    cond = (c·w_b + g/2)/(f·w_b)  → c/f  (bounded, IF the floor wall binds)
rank threshold      g*  = w_c/(2√ε)   (compression strength needed to kill a mode)
```
The crux the CAS exposed: a **relu² floor lift vanishes at the floor** (`2·relu(d) → 0`), so when the
drift exceeds `2·w_b·f` the deadest mode sinks and **cond is unbounded** — which is why we swept floor
shapes whose lift *doesn't* vanish (relu1, softplus, sigmoid). The live runs sat at cond `1e7–1e12`
under relu² — exactly the predicted unbounded regime — yet still hit +1344, *because* cond ⟂ eval.

**Power-law / criticality reframe.** Modeling the spectrum as `λ_i ∝ i^(−α)` gives a phase transition
in `eff_rank/d` (verified numerically over `d` up to 65k):
- `α < ½` **extensive** (`eff_rank/d → (1−2α)/(1−α)²`),
- `α = 1` **critical** — the 1/f, scale-free line (`eff_rank ~ (ln d)²/ζ(2)`),
- `α > 1` **collapsed** (`eff_rank → ζ(α)²/ζ(2α)`, a `d`-independent constant).

The band confines `λ` to a **box** — a state with a characteristic scale `c/f`, the opposite of
scale-free — so a true "confined critical state" would need a scale-invariant (log-spectrum) penalty
targeting `α=1`, not a box. (Conjectural; not yet run. The digamma `ψ=Γ′/Γ` appears as the ratio
between the entropy- and participation-effective-rank for a Gamma-`k` spectrum.)

---

## 7. Where it stands / open questions

- **cf21 in flight**: does a guaranteed exploration floor finally **tighten the 1–2-of-6 spread** and
  bring climbs in **stable** rather than whipsawing? Mechanism confirmed (entropy bounded); payoff TBD
  over ~280–350k.
- **Tuning axis** if the spread tightens but caps the peak: the `(ℓ_hi, ℓ_lo)` floor and its
  reward-relaxation rate.
- **Theory thread** (lower priority): swap the band box for a scale-invariant log-spectrum penalty and
  test whether the emergent `eff_rank/d` locks onto a power-law exponent (criticality) instead of
  drifting with `d`.

*Memory: see `mbrl-entropy-collapse-obstruction` for the obstruction + fix; `superconductor-collapse-framing`
for the type-II/depinning map; `dont-kill-running-experiments-without-permission` for the premature-kill rule.*
