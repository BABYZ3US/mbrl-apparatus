# Candidate Algorithms for the Operator-Field World Model

*A design memo: robust, relevant algorithms to test in combination with the dual operator
latent, mapped onto the program's own derivations and the external literature. 2026-06-21.*

**The vision (restated).** A task-agnostic core world model = a **learned spectral encoder** →
a **transformer** producing latent **objects** → which the **dynamics operator `A_d`** and
**policy operator `A_p`** project, trained by a **closed-form operator loss**, exploiting the
**Koopman↔Bellman** correspondence, on a **phased optimizer duty cycle**. The governing
objective throughout is **bias–variance control** — capacity (variance) vs. value-relevance
(bias) — which is the lens every choice below is judged by.

**Grounding.** This maps almost one-to-one onto the two derivation docs (`derivations_spectral_operator_LQG.md`, `Appendix_C_RL_derivations.md`): the Stein/operator
cross-entropy (§6), the Lyapunov↔Riccati / Koopman↔Bellman duality (§7, C.6), the
dissipative/conservative determinant asymmetry (§4–5), and the critical ratio `r⋆=1/5` (§9).
Where the program already has a result I cite it; where the literature has a robust method I
name it; honest caveats (especially 2-adic) are collected at the end.

---

## 1. The spectral encoder (producing the objects)

**What the theory already fixes.** The band-pinned Gram (§1, C.2–C.3) gives a
dimension-invariant `z_std=√⟨μ⟩` and `r_eff/d≈const` — i.e. the encoder's *capacity* is a
tunable constant decoupled from return (`corr(cond G, return)≈−0.02`). That is variance
control made explicit: the band sets how many effective modes the objects carry.

**Robust algorithms to test (all combine with the dual operator latent):**
- **Deep-Koopman encoder** (Lusch et al. 2018; Takeishi et al. 2017): train the encoder so the
  *operator* is linear in the learned coordinates — exactly the `A_d`/`A_p` substrate. This is
  the most direct upgrade from an MLP encoder to a *spectral* one.
- **Laplacian / spectral representation** (Wu, Tucker, Nachum 2019) and **SPEDER** (Ren et al.
  2023): learn features in which the transition operator factorizes and *reward is linear* —
  which is what makes the closed-form reward fit (§6) and the resolvent value (§4 below)
  well-posed. SPEDER is the strongest "spectral features for RL" baseline to port.
- **Self-predictive consistency** (SPR, Schwarzer et al. 2021; EfficientZero's consistency
  loss): a cheap regularizer that keeps the objects temporally coherent without reconstruction
  — pairs well with the band as a second variance control.

**What to measure (bias–variance):** `latent/gram_cond`, `gram_eff_rank`, `latent/eig*`
(now plottable via the new viz keys) vs. return — confirm capacity moves independently of
performance, and that the spectral encoder holds `r_eff/d` while improving the operator fit.

---

## 2. The closed-form operator loss — and the honest 2-adic scope

**This is the load-bearing one, and your own §6 already nails it.** The operator
cross-entropy `𝓛(A,Q)=log det Σ̂ + tr(Σ̂⁻¹G)`, `Σ̂=AGAᵀ+Q`, **is** Stein's loss, and (Prop 5)
it *separates per-mode* into `φ(ν)=ν−log ν−1` over the generalized spectrum `ν=eig(Σ̂⁻¹G)`.
Three properties make it the right closed-form loss to test in place of MSE + clipping:

1. **It is self-bounding (this is what reduces the need for clipping).** `φ(ν)=ν−logν−1` has
   `φ'(ν)=1−1/ν`, and `tr(Σ̂⁻¹G)→∞` as `Σ̂` loses rank — an *intrinsic* invertibility barrier
   (Prop 4 table). So divergence is penalized by the loss geometry, not by an external clamp.
   The robust-literature framing: this is the **log-det (Burg/Stein) matrix divergence**
   (James–Stein 1961; Amari 2016), and the per-mode `φ` is exactly the **Itakura–Saito**
   scalar — a mature, convex-on-`ν>0` objective.
2. **On the conservative manifold it is rational.** When `det(Σ̂⁻¹G)=1` (the `det(A_p)=1`
   structure of §5) the log terms cancel and `D∈ℚ` outright (your `{5/4,4/5,1}→1/20`).
3. **The 2-adic part — stated honestly (matches both your Remark and the external check).**
   Fixed-width binary is 2-adic; on the conservative manifold the loss is rational and its
   *value* is computable exactly by Hensel/Dixon lifting + rational reconstruction. **But**
   `ℚ₂` is **unordered**, so this certifies the loss **value** (a verification tool), **not the
   minimization** (gradient descent needs the real order); and off `det=1` the `log` is
   genuinely transcendental. The external survey agrees bluntly: exact-rational solving is
   mature (LinBox/FLINT) but "exact ≠ bounded," and *no literature bridges p-adic solving to
   RL update stability*. **Recommendation:** adopt the **per-mode `φ(ν)` Stein loss as the
   trainable objective** (that is what removes clipping, via property 1) and **use 2-adic exact
   evaluation as a verification gate on conservative-manifold runs** (where `det(op_p)=1`), not
   as the optimizer. Treat "2-adic replaces gradient clipping" as the one idea to *not* bank on.

**What to test:** swap the reward/dynamics MSE for the per-mode `φ(ν)` loss (triangularize via
Cholesky `Σ̂=LLᵀ` then `L⁻¹GL⁻ᵀ`; the per-mode gradient `1−1/ν` never needs the matrix
inverse), with `reward_clip=return_clip=0` — and measure whether the new `stab/*_clip_frac`
telemetry stays at zero (i.e. the loss self-bounds) vs. the MSE baseline.

---

## 3. Transformer memory → the objects the operators project

**The novel architectural piece.** Today the encoder is an MLP; the vision is a **transformer**
producing the latent objects `z_t` that `A_d`/`A_p` then evolve. The robust precedents for the
*memory/credit-assignment* half are the transformer world models — **IRIS** (Micheli 2023),
**TWM** (Robine 2023), **STORM** (Zhang 2023), and **TWISTER** (2025, contrastive long-horizon)
— and on the SSM side **R2I** for very long credit assignment.

**The combination to test (and its risk):** transformer as the *sequence encoder/memory* whose
per-step output vector is the object `z_t`; the dual operators stay the linear evolution on
`{z_t}`. This cleanly separates "what to remember" (attention) from "how state evolves"
(operator) — and the operator's linearity is exactly what keeps long imagined rollouts
differentiable and analyzable (eigenvalues, `det`, Lyapunov). **Bias–variance caveat:** a
transformer is high-capacity (variance); the band + `φ`-loss + `r_eff` cap are the
counter-pressure. Test transformer-encoder vs MLP-encoder *at matched `r_eff`*, watching
`gram_cond` and the new `op/sv*` spectrum for instability the extra capacity might inject.
STORM's result (stochastic latents + attention) suggests keeping the *stochastic* `Q` channel
(§2 Stein) rather than a deterministic transformer latent.

---

## 4. Koopman↔Bellman: the value as a resolvent (the biggest lever)

**The correspondence you want is exact, and it's the highest-leverage idea here.** Your §7
already has the LQG dual: `A_d↔Σ` (Lyapunov/estimation), `A_p↔P` (Riccati/control), with
`det(A_p)>0 ⟺ P≻0`. C.6 sharpens it: a Bellman operator commuting with the Koopman operator
is a **spectral function `f(K)`**. The external literature closes the loop with the **exact
identity**: the value function is the **resolvent** of the (Koopman/transition) operator,
`V = (I−γK)⁻¹ r`, diagonalized by the operator's eigenfunctions — each eigenvalue `λ` becomes a
pole `1/(1−γλ)`. This is the **successor-representation / resolvent identity** (Dayan 1993;
Barreto et al. 2017, successor features) and, fused explicitly, **Koopman-Assisted RL**
(Rozwood et al. 2024, treats `V` as a Koopman observable).

**What to test:** compute the critic **as a spectral function of the learned `A_d`** rather than
(or alongside) a free value net — i.e. once the encoder makes reward linear (§1 SPEDER) and
`A_d` is the Koopman model, `V = (I−γA_d)⁻¹ r` is a *closed-form linear solve on the spectrum
you already have*. This is the maximal exploitation of the correspondence and it directly
attacks variance: the value stops being an independently-fit, high-variance network and becomes
a deterministic readout of the dynamics operator. Keep the learned λ-return critic as the EMA
*target/teacher* (your §8: the policy is retained as the **data-quality engine**, not the
controller) and test the resolvent value as the *bias-reduced* student. Robust fallbacks if the
full resolvent is unstable: successor features (linear `V = ψ·w`), or the projected/Krylov form
of C.6 (value in the cyclic subspace of `K`).

## 5. Entropy and Lyapunov: the ratios you asked about

- **Entropy coefficient — the principled target exists.** The literature default is
  **target-entropy `H̄ = −dim(A)`** with a *learned* temperature `α` (SAC "Algorithms and
  Applications", Haarnoja et al. **2018b/arXiv:1812.05905** — note: NOT the original ICML SAC,
  which fixed `α`). The repo currently uses a fixed `entropy_coef = 3.0e-4` with `auto_alpha`
  present but off (`base.yaml:446-449`). **Test:** turn on auto-α with `target_entropy=−dim(A)`
  — dimension-aware and self-tuning, replacing the hand-set 3e-4.
- **Lyapunov energy — it's a constraint, not a fixed scalar.** Per-mode stationary energy is
  `ν_i = q̂_i/(1−|λ_i|²)` (§2 Prop 3); the Lyapunov/Stein term `‖G−A_d G A_dᵀ−Q̂‖²` enforces
  it. There is **no principled constant** entropy:Lyapunov ratio in either the repo or the
  literature (the web check confirms: stability is an energy-decrease *constraint* with a tuned
  multiplier). The repo's only causal evidence is **A7 (`lyap_weight=0.3`)** = the single clean
  positive in Ablation-1 (2/2 seeds, +486±172), now in the 8×8 prereg.
- **The real "ratio" is the determinant ratio, and it has a critical value.** The energy/
  entropy-exponent ratio is `log det(A_d)/log det(A_p)` (§5), and §9 conjectures a **critical
  energy-retention ratio `r⋆ = |λ(A_d)|² = 1/5`** at which the policy determinant
  (`∝ e^{policy entropy}`), imagined return, and realized return break upward **together**
  (one-seed support: entropy −2.5→−0.3, eval −187→+540 at ~220k steps). **Test the reliable
  form:** hold `r→r⋆` via a *persistent descending/clamped spectral target* on `A_d` (a one-shot
  init reverts as the fit pulls `|λ|→1`). This is the experiment most likely to convert the
  `r⋆=1/5` conjecture into a claim — and it ties entropy (via `det A_p`) to the Lyapunov
  spectrum (via `r`) through the criticality, which is the closest thing to the entropy↔Lyapunov
  relationship you're after.

## 6. Optimizer duty cycles (two-timescale / phased)

**Your instinct is theoretically backed.** Updating every step fights the long-horizon operator
structure; alternating, episode-boundary updates have firm grounding:
- **Two-timescale stochastic approximation** (Borkar 1997; Konda–Tsitsiklis 2003; finite-time:
  Wu et al. 2020, Xu et al. 2020): a fast loop and a slow loop provably co-converge — the exact
  template for "fit the dynamics operator on one cadence, improve the policy operator on
  another." **This is the principled form of your trade-off-update-cycles idea.**
- **Two-block alternating minimization** (Beck–Tetruashvili 2013): convergence for "optimize
  `A_d`, then `A_p`, alternating" — your policy↔dynamics duty cycle.
- **Update-to-data / replay ratio** (REDQ, DroQ, primacy-bias resets, SR-SPR): the cadence is a
  real, tunable lever *if paired with a stabilizer* — which you have (the band, `φ`-loss).
- **Episode-boundary updates** (your stated win: "optimize at end of each episode, coherent
  trajectory"): consistent with the Dreamer/MBPO model-then-behave loop and with two-timescale.

**The repo mechanism already exists but is barely tested:** `struct_every` (phased SVD: amortize
the `O(d³)` operator priors to once/episode, keep the matmul-only Stein/`det` levers
every-update; `base.yaml:45-48`) and the fixed **4:1 model:behaviour ratio** (`200:50`,
`base.yaml:477-478`). **Honest caveat:** the *only* phased result on record (A16/A17) was
**negative** — but that ran at the failed env-dim² latent (`d=289`, where the `O(d³)` SVD itself
was the problem), so phasing was never cleanly tested at the normal latent. **Test:** a
two-timescale duty cycle at the *normal* latent — shared encoder updated every episode; `A_d`
and `A_p` on alternating cycles — measured on `latent/gram_cond` stability and the
**divergence between the policy and dynamics determinants** (`op/det_p_mean` vs the `A_d`
log-det), which is exactly the quantity you want to tamp down. (Naming note: the repo has a
policy-vs-**dynamics** determinant ratio, not a policy-vs-**reward** one — there is no
reward-operator determinant today; if you mean a reward-operator `det`, that's a new object to
define.)

---

## The bias–variance frame (the unifying judge)

| lever | controls | knob |
|---|---|---|
| spectral band / `r_eff` cap (§1) | **variance** (encoder capacity) | `c,f`, `latent_cap_mult` |
| value-equivalence / resolvent value (§4) | **bias** (spend capacity on what Bellman needs) | resolvent vs free critic |
| `φ(ν)` Stein loss + `det` barrier (§2) | **variance** (self-bounding, no clip) | the loss itself |
| two-timescale duty cycle (§6) | **variance** (stable targets) | update ratio, episode cadence |
| entropy target `−dim(A)` (§5) | **bias↔variance** (exploration) | auto-α |
| `r⋆=1/5` spectral target (§5) | the **phase transition** | descending `|λ(A_d)|²` clamp |

Read top-to-bottom: capacity is set by the band, *aimed* by value-equivalence, *stabilized* by
the self-bounding loss and the duty cycle, *explored* by the entropy target, and *triggered*
into the high-return regime by holding `r⋆`.

## Honest caveats (so you don't over-invest)
1. **2-adic ≠ anti-clipping.** It certifies the loss *value* on the conservative manifold
   (verification), not the minimization. The self-bounding `φ(ν)` loss is what actually removes
   clipping. (Both your Remark and the external survey say this.)
2. **Phased schedule has a negative on record** (A16/A17) — but confounded by `d=289`; retest at
   normal latent before trusting or discarding.
3. **Neuron spiking is essentially absent** from the codebase. The only spiking-*like* mechanism
   is the `|λ|≈1` marginal-oscillator **excitation** (`loop.py:382-386`, off by default) and the
   `r⋆=1/5` co-transition (§9). If "spiking" means edge-triggered operator excitation, that's
   where it lives; there is no integrate-and-fire/membrane model.
4. **`det` naming:** policy-vs-**dynamics** determinant ratio exists; policy-vs-**reward** does
   not (no reward-operator `det`).
5. **Entropy:Lyapunov has no magic constant** — `target_entropy=−dim(A)` is the defensible
   entropy target; the Lyapunov term is a tuned constraint; their coupling is the `r⋆` criticality.
6. **Refuted relations — do not add to the experiment (the derivations doc asks to forestall
   rediscovery).** `g_p = 1 + 1/g_d`, `g_p = 1 + eff/H`, and the golden-ratio operator-coupling
   fixed point are **rejected** as coincidences of two independently band-pinned constants
   (`derivations_spectral_operator_LQG.md` "Rejected along the way" + `verify_rejected_coincidences.py`).
   Note the cross-doc tension: `Appendix_C_RL_derivations.md` C.5b still lists `g_p = 1 + 1/g_d`
   as a "[PROVED structure] to be tested" — the main derivations doc (newer) **supersedes** the
   seed-appendix here and marks it refuted. The *genuine* conserved invariant from C.5 is the
   **equal effective rank** of `op_p`/`op_d` (12.23 vs 12.21, ratio 0.998), not the golden ratio.

## A suggested first combined experiment (you design the real one)
The cleanest single test that exercises the most of the above on the dual operator latent, at
the normal latent (no `d=289` confound), measured on bias–variance signals now visible in the
Studio: **{spectral encoder + `φ(ν)` Stein loss (clips off) + two-timescale episode-boundary
duty cycle}**, with `r⋆=1/5` held by a descending `|λ(A_d)|²` clamp, and the **resolvent value**
as an ablation arm against the free λ-return critic. Primary readouts: `eval/return_det`
(reliability/fire-rate), `latent/gram_cond` + `op/sv*` (variance), `op/det_p_mean` vs `A_d`
log-det divergence (the determinant trade-off), and `stab/*_clip_frac` (does the loss
self-bound?). That isolates the architecture while letting the Koopman↔Bellman value be the one
high-variance-reduction swap under test.

