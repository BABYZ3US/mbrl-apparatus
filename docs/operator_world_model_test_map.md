# Testing Map — the Operator-Field World Model

*A dependency-ordered, pre-registered test plan that turns the vision into experiments on the
dual operator latent. Companion to `operator_world_model_candidate_algorithms.md` (the *what*);
this is the *in what order, with what arms, judged how*. 2026-06-21.*

## The vision, as testable levers
A task-agnostic core: **spectral encoder → transformer objects → dynamics/policy operators
(`A_d`,`A_p`) → closed-form operator loss → Koopman↔Bellman value**, on a **phased duty cycle**,
judged by **bias–variance control**. Each arrow is one lever below; each lever is one phase.

## Global protocol (applies to every phase)
- **Substrate:** the dual operator latent, at the **normal latent dim** — *not* the `d=289`
  env-dim² latent (that direction is closed; it is the confound that made the only prior phased
  result, A16/A17, falsely negative).
- **Metrics are relative / reliability-based, not absolute.** The mlp-recipe **anchor
  regression is unresolved**, so absolute HalfCheetah return is void. Use the abl1 reliability
  metric: `fired ≡ final eval/return_det > threshold`, `k/n` seeds fired, and within-arm deltas.
- **Pre-register before launch** (abl1 discipline): fix arms, primary metric, and SHIP/DROP/
  INCONCLUSIVE rule *before the first seed*. Reliability claims need **n ≥ 8 seeds** (8v8 ≈ the
  power floor; require a ≥3-seed margin).
- **Every phase is observable live** via the shipped Studio telemetry — the bias–variance dials:
  `latent/gram_cond`, `gram_eff_rank`, `latent/eig*` (variance/capacity); `op/sv*` (operator
  spectrum, via the new `pull.operator_spectrum` verb); `op/det_p_mean` vs the `A_d` log-det
  (the determinant trade-off); `spectral/fit_r2|fit_mse` (closed-form reward-fit quality);
  `stab/*_clip_frac`, `stab/value_grad_norm`, `stab/nonfinite_skips` (stabilizer activity).
- **Turn on early, keep fixed:** `auto_alpha` with `target_entropy = −dim(A)` (the defensible
  entropy setting); the band (variance cap). **Monitor, don't tune:** the equal-`eff_rank`
  invariant of `op_p`/`op_d` (C.5; ratio ≈0.998) — a health gauge, not a lever.

---

## Phase 0 — Unblock + instrument (gates everything; cheap)
- **0a. Resolve the anchor regression** (ledger improvement-plan #1) *or* formally commit to
  relative/reliability metrics for the whole map. Diff the recipe arm's effective config vs the
  original report (smoothing.sigma 1.5 vs 1.0, eval protocol, Hutchinson probe count) before any
  spectral relaunch. **Until done, no absolute-return claim is valid.**
- **0b. Confirm the dials.** One throwaway dual-latent run end-to-end through the Studio (RunPod):
  verify `op/sv*`, `spectral/fit_*`, `op/det_p_mean`, `gram_cond`, `stab/*` all stream to the
  panels. This is the measurement substrate every later decision rests on. (Verb + emitters
  already shipped; this just confirms the loop on GPU.)

## Phase 1 — The closed-form operator loss (anti-clipping). *Cheapest, highest-leverage.*
- **Hypothesis.** The per-mode Stein loss `φ(ν)=ν−log ν−1` (operator cross-entropy, §6) is
  **self-bounding** (`tr(Σ̂⁻¹G)→∞` as `Σ̂` loses rank), so reward/return/value clipping can be
  turned **off** without divergence — at equal or better reliability than MSE+clips.
- **Arms (dual latent, normal dim).** `A0` baseline: current MSE losses + cf4 clips ON
  (`imagination.reward_clip/return_clip`, `optim.value_clip` > 0). `A1`: `φ(ν)` Stein loss on the
  reward/dynamics heads (Cholesky `Σ̂=LLᵀ` → `L⁻¹GL⁻ᵀ`; gradient `1−1/ν`, no matrix inverse) +
  **all clips OFF**.
- **Primary metric + rule.** Divergence/NaN rate (`stab/nonfinite_skips`) and `k/n` fired. SHIP
  `A1` if it fires ≥ `A0` with **zero** clip-driven interventions and stable `gram_cond`.
- **Bias–variance.** Pure variance test: does the loss geometry replace the external clamps?
- **Verification overlay (not a perf arm):** on conservative-manifold runs (`det(op_p)=1`,
  `detpos` engaged) run the **2-adic exact-value check** to certify the loss value bit-exactly —
  this is the *only* role of the 2-adic machinery (ℚ₂ is unordered → it cannot do the
  minimization). Do **not** expect 2-adic to replace clipping; `φ(ν)` does that.

## Phase 2 — The optimizer duty cycle (two-timescale). *Cheap; tests "trains faster".*
- **Hypothesis.** Optimizing at **episode boundaries** (coherent trajectory) with the **shared
  encoder updated every episode** and `A_d`/`A_p` on **alternating** update cycles trains faster
  and stabilizes `gram_cond` + shrinks the policy↔dynamics determinant divergence — vs.
  every-step updates. (Grounded: two-timescale SA [Borkar], alternating-min [Beck–Tetruashvili].)
- **Arms.** `B0`: every-step (`struct_every=1`, `model:behaviour = 200:50` interleaved). `B1`:
  two-timescale phased — `struct_every ≈ model_updates_per_iter` (operator SVD priors once/
  episode, the matmul-only Stein/`det` levers every update), encoder every episode, `A_d`/`A_p`
  alternating duty cycles, optimizer stepped at episode end.
- **Primary metric + rule.** **Wall-clock to a fixed reliability threshold** (speed) +
  `gram_cond` variance + `|log det A_p − log det A_d|` divergence. SHIP `B1` if faster AND not
  worse on stability. (This is the clean re-test of the A16/A17 idea at the normal latent.)
- **Bias–variance.** Variance reduction via stable targets / coherent-trajectory updates.

## Phase 3 — The spectral encoder (the objects). *Architectural; unlocks Phase 5.*
- **Hypothesis.** A **deep-Koopman / SPEDER** spectral encoder (features in which the operator is
  linear and **reward is linear**) holds `r_eff/d` while improving the operator/reward fit, vs.
  the current MLP encoder — at matched capacity (band).
- **Arms.** `C0`: MLP encoder. `C1`: spectral encoder (deep-Koopman objective, or SPEDER
  spectral features) feeding `A_d`/`A_p`; band/`r_eff` matched to `C0`.
- **Primary metric + rule.** `spectral/fit_r2` (reward-fit quality) and `op/sv*` conditioning at
  equal `r_eff`; reliability not worse. SHIP `C1` if it raises `fit_r2` / conditions `op/sv*`
  without costing reliability. **Linear reward here is the prerequisite for Phase 5.**
- **Bias–variance.** Aims capacity at value-relevant structure (bias) at fixed variance.

---

## Phase 4 — Transformer memory → the objects. *Higher capacity, higher risk; gated on 2+3.*
- **Hypothesis.** A **transformer** sequence encoder whose per-step output vector is the latent
  object `z_t` (operators evolve `{z_t}` linearly) improves long-horizon credit assignment vs.
  MLP/RSSM — **without** blowing `gram_cond`, because the band + `φ`-loss + `r_eff` cap hold the
  extra capacity (variance) in check. (Lineage: IRIS/TWM/STORM/TWISTER; keep the stochastic `Q`
  channel — STORM's lesson — rather than a deterministic latent.)
- **Arms.** `D0`: current encoder. `D1`: transformer encoder → objects, operators on top;
  matched `r_eff`. (Optionally `D2`: SSM/Transformer-XL memory for very long horizons.)
- **Primary metric + rule.** Long-horizon reliability + `gram_cond`/`op/sv*` stability (does the
  capacity destabilize the operator spectrum?). SHIP `D1` only if it improves credit assignment
  **and** holds operator-spectrum conditioning. This is the riskiest variance bet — gate it
  behind a stable loss (P1) and duty cycle (P2).
- **Bias–variance.** Buys representational capacity (variance); the prior phases are the
  counter-pressure that makes it safe.

## Phase 5 — Koopman↔Bellman: the resolvent value. *The headline lever; gated on Phase 3.*
- **Hypothesis.** With reward linear in the spectral features (P3), the value is the **resolvent
  of the dynamics operator**, `V = (I−γA_d)⁻¹ r` (§7 LQG duality; C.6 Bellman `= f(K)`) — a
  closed-form readout of the operator spectrum you already learn. Replacing/augmenting the free
  critic with it **reduces value variance** and improves reliability.
- **Arms.** `E0`: free Dreamer λ-return critic (current). `E1`: **resolvent value** computed from
  the `A_d` spectrum + linear reward, with `E0` retained as the **EMA target/teacher** (§8: the
  policy stays the *data-quality engine*, not the controller). Fallbacks if the full resolvent is
  unstable: **successor features** (`V = ψ·w`, linear) or the **Krylov/cyclic-subspace** value of
  C.6.
- **Primary metric + rule.** Value-estimate variance + sample-efficiency-to-threshold +
  reliability. SHIP `E1` if it lowers value variance at equal/better reliability. Watch
  `op/det_p_mean` (P≻0 / `det(A_p)>0` is the resolvent's well-posedness condition, §4).
- **Bias–variance.** The cleanest **bias reduction**: value becomes a deterministic operator
  readout instead of an independently-fit, high-variance net.

## Phase 6 — The criticality: hold r⋆ = 1/5. *Overlay; converts Conjecture 1 → claim.*
- **Hypothesis (your §9).** Holding the energy-retention ratio `r = |λ(A_d)|² → r⋆ = 1/5`
  (i.e. `|λ(A_d)| ≈ 0.447`) via a **persistent descending/clamped spectral target** triggers the
  sharp co-transition (policy determinant `∝ e^{entropy}`, imagined return, realized return break
  upward together) **reliably across seeds** — where a one-shot init at `r⋆` reverts (the fit
  pulls `|λ|→1`).
- **Arms.** `F0`: no spectral target. `F1`: descending `|λ(A_d)|²` clamp to `r⋆=1/5` (a moving
  `radius_max` target, not a one-shot init). **n ≥ 8 seeds** (this is the reliability claim that
  most needs power). Pairs with the A7 `lyap_weight=0.3` Stein arm (the lone clean abl1 positive).
- **Primary metric + rule.** `k/n` fired + the co-transition detector (`verify_critical_
  transition.py`): does `F1` raise the fire-rate and produce the vertical co-break? SHIP if
  `kF1 ≥ 6/8` and `≥ 3` over `F0`. This is the experiment that turns the `r⋆=1/5` conjecture into
  a result (or cleanly falsifies it — equally publishable).
- **Bias–variance.** Not a bias/variance knob per se — the *order parameter*: it puts the system
  on the edge where the operator structure pays off.

---

## Dependency DAG (read top-down; → = "unlocks / should precede")
```
Phase 0  Unblock anchor + confirm dials   ──────────────┐ (gates ALL)
                                                         ▼
Phase 1  φ(ν) Stein loss (clips off) ──┐     Phase 2  Two-timescale duty cycle
   (cheap, independent)                │        (cheap, independent)
                                       ▼                  │
Phase 3  Spectral encoder (linear reward) ◄───────────────┘
                       │
            ┌──────────┴───────────┐
            ▼                      ▼
Phase 4  Transformer objects   Phase 5  Resolvent value  ★ headline
   (gated on 2 + 3)               (gated on 3; the big variance win)
            │                      │
            └──────────┬───────────┘
                       ▼
Phase 6  Hold r⋆=1/5 (overlay; best once 1+2 stable; n≥8 for the claim)
```
**Suggested execution order:** 0 → {1, 2 in parallel} → 3 → 5 (★) → 6, with 4 as an opt-in
capacity bet once 1–3 are green. Phases 1, 2, 6 are cheap/relative and anchor-immune — safe to
run before 0a finishes; 3/4/5 want the dials confirmed (0b) first.

## Excluded / refuted (do not spend cycles here)
- **2-adic as an optimizer / clipping-replacement** — refuted in scope: certifies the loss
  *value* on the conservative manifold only (ℚ₂ unordered). It's a Phase-1 *verification overlay*,
  nothing more.
- **`g_p = 1 + 1/g_d`, `g_p = 1 + eff/H`, golden-ratio operator coupling** — refuted as
  band-pinned coincidences (`derivations` "Rejected"; supersedes Appendix C.5b). The real C.5
  invariant to monitor is **equal `eff_rank`** of `op_p`/`op_d`.
- **Radius-pinning of `|λ|` as an `|λ|`-actuator (A9–A13/A18/A19)** and the **env-dim² latent
  (A14–A17)** — both closed by abl1 (radius penalty can't move `|λ|` off 1.0; `O(d³)` SVD
  infeasible at `d=289`). The duty-cycle (P2) is re-tested at the *normal* latent precisely to
  escape the A16/A17 confound.

## What "done" looks like
A task-agnostic core where: the loss self-bounds (P1), the duty cycle is stable+fast (P2), the
spectral encoder makes reward linear (P3), the value is a resolvent readout (P5), optionally
transformer-objects carry long memory (P4), and `r⋆=1/5` reliably triggers the high-return
regime (P6) — each shipped only on a pre-registered, anchor-immune, multi-seed reliability win,
every step watched on the Studio's bias–variance dials.

