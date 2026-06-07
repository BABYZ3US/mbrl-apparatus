# MBRL Research Findings — Comprehensive Report
## Hessian-Regularized Reward Models in Model-Based Reinforcement Learning

*User-supplied 2026-06-06: the summary of findings from the ORIGINAL investigation.
This is the primary empirical record this project builds on; the founding doc
(`../../mbrl_foundations_and_framework.md`) and claims ledger (`claims_ledger.md`)
should be read alongside it. Apparatus-relevant consequences are flagged at the end.*

---

## Origin
The project began with a single empirical question: in model-based reinforcement learning, the policy optimizer exploits errors in the learned reward model, producing trajectories with high predicted reward but low actual reward (reward hacking). Does penalizing the Frobenius norm of the reward model's Hessian, λ‖∇²R‖²_F, suppress this exploitation?
The empirical answer was yes, but only at exactly second-order smoothness. Lower-order penalties (weight decay, spectral norm) did not help. The question that drove everything downstream was: *why specifically s = 2?*

## 1. The Central Falsification Test
The single most important experiment in the project is the regularizer ablation on HalfCheetah-v5.
**Setting.** HalfCheetah-v5 (MuJoCo), 200K environment steps, 3 seeds per condition, T4 GPU on Colab (~2 hours wall-clock). The Hessian penalty uses Hutchinson's 2-probe stochastic estimator.

| Penalty | Final Return | Outcome |
|---------|:------------:|---------|
| Baseline (no penalty) | −165 ± 41 | catastrophic reward exploitation |
| s = 0 (weight decay) | −152 ± 38 | effectively baseline |
| s = 1 (spectral / Jacobian) | unstable | divergence in 2/3 seeds |
| **s = 2 (Frobenius Hessian)** | **+98 ± 23** | substantial improvement |

The framework would have been refuted if any of s = 0, s = 1 had matched s = 2's improvement. They did not. Combined with Nesterov's result that s ≥ 2 is required for fast first-order convergence, two independent reasons select the exponent.

## 2. The Core MBRL Result
**Setting.** HalfCheetah-v5, four conditions, 3 seeds each, 200K steps.

| Condition | Final Return |
|-----------|:------------:|
| Baseline (no penalty, no smoothing) | −165 ± 41 |
| Hessian only (annealed λ) | −89 ± 35 |
| DreamSmooth only (σ_t = 1) | −137 ± 28 |
| **DreamSmooth + Hessian** | **+98 ± 23** |

Both ingredients contribute substantively; the components are not redundant.

## 3. The Null-Lagrangian Test
Is ‖∇²R‖²_F equivalent to (ΔR)² in stochastic NN training? (They share the biharmonic Euler–Lagrange equation in flat ℝᵈ.)
**Setting.** HalfCheetah-v5, 200K steps, 5 seeds.

| Estimator | Non-negative? | Final Return |
|-----------|:-------------:|:------------:|
| Frobenius ‖Hv‖² (Hutchinson 2-probe) | Always | +73 |
| Laplacian-trace (vᵀHv)² (2-probe) | Always | +9 |
| No penalty | — | −39 |
| Laplacian-2 product (v₁ᵀHv₁)(v₂ᵀHv₂) | Can be < 0 | −51 |

The non-negative estimators outperform the sign-indefinite ones; ordering by performance matches ordering by non-negativity.
**Thermodynamic consistency follow-up.** Clamping the sign-indefinite estimator to be non-negative (max with zero) transforms it into a competitive performer:

| Estimator | Non-negative? | Mean Return |
|-----------|:-------------:|:-----------:|
| **Lap-2 clamped (forced ≥ 0)** | Forced | **+41** |
| Frobenius | Always | −40 |
| Lap-2 unclamped | Can be < 0 | −79 |

The clamping experiment is the cleanest evidence that **non-negativity itself** — not any other property of the estimator — is what matters.

## 4. Theory Validation on Pendulum
4.1 Gradient variance decreases monotonically with λ (~2 orders of magnitude). ✓
4.2 Horizon-variance amplification: linear at H ≤ 10, super-linear beyond; penalty controls growth. ✓
4.3 Bias-variance U-curve: λ* > 0 in every condition; λ* grows with σ_r. ✓
4.4 Denoising at σ_r = 3: unregularized test MSE 0.376 → regularized (λ = 10⁻³) 0.197. ✓ (strongest single evidence for the low-pass-filter reading)
4.5 ‖∇²R‖²_F falls ~36 → ~0.02 at largest λ (≈3 orders of magnitude). ✓

## 5. Transversality Measurement
**Setting.** Synthetic supervised task with HalfCheetah-style 17-dimensional state.
**Result.** Angles between ∇²R and ∇²T distribute in **60°–71°** in Frobenius space — substantially distinct curvature subspaces.

## 6. Multi-Kernel Ablation (Pendulum)
R only — best. R + T — close. R + T + π — underperforms. **Net rule:** penalize R, optionally T, never π.

## 7. Annealing Schedule
HalfCheetah-v5: constant λ = 0.5 vs step-anneal (λ = 0.5 first 100K, 0 last 100K). Step-anneal wins by ~25 return units; constant λ caps late performance.

## 8. Latent-Space MBRL
Generalizes, with the caveat that the penalty must be applied **in latent coordinates**.

## 9. Cross-Algorithm Generalization (MBPO)
Improvement present but smaller (short branched rollouts already mitigate exploitation). Effect magnitude scales with exposure to long-horizon imagined returns.

## 10. Affine vs MLP Dynamics
Affine-in-action gives ~2× tighter gradient-variance control under the penalty.

## 11. The Recipe (Empirically Supported)
```
loss_R = MSE(R_pred, R_obs) + λ(t) * ‖∇²R‖²_F     (Hutchinson 2-probe, step-annealed)
dynamics: affine in a (preferred)
reward signal: DreamSmooth (σ_t ≈ 1)
penalty on: R only (or R + T); never π
schedule: step-anneal (strong early, release in second half)
estimator: unbiased and NON-NEGATIVE (Frobenius, or clamped Laplacian)
latent: apply penalty in latent coordinates
```

## 12–13. What is/isn't established
Established: mechanism (3-orders ‖∇²R‖² reduction), Pendulum predictions quantitative, s=2 uniquely selected, null-Lagrangian equivalence in NN training, full-recipe benchmark win, generalization across algorithm families/latents/multi-kernel choices.
Not established: breadth (mostly HalfCheetah), mechanism uniqueness (NTK/implicit-regularization accounts not ruled out), dimensional/physics claims (separate document), penalty novelty (Peebles et al. 2020 — contribution is the MBRL application, the s∈{0,1} falsification, the null-Lagrangian test, the Sobolev interpretation).

## 14. Highest-Priority Next Experiments
Multi-env replication; D4RL offline; generic-vs-critical smoothness sweep; Stone-rate curves; latent-dimension sweep; schedule ablation incl. (t₀/(t₀+t))^⅓.

## 15. One-line summary
A cheap, theoretically principled (critical Sobolev order s = 2), empirically validated, isotropic reward-curvature regularizer — with reward smoothing and a step-annealing schedule — that suppresses reward exploitation, stabilizes long-horizon latent imagination, and offers a credible sample-efficiency gain via joint reward+dynamics curvature control, at small constant compute overhead.

## 16. References
Peebles et al. (2020) ECCV; Hutchinson (1990); Avron & Toledo (2011); Roosta-Khorasani & Ascher (2015); Hansen et al. (2024) DreamSmooth; Janner et al. (2019) MBPO; Nesterov (1983, 2004); Stone (1982); Birman & Solomjak (1967).

---

## Apparatus consequences (added on integration, 2026-06-06)

1. **§3 redefines "clamped":** clamping = max(estimator, 0) on the sign-indefinite
   Lap-2 product estimator — NOT a schedule property. The clamped trace beat
   Frobenius (+41 vs −40) in the follow-up run set. → `laplacian_trace_penalty`
   now has `clamp=True` by default; the user's narrowed-down recipe
   ("clamped decaying trace") = `penalty.form=laplacian_trace` (clamped) + a
   decaying schedule.
2. **§7's working λ was 0.5 on HalfCheetah** — two to three orders above this
   repo's base default (1e-3). Penalty dosing should be revisited per-environment
   (consistent with the multitask under-dosing finding F2).
3. §5's transversality was measured on a synthetic 17-d supervised task, not the
   live environment — matches this repo's experiment-10 design.
