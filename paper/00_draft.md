# Spectral Operator Regularization and the Linear–Quadratic–Gaussian Structure of Latent Model-Based Control

*Working draft. Theory is isolated in [`theory/derivations.md`](theory/derivations.md); every
analytical claim has a symbolic reproduction in [`reproduction/`](reproduction/) and a numerical
check in [`verification/`](verification/). Citations in [`references.bib`](references.bib).*

---

## Abstract

We study a latent model-based reinforcement learner whose dynamics are an explicit operator field
`z' = A(z) z + B(z) a`, affine in the action. We show that a single information-geometric objective —
the **operator cross-entropy** `L = log det Σ̂ + tr(Σ̂⁻¹ G)` with `Σ̂ = A G Aᵀ + Q` — subsumes a family
of regularizers we had introduced separately (spectral-band consistency, operator invertibility, and
an entropy-exponent term), because it is the Gaussian cross-entropy of the latent under the operator's
own predicted covariance, i.e. Stein's loss. We then show this objective is the **estimation half of an
LQG problem**: it solves the discrete Lyapunov equation, whose transpose-dual is the LQR Riccati
equation, so the optimal controller is closed form and the two operator branches `(op_d, op_p)` are the
estimation/control (Lyapunov/Riccati) pair, with the constraint `det(op_p) > 0` equal to the LQR
cost-to-go positivity `P ≻ 0`. Empirically, spectral-band regularization lifts a latent learner from
random to PPO-tier on HalfCheetah-v5, and adding operator-consistency is the strongest single lever we
found; we further report (one-seed) evidence for a **critical energy-exponent ratio** `|λ|² = 1/5` at
which the policy determinant, imagined return, and realized return undergo a simultaneous sharp
transition. We are explicit about what is proved, what is empirical, and what is conjectured.

## 1. Introduction

Latent model-based RL [Hafner2020, Hafner2023] learns a world model in a learned latent and optimizes a
policy by imagination. We make the latent dynamics an *operator field* and ask what regularizes that
operator. The thread of the work is a sequence of regularizers — a spectral band on the latent Gram, an
anti-freeze constraint on the dynamics operator's spectrum, an invertibility constraint on the policy
operator, a Lyapunov/Stein consistency term — that we eventually recognize as facets of one
information-geometric loss, and that one loss as one half of classical LQG control [Wiener1948,
Kalman1960a, Kalman1960b, Anderson1990].

## 2. The controlled-operator model

(See [`theory/derivations.md` §0](theory/derivations.md).) Shared encoder `z`; dynamics latent
`d = D(z)`, policy latent `p = P(z)`; each branch an `OperatorDynamics` `A(z)=Â(z)+cI`, affine in `a`
(`∂²z'/∂a²≡0`). Behaviour learning is Dreamer λ-returns through the model with an EMA target value net.

## 3. Spectral structure (theory summary)

- **Band-pinned laws** (§1): `z_std = √⟨μ⟩` (dimension-invariant; measured 0.79 at d=16 and 32) and
  `r_eff/d` constant (measured ≈0.91, linear). *[proved; coefficient empirical]*
- **Lyapunov/Stein covariance** (§2): `Σ = A Σ Aᵀ + Q`, normal-operator form `Σ_i = q/(1−|λ_i|²)`.
  *[proved]*
- **Anti-freeze** (§3): the covariance band is an annulus on `|λ(A_d)|`; a frozen `|λ|→1` blows the
  covariance up. *[proved]*
- **Entropy exponent** (§4): `log det A = Σ log|λ_i|`; `det(op_p)>0` keeps it finite and is the LQR
  `P ≻ 0`. *[proved]*

## 4. The compact loss and LQG (theory summary)

- **Operator cross-entropy** (§6): `L = log det Σ̂ + tr(Σ̂⁻¹ G)` = Stein's loss; unique minimizer
  `Σ̂ = G`; unifies consistency + invertibility + entropy in one functional. *[proved]*
- **LQG / Lyapunov–Riccati duality** (§7): the cross-entropy is the estimation half; the policy is the
  Riccati (control) half; `op_d ↔ Σ`, `op_p ↔ P`; the program's regularizers are the LQG
  preconditions. *[proved, control theory]*
- **`Q`–policy coupling** (§8): `Q` is policy-dependent, so the controller can be closed form but the
  *explorer* cannot — the policy is retained as the data-quality engine. *[argued]*

## 5. Experiments

Apparatus: HalfCheetah-v5 [Todorov2012], deterministic-mean evaluation. SOTA context for calibration:
SAC ≈ 9.6k [Haarnoja2018], TD3 ≈ 9.6k [Fujimoto2018], MBPO ≈ 10–12k [Janner2019], PPO ≈ 1.5–3k.

| arm | what it adds | result (det-eval) | reading |
|---|---|---|---|
| spectral band (cf22) | band on Gram `G` | +1507, 6/6 seeds | random → PPO-tier; the validated win |
| svband (A6) | static anti-freeze on `op_d` | 169 | too weak at the tested strength (`σ_max` stayed ≈1) |
| **+ Stein consistency (A7)** | operator-consistency term | **487, peak 996** | **strongest single operator lever** |
| det(op_p)>0 (A8) | invertibility barrier | barrier binds; eval poor | positivity necessary, not sufficient |
| ratio init (A10, `|λ|²=0.2`) | initialize at the critical ratio | **sharp co-transition** (1 seed) | criticality evidence |
| ratio init+hold (A12) | sit at `0.2` across 4 seeds | *(running — reliability test)* | open |

Two honest caveats: results are PPO-tier (≈0.15× SAC), so the contribution is the *framework and the
spectral mechanism*, not a new SOTA; and the critical-ratio transition is one-seed so far (A12 tests
reliability). The latent geometry is decoupled from return (`corr(cond G, return) ≈ −0.02`); what moves
return is the operator/consistency side.

## 6. Discussion and limitations

The operator cross-entropy makes the world-model objective an LQG estimation problem and the controller
a Riccati solve, with the spectral regularizers as the preconditions. Limitations: the reward must be
locally quadratic for the LQR reading; `A_d(d)` is state-dependent so the regulator is SDRE/iLQR
[Cimen2008, Tassa2012]; PPO-tier returns; the critical ratio is conjectural. Rejected en route (kept to
forestall rediscovery): a golden-ratio operator fixed point and the relations `g_p=1+1/g_d`,
`g_p=1+eff/H` — coincidences of band-pinned constants
([`verification/verify_rejected_coincidences.py`](verification/verify_rejected_coincidences.py)).

## 7. Reproduction

```bash
# symbolic (CAS) reproductions of every proved result
for f in reproduction/*.py;   do python "$f"; done
# numerical verifications against experiment
for f in verification/*.py;   do python "$f"; done
# point the transition detector at a real run:
python verification/verify_critical_transition.py results/runs/abl1-A12critical-s0-*/metrics.jsonl
```

## References

See [`references.bib`](references.bib).
