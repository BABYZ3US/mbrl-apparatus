# Phase-transition notes — the latent as a type-II superconductor

*PM framing, 2026-06-14. Intuition + knob-source, NOT a literal physical claim — see
the caveat at the end.*

## The picture

The latent representation carries a fast, **generalizing "supercurrent"** (a policy that
transfers across states) only while its **vortices** — singular-Gram defects in the
representation — stay **pinned** and don't proliferate. The collapse/NaN failure mode is
**driving past the critical current**: the vortices *depin*, the coherent state unwinds,
and the policy degrades to useless.

This was the literal shape of the cf3 NaN. On `cf3-i0-s2`, `actor/grad_norm` went
0.9 → **46,293** → NaN *exactly as* `imagine/return_mean` crossed 0 (−0.7 → +1.8). That
is a relative-phase **slip** under drive, not an equilibrium transition — driving the
policy through the return≈0 "phase boundary" faster than the (expansive, radius_p≈1.06)
operator could re-lock.

## Knob map (analogy → lever we actually have)

| Superconductor | Lever | Where it lives | Status |
|---|---|---|---|
| **Healing length** ξ = ridge λI on the Gram; finite vortex-core radius | operator radius bound (`model.dual_latent.radius_p`, `model.operator.w_radius`); a representation-Gram ridge would be the *feature*-level version | dynamics.py `structural_penalties.radius`; cf4 | **op-level present; feature-level ridge not yet** |
| **Order-parameter amplitude** \|ψ\| = cond(G)/eff-rank of the representation | `latent/gram_cond`, `latent/gram_eff_rank`, `latent/gram_spectral_entropy` | loop.py `_representation_readouts` | **WIRED (cf4.1)** |
| **Pinning** = regularization/disorder; mild dissipation *helps* | AdamW weight-decay; VAE/Gaussian noise (when on); policy inertia EMA | optim; encoder/dynamics | latent — cf4 runs deterministic; "don't chase a frictionless representation" |
| **Critical current** Jc = trust region on the policy-improvement rate | `actor_clip`, policy inertia `‖θ−θ_ema‖²` (weight-space proxy); a true per-update **KL cap** would be the principled version | loop.py | **partial — proxy only, KL trust region is the gap** |
| **Coupling** = inter-sector phase **stiffness** | `model.dual_latent.couple_weight` (L_couple = ‖W_d d − W_p p‖²) | dual_latent.py | **present, finite/tunable — the soft-not-hard sweet spot** |
| **Operate at the edge, not deep super** | λ cuberoot anneal; keep radius/clip strong enough to pin but not *freeze* z | schedule; cf4 stack | watch — over-damping = a frozen z that holds but doesn't learn |

The coupling row is the payoff: the session-long **"soft beats hard"** finding (finite
`couple_weight`, not hard Koopman/Bellman commutation) now has a mechanism. Too much
stiffness → a rigid superfluid that shatters at the first defect (the brittle
hard-commutation regime); too little → the two condensates desync. You want **finite,
tunable** relative-phase stiffness.

## The three readouts

1. **cond(G) / eff-rank / spectral entropy of the representation Gram** — the live
   distance-to-collapse. `G = Zc^T Zc / B` over a batch; `cond = λ_max/λ_min`,
   `eff_rank = exp(spectral_entropy)`. As the coherent state depins, eigendirections of
   G vanish → cond → ∞, eff_rank → 1. **The early-warning signal cf3 lacked.** *Status:
   WIRED, logged every model update (`latent/gram_*`), no_grad so it cannot perturb
   training.*
2. **Relative-phase drift between the two sectors** — scale-free desync of d and p:
   `‖W_d d − W_p p‖ / mean(‖W_d d‖,‖W_p p‖)` (0 = phase-locked, →1 = slipping). Raising
   `couple_weight` should *lower* it; a run-up precedes a slip. *Status: WIRED for twin
   (`dual/phase_drift`); complements `dual/couple` (the raw L_couple).*
3. **Depinning-threshold (critical-current) estimate** — *intentionally NOT a live
   estimator yet.* Per the caveat below, the threshold is a bifurcation point, not a
   thermodynamic constant, so a "Jc" number would over-promise. The honest proxies to
   watch instead: (a) **headroom** = how close `gram_cond` is to its running max /
   blow-up reference; (b) the **drive-vs-relock ratio** = actor update magnitude
   (`actor/grad_norm`) against the model's re-lock rate (dynamics-loss improvement /
   `dual/p_consistency`). A real estimator would come from depinning / bifurcation
   theory (ramp the drive, watch for the order-parameter knee), not from this doc.

## Honest caveat

A finite network has **no sharp thermodynamic transition**. Collapse is a **driven
dynamical bifurcation**, so the predictive math is **depinning / non-equilibrium
critical phenomena and bifurcation theory**, not equilibrium Landau theory. The analogy
earns its keep as (i) intuition and (ii) a source of the right knobs — ridge ↔ healing
length, trust region ↔ critical current, coupling ↔ phase stiffness. It would **mislead**
if used to expect universal critical exponents or a true latent-heat discontinuity in a
fixed-size net. Treat `gram_cond`/`phase_drift` as engineering distance-to-collapse
gauges, not as evidence of a phase transition.
