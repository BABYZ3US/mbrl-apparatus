# Curvature-Regularized MBRL — the validated recipe

A self-contained, directly-usable summary of the model-based RL recipe that came out of this project.
Tags: `[EMPIRICAL]` = observed in experiments; `[PROVED]` = derivation/identity; `[THEORY]` =
proved in an idealized regime, not the full NN setting; `[HEURISTIC]` = soft prior, test don't assume.

---

## TL;DR

Penalize the **reward model's Hessian** (not its Jacobian, not the policy), **isotropically**, estimated
with a **2-probe Hutchinson** Hessian-vector product, with an **annealed** coefficient, on top of
**episode reward smoothing**. It's cheap (2 HVPs/batch), it denoises, and it lets you run **longer imagined
rollouts** — which is the real sample-efficiency win. Full stack took HalfCheetah `−165 → +98`; ablations
show s=1 (Jacobian) **diverges** while s=2 (Hessian) trains stably.

## 1. The penalty

$$
\mathcal L \;=\; \mathcal L_{\text{fit}} \;+\; \lambda\,\mathbb E_{(z,a)}\big\|\nabla^2_{(z,a)}\hat R_\theta(z,a)\big\|_F^2
$$

- **Order s = 2 (the Hessian), not s = 1.** `[EMPIRICAL]` head-to-head: s=0 (weight decay) → no gain;
  s=1 (Jacobian/spectral) → diverges 2/3 seeds; s=2 (Frobenius Hessian) → works. `[PROVED]` reason: the
  s=2 gradient flow is 4th-order **biharmonic**, singularity-free, where s=1's is not. Also `s = d/2`
  critical-regularity (Birman–Solomjak + Sobolev trace) `[PROVED]`.
- **Isotropic** (plain Frobenius norm). Weighting eigendirections breaks the `O(k)`-invariance of the
  `Δ²` symbol — do **not** anisotropize. `[PROVED]`
- Apply in **latent coordinates** `(z,a)`.

## 2. The estimator (the compute lever)

`‖∇²R̂‖²_F = E_v‖∇²R̂ · v‖²` for Rademacher `v` — one Hessian-vector product per probe; **N = 2 probes**
suffice. `[PROVED]` unbiased, `[EMPIRICAL]` at N=2. Never form the full Hessian.

```python
import torch

def reward_hessian_penalty(R_hat, x, n_probes=2):
    """x: (batch, d) latent-action inputs, requires_grad=True.
       R_hat(x) -> (batch,) scalar reward. Returns scalar penalty = E_v||H v||^2 = ||H||_F^2."""
    out = R_hat(x).sum()
    grad = torch.autograd.grad(out, x, create_graph=True)[0]        # (batch, d)
    pen = 0.0
    for _ in range(n_probes):
        v = (torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1)   # Rademacher ±1
        Hv = torch.autograd.grad((grad * v).sum(), x, create_graph=True)[0]    # (batch, d)
        pen = pen + (Hv ** 2).sum(dim=1)                            # ||H v||^2 per sample
    return (pen / n_probes).mean()
```

(The Laplacian-trace form `(ΔR̂)²` is statistically indistinguishable in NN training `[EMPIRICAL]` — use
whichever is cheaper in your autodiff stack, but keep the **unbiased 2-probe** version; the 1-probe biased
trace underperforms.)

## 3. What to penalize — and what never to

- **Penalize `R̂`** (the reward model). This is the validated target.
- **Optionally add the dynamics model `T`'s Hessian** for a sample-efficiency gain via transversality
  (R,T Hessians are empirically transversal, `60°–71°`). `[THEORY]` / partial — **start R-only, add T only
  if you measure a gain.**
- **Never penalize the policy `π`.** `R+T+π` was *worse* — the policy needs curvature freedom for action
  selection. `[EMPIRICAL]`

## 4. Schedule

- **Anneal λ:** strong early, decay to a residual floor — `λ*(t) ∝ (t₀/(t₀+t))^{1/3}` `[THEORY]`, or a
  step-anneal approximation (strong through exploration, released for fine policy tuning). Beats constant λ
  by ~25 return units; constant λ caps late performance by distorting the converged reward. `[EMPIRICAL]`
- **Dose it by measurement, not guessing:** log the actual penalty value and drive it toward the
  true-curvature `O(1)` scale. A real run was under-dosed by ~2 orders of magnitude because λ was guessed.

## 5. The full stack (what actually produced the win)

`DreamSmooth (episode-level reward smoothing) + Hessian penalty + annealing`. Both ingredients contribute:
Hessian-only `−89`, DreamSmooth-only `−137`, **combined `+98`** (from a `−165` baseline) on HalfCheetah.
`[EMPIRICAL]`

## 6. Where the sample-efficiency actually comes from

The penalty controls **horizon-variance amplification** (imagined-return variance grows super-linearly with
rollout length), so you can run **longer imagined rollouts** than an unregularized model tolerates. Since
real environment steps dominate cost, trading controlled imagination variance for fewer real steps is the
main lever. `[EMPIRICAL]` Largest in long-horizon (Dreamer-style) settings; smaller in short-horizon MBPO.
Bonus: under reward-observation noise the penalty acts as a **denoiser** (test MSE `0.38 → 0.20` at σ=3).
Tip: **affine-in-action dynamics** `T = z + f(z) + G(z)a` removes the dynamics-curvature floor and tightens
variance control ~2×. `[EMPIRICAL]`

## 7. Architecture defaults

Dreamer/MBPO-style latent MBRL: encoder `e_φ: obs → z ∈ ℝ^k` (EMA-stabilized; compact `k` as a soft prior,
`[HEURISTIC]`), affine-in-action latent dynamics `T_ψ`, reward model `R̂_θ` (the regularization target),
policy/value on imagined latent rollouts. **Normalize imagined returns** (Dreamer-V3 style: scale
λ-returns by an EMA of their 5–95% range) before trusting any ablation — un-normalized returns through long
rollouts cause variance explosions that hit all arms identically and masquerade as treatment effects.

## 8. Do / Don't checklist

| Do | Don't |
|--|--|
| Penalize `R̂`'s **Hessian** (s=2) | Use a Jacobian/spectral (s=1) penalty — it diverges |
| Use the **2-probe Hutchinson** HVP | Form the full Hessian, or use the 1-probe biased trace |
| Keep the penalty **isotropic** | Weight eigendirections (breaks `O(k)`-invariance) |
| **Anneal** λ; dose by logged penalty value | Set λ by guessing, or hold it constant |
| Penalize `R` (optionally `T`) | Ever penalize the policy `π` |
| Use it to run **longer imagination** | Ignore return normalization before comparing arms |
| Difficulty-match generalization metrics | Trust raw train−interp gaps (they overstate) |
| Validate with a **matched-null** statistic | Assume an effect is real before the null survives |

## 9. Bonus: representation + value learning (Koopman–Bellman)

If you learn a representation with linear latent (Koopman) dynamics and a value/Bellman operator on top, do
**not** enforce hard commutation/invariance between them — forcing `[K,B]=0` collapses training (the
feature Gram goes singular, `cond → ~10¹⁶`; eigendecomposition blowups, exploding losses). Use a **soft**
spectral regularizer + a non-commuting correction, and **ridge/Tikhonov-precondition the feature Gram**.
Diagnostics: (1) monitor `cond(G)` and ridge when it blows up rather than fighting it; (2) a **positive TD/
Bellman residual floor** can signal reward mass *outside the controllable subspace* of your representation —
a representation mismatch, so enrich features rather than train longer. `[THEORY + toy-verified]`

## 10. Honest caveats

- **Small latent `k≈4`** is `[HEURISTIC]` — it does not uniquely select 4; treat as a soft prior toward
  compact latents (which independently helps via the sample-efficiency "GPS effect").
- **Transversality / joint R+T** is `[THEORY]`, partial in general, and hard to demonstrate cleanly (needs
  tasks with curvature in every direction). Start R-only.
- Prior art for the penalty *object* + Hutchinson estimator: Peebles et al. 2020 (GAN disentanglement); the
  novelty here is the MBRL application and the s=2 rationale. The reward-smoothing ingredient is DreamSmooth.

---

*Source: `mbrl_foundations_and_framework.md` (R1–R17 + Part 2 design); `mbrl/results/analysis_*.md`
(dosing, matched-null, apparatus findings); `koopman_bellman_NB_correspondence_2026-06-13.md` (§9).*
