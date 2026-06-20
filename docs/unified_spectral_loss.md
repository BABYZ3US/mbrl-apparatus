# Unified spectral loss — combining the theoretical levers (2026-06-15)

A single loss that folds every spectral/operator lever the program has surfaced into one
object, with each term tied to the theory that motivates it and its **validation status**.
The line between *validated* and *proposed* is kept sharp on purpose (claims-ledger
discipline — `docs/claims_ledger.md`): only the band and the behaviour stabilizers are
earned; the operator-side terms are new theory awaiting an arm.

## Notation

- `z ∈ R^d` — encoder latent, `d = latent_dim`.
- `G = E[z zᵀ] ∈ R^{d×d}` — empirical latent second-moment (Gram), eigenvalues `μ₁…μ_d`.
- `A_d(z) = rawA_d(z) + I` — the **cold/dynamics** operator `op_d` (near-identity init),
  singular values `s₁…s_d`, eigenvalues `λ₁…λ_d`. `B(z)` — its affine-in-action map (R15).
- `A_p(z) = rawA_p(z) + I` — the **hot/policy** operator `op_p` (dual twin).
- `Q̂ = E[(z' − A_d z − B a)(·)ᵀ]` — one-step innovation (prediction-residual) covariance.
- `Σ̂` — model-predicted stationary covariance: solution of the **discrete Lyapunov
  (Stein) equation** `Σ̂ = A_d Σ̂ A_dᵀ + Q̂`, i.e. `Σ̂ = Σ_{k≥0} A_d^k Q̂ (A_dᵀ)^k`.
  For **normal** `A_d` it diagonalizes in closed form: `ν_i = q̂_i / (1 − |λ_i|²)` with
  `q̂_i = u_iᵀ Q̂ u_i` along eigenvector `u_i`.
- Band wall `B_{f,c}(x) = relu(x − c)² + w_f · Φ((f − x)/τ)`, `Φ` a sigmoid floor wall;
  `c` ceiling, `f` floor. (The cf22 winner: `c=1, f=0.1, sigmoid floor`.)

## The loss

```
L  =  L_model  +  β · L_behaviour  +  L_spectral
```

### 1. World model  `L_model`   — standard, EXISTING
```
L_model = E_t [ ‖z_{t+1} − A_d(z_t) z_t − B(z_t) a_t‖²        (latent forward model)
              + ‖r̂_t − rew(z_t)‖²                            (reward head — the reg. target)
              + ‖ô_t − dec(z_t)‖²  ]                          (reconstruction, if decoding)
```

### 2. Behaviour  `L_behaviour`  — Dreamer λ-returns, NOT SAC.  VALIDATED
λ-returns through the imagined model with an EMA target value net (`returns.py`), plus the
reward-adaptive stabilizers that solved the seed spread (cf21/cf22, **6/6 stable**):
- hard `log_std` variance floor (reward-adaptive `hi=-1, lo=-4`);
- return gate `leaky_relu`, ratchet, floor 0.1 (anneals the penalty as eval climbs);
- entropy anneal; ratchet imagination horizon (`h_min=15`).
These are earned and stay as-is. **SAC is imported only as tractable ingredients**
(auto-α, clipped-double-value) — never the model-free Q-critic, which would orphan the
operator-field program.

### 3. Spectral regularizer  `L_spectral`  — the heart

```
L_spectral =  w_G · Σ_i B_{f,c}(μ_i)                       (a) empirical covariance band
           +  w_Σ · Σ_i B_{f,c}(ν_i)                       (b) Lyapunov covariance band   [NEW]
           +  w_J · ‖ sort eig(G) − sort eig(Σ̂) ‖²         (c) Lyapunov consistency       [NEW]
           +  w_n · ‖ A_d A_dᵀ − A_dᵀ A_d ‖²               (d) operator normality
           +  w_s · E_{ij} ‖A_d(z_i) − A_d(z_j)‖² / ‖z_i − z_j‖²   (e) operator smoothness
           +  w_k · L_couple(A_d, A_p)                     (f) dual hot/cold coupling
           +  λ_curv · L_Hutch(reward)                     (g) curvature — VESTIGIAL, at floor
```

**(a) Empirical covariance band — `w_G · Σ B_{f,c}(μ_i)`.  VALIDATED (cf22, +1507, 6/6).**
The program's one earned winner. Pins the encoder's latent geometry into the box
`μ_i ∈ [f, c]`. Consequences measured and confirmed: `z_std = √(Tr G/d) = √⟨μ⟩ ≈ 0.79`
(d-invariant), `eff_rank/d ≈ 0.91` (linear). **Caveat that motivates (b):** this term acts
on `G`, which the **encoder alone** can satisfy. It places *no* constraint on `A_d` — so
`op_d` is free to freeze at the identity while the band looks perfectly healthy. That is
exactly how A3 died (`|λ(A_d)|_max = 1.0004`, piled on the unit circle) with a textbook band.

**(b) Lyapunov covariance band — `w_Σ · Σ B_{f,c}(ν_i)`.  NEW / PROPOSED.**
Apply the *same* band to the **model-predicted** stationary covariance `Σ̂ = dlyap(A_d, Q̂)`.
This is the closed form of the covariance, differentiable in the operator weights. With
`ν_i = q̂_i/(1−|λ_i|²)` the two band edges read off as operator constraints:
- **ceiling** `ν_i ≤ c`  ⟺  `|λ_i|² ≤ 1 − q̂_i/c`  →  keeps `|λ_i|` **off the unit circle**
  = the **anti-freeze** term. This is the lever A3 was missing; `gd↔det` correlates **+0.58**.
- **floor** `ν_i ≥ f`  ⟺  `|λ_i|² ≥ 1 − q̂_i/f`  →  keeps modes alive = **anti-collapse**.

So (a) shapes the encoder, (b) shapes the dynamics operator; they are the *same band on two
different objects* and are **non-redundant** because encoder and operator are separately
parametrized. (b) is the unification: the covariance band and the operator-spectrum annulus
are one object seen through Lyapunov.

> **Implementable proxy (recommended first).** Eigenvalues of a non-normal matrix have
> unstable backward (see `dynamics.py`); the codebase uses **singular values** instead, which
> equal `|λ_i|` as `A_d → normal` (term (d) enforces this). So implement (b) as a band on the
> singular values: `w_Σ · Σ_i B_{ρ_min, ρ_max}(s_i(A_d))` with `ρ_max = √(1 − q̂/c) ≲ 1`
> (anti-freeze) and `ρ_min` (anti-collapse). This **extends the existing `w_radius` term**
> `relu(s_max − 1)²`, which only caps the top and has *no floor* — the missing piece is the
> ceiling pulled below 1 and a floor on the bulk. No Lyapunov solve, no `Q̂` estimate needed
> for the proxy.

**(c) Lyapunov consistency — `w_J · ‖eig(G) − eig(Σ̂)‖²`.  NEW / PROPOSED.**
At a faithful world model the empirical and predicted covariances coincide. Penalizing their
spectral gap ties `A_d` to the encoder geometry directly (the dual twin's self-consistency,
made spectral). Measured gap today: the exact Stein identity holds only approximately
(implied `Q̂` had a small negative eigenvalue) because `A_d` is state-dependent and `z` is the
encoder's, not the operator's own stationary measure — (c) is the term that *drives* that gap
toward zero rather than assuming it.

**(d) Normality — `w_n‖A_dA_dᵀ − A_dᵀA_d‖²`.  EXISTING (`w_normal=0.05`).**
Makes `s_i ≈ |λ_i|` so the closed-form `ν_i = q̂_i/(1−|λ_i|²)` is valid and (b)'s singular-value
proxy is exact. Also gives clean, non-defective modes.

**(e) Smoothness — `w_s · E ‖ΔA_d‖²/‖Δz‖²`.  EXISTING (`w_smooth=0.1`).**
Operator Lipschitz in `z` — keeps `A_d(z)` a coherent bundle, so a single `Σ̂` linearization
is meaningful.

**(f) Dual coupling — `w_k · L_couple(A_d, A_p)`.  EXISTING (`couple_weight=0.1`).**
Couples the cold (`op_d`) and hot (`op_p`) fields. Measured asymmetry (robust across seeds):
`op_p` is the hot, rotational field (generator norm ~2.2, ~3× more complex eigenvalues),
`op_d` the cold, near-real field whose warmth tracks performance. Equal generator eff_rank
(~12.2, ratio 0.998) is a conserved quantity worth preserving — a candidate explicit term
`(eff_rank(A_d) − eff_rank(A_p))²`. The golden-ratio / `gp=1+1/gd` / `gp=1+eff/H` couplings
were **tested and rejected** (coincidences of band-pinned constants) — *not* included.

**(g) Curvature — `λ_curv · L_Hutch(reward)`.  VESTIGIAL.**
The founding isotropic 2-probe Hutchinson penalty (R4/R16). cf22 set the record with it
**pinned at its `1e-4` floor** → not load-bearing. Kept at the floor for continuity, never a
primary lever. Isotropic, latent-coords, detached, reward-only (never policy, R10).

## Why this is "all the levers" in one place

| object | term | what it pins | status |
|---|---|---|---|
| encoder geometry | (a) band on `G` | latent spectrum `μ_i ∈ [f,c]` | **validated** |
| dynamics spectrum | (b) band on `Σ̂(A_d)` | `|λ(A_d)|` annulus = anti-freeze + anti-collapse | proposed |
| model faithfulness | (c) `eig(G)=eig(Σ̂)` | empirical = predicted covariance | proposed |
| operator regularity | (d) normal, (e) smooth | clean modes, Lipschitz bundle | existing |
| field coupling | (f) `L_couple` | hot/cold `op_p`/`op_d` balance | existing |
| reward surface | (g) Hutchinson | curvature (floor only) | vestigial |

The single new idea making it *one* loss: **the band is a statement about a covariance, and a
covariance is a Lyapunov image of an operator.** Apply the band to the *measured* covariance
(a) and you constrain the encoder; apply it to the *Lyapunov-predicted* covariance (b) and you
constrain the operator; make the two agree (c) and the world model is forced consistent. (a)
is earned; (b)+(c) are the falsifiable next step.

## Defaults & the one arm that tests the new terms

Keep the cf22 canonical for (a),(d),(e),(f),(g): `w_G(band)=5.0, c=1.0, f=0.1, sigmoid;
w_n=0.05, w_s=0.1, couple=0.1, λ_curv=1e-4`. The decisive arm adds **only** the proxy form of
(b) — a singular-value band floor on `A_d`:

```
model.operator.w_radius_floor = <w_Σ>     # NEW: band s_i(A_d) into [ρ_min, ρ_max]
model.operator.radius_max     = ρ_max≈0.95 # anti-freeze ceiling, pulled below 1
model.operator.radius_min     = ρ_min≈0.30 # anti-collapse floor
```

Prediction under test: un-freezing `op_d` (forcing `s_max(A_d) < 1`) lifts `eval/return_det`
relative to A0 — the operator-side, mechanistic complement to the encoder-side band. If it
does, (c) and the full Lyapunov band are the follow-on; if it doesn't, the gd↔det link was
correlation without lever and (b) is dropped.
