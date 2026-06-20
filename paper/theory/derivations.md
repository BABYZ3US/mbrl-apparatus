# Theoretical Derivations
## Spectral Operator Regularization and the Linear–Quadratic–Gaussian Structure of Latent Model-Based Control

*All analytical results for the operator-field latent-MBRL program are collected in this single
document. Each result is tagged **[proved]**, **[conjectured]**, or **[empirical]** following the
project's claims-ledger discipline. Every **[proved]** result has a symbolic reproduction in
[`../reproduction/`](../reproduction/); numerical checks against experiment are in
[`../verification/`](../verification/). Citation keys resolve in [`../references.bib`](../references.bib).*

---

### 0. Notation and the controlled-operator model

Let `z ∈ ℝ^d` be the encoder latent and `G = 𝔼[z zᵀ] ∈ ℝ^{d×d}` its second-moment (Gram) matrix,
with eigenvalues `μ₁ ≥ … ≥ μ_d ≥ 0`. The dual-latent model [Hafner2020] splits `z` into a
**dynamics latent** `d = D(z)` and a **policy latent** `p = P(z)`. Each branch carries an
operator-valued, affine-in-action dynamics

$$z' \;=\; A(z)\,z + B(z)\,a, \qquad A(z) = \widehat A(z) + c\,I,$$

with `∂²z'/∂a² ≡ 0` (the **R15** drop-in property; the action channel is exactly linear so the
curvature penalty of [Gunasekar2023]-style smoothing applies only to the reward). `A` has singular
values `s₁ ≥ … ≥ s_d` and complex eigenvalues `λ₁,…,λ_d`; `c` is the identity-shift coefficient
(`c=1` is the near-identity default). Process/innovation noise is `w ∼ 𝒩(0,Q)`. We write the
operators of the two branches `A_d` (dynamics) and `A_p` (policy).

---

### 1. The latent spectral band and its pinned laws

The validated regularizer is the two-sided spectral **band** on the Gram eigenvalues
[**empirical**; cf22, +1507 return, 6/6 seeds]:

$$\mathcal{B}_{f,c}(\mu) \;=\; \operatorname{relu}(\mu-c)^2 \;+\; \Phi\!\big(f-\mu\big),$$

a quadratic ceiling at `c` and a sigmoid floor wall `Φ` at `f`, pinning `μ_i ∈ [f,c]`. Two exact
consequences follow.

**Proposition 1 (latent-scale identity). [proved]**
$$z_{\mathrm{std}} \;=\; \sqrt{\operatorname{Tr} G / d} \;=\; \sqrt{\langle\mu\rangle},$$
where `⟨μ⟩` is the mean Gram eigenvalue.

*Proof.* `z_std² = (1/d)\sum_i \mathrm{Var}(z_i) = (1/d)\operatorname{Tr}G`, and
`\operatorname{Tr}G = \sum_i \mu_i = d\langle\mu\rangle`. ∎

Because the band pins `⟨μ⟩` independently of `d`, `z_std` is **dimension-invariant**. Measured:
`0.788` (d=16) and `0.786` (d=32) against the predicted `√⟨μ⟩` of `0.787 / 0.785` — see
[`../verification/verify_zstd_identity.py`](../verification/verify_zstd_identity.py).

**Proposition 2 (effective-rank linearity). [proved, empirical coefficient]**
With effective rank `r_{\mathrm{eff}}(G) = \exp\!\big(H(\mu/\!\operatorname{Tr}G)\big)`,
`H` the Shannon entropy of the normalized spectrum [Roy2007], a spectrum that is band-pinned to a
fixed *shape* gives `r_{\mathrm{eff}}/d = \text{const}`. *Proof.* `H` of a distribution that is the
`d`-fold tensor of a fixed per-mode law grows as `log d + H₀`, so `r_eff = e^{H} ∝ d`. ∎
The coefficient is empirical: `r_eff/d ≈ 0.91` at both `d=16` (≈14.6) and `d=32` (≈29.0). This is
a *consequence* of the band, not an independent lever — latent geometry is decoupled from return
(`corr(cond(G), return) ≈ −0.02`).

---

### 2. The Lyapunov / Stein covariance closed form

Under the linear latent flow the marginal second moment propagates as
`M' = 𝔼[(Az+w)(Az+w)ᵀ] = A M Aᵀ + Q`. Stationarity `M'=M=Σ` gives the **discrete Lyapunov
(Stein) equation** [Anderson1990]:

$$\boxed{\;\Sigma \;=\; A\,\Sigma\,A^{\top} + Q\;}\qquad\Longleftrightarrow\qquad
\Sigma = \sum_{k\ge 0} A^{k}\,Q\,(A^{\top})^{k},$$

the series converging iff the spectral radius `ρ(A)<1`. This `Σ` is the closed form of the latent
covariance as a function of the operator parameters.

**Proposition 3 (normal-operator diagonalization). [proved]**
If `A` is normal, `A = UΛUᴴ` with `|λ_i|<1`, and `\tilde Q = Uᴴ Q U`, then in the eigenbasis
$$\tilde\Sigma_{ii} \;=\; \frac{\tilde Q_{ii}}{1-|\lambda_i|^2}.$$

*Proof.* In the eigenbasis the Stein equation decouples on the diagonal:
`\tildeΣ_{ii} = |λ_i|² \tildeΣ_{ii} + \tilde Q_{ii}`, solved by the stated ratio. ∎
Symbolic reproduction: [`../reproduction/lyapunov_covariance.py`](../reproduction/lyapunov_covariance.py).
The normality assumption is enforced in practice by the operator penalty
`‖AAᴴ − AᴴA‖²` (so singular values stand in for `|λ|` with stable gradients).

---

### 3. The operator-spectrum band (anti-freeze) as a covariance band

Proposition 3 turns any constraint on the covariance spectrum into a constraint on the **operator**
spectrum. The Gram-band `μ_i ∈ [f,c]`, read through `μ_i = q_i/(1-|λ_i|^2)` with `q_i=\tilde Q_{ii}`,
becomes an **annulus** on the operator eigenvalues:

$$\mu_i \le c \iff |\lambda_i|^2 \le 1 - \tfrac{q_i}{c}\quad(\text{anti-freeze ceiling}),\qquad
\mu_i \ge f \iff |\lambda_i|^2 \ge 1 - \tfrac{q_i}{f}\quad(\text{anti-collapse floor}).$$

The ceiling keeps `|λ|` strictly inside the unit circle; a **frozen** operator `|λ|→1` sends the
covariance `q/(1-|λ|²)→∞` (marginal stability), the failure mode the ceiling prevents. The
implementable proxy bands the singular values directly,
`\mathcal{B}_{\rho_{\min},\rho_{\max}}(s_i)` with `ρ_max = \sqrt{1-q/c} < 1`; mapped to a single
covariance band on the empirical Gram (§1) and the predicted covariance Σ̂ (§6), this is the
non-redundant operator-side term, because the encoder can satisfy the Gram band on its own while
leaving the operator free to freeze (see [`derivations`](#) §6 and the experimental log). Reproduction:
[`../reproduction/band_to_annulus.py`](../reproduction/band_to_annulus.py).

---

### 4. The determinant and the entropy exponent

The log-determinant of the operator is the per-step log phase-space-volume change:

$$\log\det A \;=\; \sum_i \log|\lambda_i| \;=:\; \text{entropy exponent},$$

the sum of the discrete-time Lyapunov exponents; Pesin's formula identifies the sum of the *positive*
exponents with the Kolmogorov–Sinai entropy rate [Pesin1977]. Hence:

- `det A_p ≠ 0` ⟺ `A_p` invertible (no control mode collapses to a singular direction);
- `det A_p > 0` ⟺ `A_p ∈ GL⁺(d)` (the identity component — orientation-preserving, natural for a
  rotational operator, whose complex eigenvalues pair up to keep `det` real and positive);
- together they keep `log det A_p` **finite and real** (a singular `A_p` gives `log det → −∞`, an
  entropy singularity).

In the control reading (§7) `det(A_p) > 0` is exactly the cost-to-go positivity `P ≻ 0` of the LQR
solution. Reproduction: [`../reproduction/entropy_exponent.py`](../reproduction/entropy_exponent.py).

---

### 5. The dissipative / conservative asymmetry

The two operators play thermodynamically dual roles, robust across seeds and training [**empirical**]:

| operator | constraint | `det` | entropy exponent | role |
|---|---|---|---|---|
| `A_d` (dynamics) | `s_i < 1` (svband) | `<1` | `log det<0` | **dissipative** — contracts, forgets (the bath) |
| `A_p` (policy) | `det>0` | `>0,\approx1` | `log det\approx0` | **conservative** — invertible, volume-preserving (the system) |

The *energy/entropy exponent ratio* is `\log\det(A_d)\,/\,\log\det(A_p)`. The Lyapunov term (§2,§6)
sets its **starting** value; the two determinant constraints hold its **sign structure** for the run.

---

### 6. The operator cross-entropy — one compact loss

**Theorem 1 (operator cross-entropy = Stein's loss). [proved]**
For `p_{\mathrm{data}}=\mathcal{N}(0,G)` and `p_{\mathrm{op}}=\mathcal{N}(0,\widehat\Sigma)` with
`\widehat\Sigma = A G Aᵀ + Q`, the Gaussian cross-entropy is

$$H(p_{\mathrm{data}},p_{\mathrm{op}}) \;=\; \tfrac12\big[d\log 2\pi + \log\det\widehat\Sigma +
\operatorname{tr}(\widehat\Sigma^{-1}G)\big].$$

Dropping the additive constant, the operator loss is

$$\boxed{\;\mathcal{L}(A,Q) \;=\; \log\det\widehat\Sigma \;+\; \operatorname{tr}\!\big(\widehat\Sigma^{-1}G\big),
\qquad \widehat\Sigma = A G A^{\top}+Q.\;}$$

Its KL form is **Stein's loss** / the log-determinant (Burg) matrix divergence [James1961, Amari2016]:

$$D(G\,\|\,\widehat\Sigma) \;=\; \operatorname{tr}(\widehat\Sigma^{-1}G) - \log\det(\widehat\Sigma^{-1}G) - d \;\ge 0,
\quad =0 \iff \widehat\Sigma = G.$$

*Proof.* `−\log p_{op}(z) = ½[d\log2π + \log\det\widehat\Sigma + zᵀ\widehat\Sigma^{-1}z]`; taking
`𝔼_{p_data}` and using `𝔼[zᵀ\widehat\Sigma^{-1}z]=\operatorname{tr}(\widehat\Sigma^{-1}G)` gives `H`.
`H = D(\,\cdot\,) + ½\log\det G + ½ d(1+\log2π)`, and `D≥0` with equality iff `\widehat\Sigma=G` is the
Gaussian KL [Cover2006]. ∎

**Proposition 4 (fixed point and what it subsumes). [proved]**
`∂\mathcal{L}/∂\widehat\Sigma = \widehat\Sigma^{-1} - \widehat\Sigma^{-1}G\widehat\Sigma^{-1} = 0 \Rightarrow \widehat\Sigma = G`,
i.e. `A G Aᵀ + Q = G` — the Stein stationarity. The single functional therefore unifies the three
previously separate terms:

| separate term | inside `𝓛` |
|---|---|
| Stein/Lyapunov consistency | the fixed point `\widehat\Sigma=G` (information geometry, not Euclidean) |
| `det(A_p)>0` / invertibility | `\operatorname{tr}(\widehat\Sigma^{-1}G)→∞` as `\widehat\Sigma` loses rank — a built-in barrier |
| entropy exponent | the `\log\det\widehat\Sigma` term |

Symbolic proof of the fixed point and the Stein-loss identity:
[`../reproduction/operator_cross_entropy.py`](../reproduction/operator_cross_entropy.py).

**Proposition 5 (spectral separation and rational reduction). [proved]**
Let `ν₁,…,ν_d = eig(Σ̂⁻¹G)` be the generalized eigenvalues of `(G,Σ̂)` (read off a triangular
factorization — Cholesky `Σ̂=LLᵀ` then the symmetric `L⁻¹GL⁻ᵀ`, or the QZ generalized-Schur form).
Since `tr` and `det` are the sum and product of eigenvalues, the loss **separates** into identical
scalar per-mode losses,

$$D(G\,\|\,\widehat\Sigma) \;=\; \sum_{i=1}^{d}\varphi(\nu_i),\qquad
\varphi(\nu)=\nu-\log\nu-1\ \ (\ge 0,\ =0\ \text{at}\ \nu=1),$$

with the determinant (normalizer) as the **product** `det(Σ̂⁻¹G)=∏_i ν_i` and the likelihood as the
matching product `e^{-D}=∏_i e^{-\varphi(\nu_i)}` (the triangularization decorrelates the modes). The
per-mode gradient `φ'(ν)=1-1/ν` never needs the matrix inverse. Restricting the generalized spectrum
to `ℚ₊` makes the loss exact — a rational minus logs of rationals, the transcendental content living
in the `ℤ`-span of `{\log p:p\ \text{prime}\}`. In the **conservative / volume-preserving** case
`det(Σ̂⁻¹G)=1` (a reciprocal spectrum, the `det(op_p)=1` structure of §5), the log terms cancel and
`D ∈ ℚ` outright — e.g. spectrum `{5/4,4/5,1}` gives `D = 1/20` exactly. Reproduction:
[`../reproduction/stein_loss_separation.py`](../reproduction/stein_loss_separation.py).

**Remark (binary / 2-adic exactness). [proved]**
Fixed-width binary arithmetic is `2`-adic. In the conservative regime the loss is rational
(`D = tr(Σ̂⁻¹G) − d`), so the triangularization and the loss value are computable **exactly** in
fixed-width binary by Hensel/Dixon `2`-adic lifting (`x ← x(2−ax)`) and rational reconstruction —
no float rounding, no fraction blow-up. Two boundaries: `ℚ₂` is **unordered**, so this certifies the
loss *value* (verification), not the *minimization* (which needs the real order); and off `det=1`
the `log` is genuinely transcendental, so the purely-rational, `2`-adic-exact regime is exactly the
conservative manifold of §5. The critical ratio `1/5` is a `2`-adic unit (`5` odd). Reproduction:
[`../reproduction/padic_exact_loss.py`](../reproduction/padic_exact_loss.py).

A **pure-parameter** variant targets the operator's *stationary* covariance `Σ_∞(A,Q)` (§2) against a
desired band-covariance `Σ_⋆`: `D(Σ_∞(A,Q)\,\|\,Σ_⋆)`. This is `KL` of the operator's own stationary
law against a target — an operator cross-entropy depending on `A` alone (given `Σ_⋆`), and the locus
where the spectral ratio of §9 is imposed.

---

### 7. The Linear–Quadratic–Gaussian structure (Lyapunov–Riccati duality)

The cross-entropy is the **estimation** half of an LQG problem [Kalman1960a, Kalman1960b, Wiener1948].
Its control dual is the discrete-time **Riccati** equation [Anderson1990, Bertsekas2017]: for quadratic
cost `\sum_t (d_t^{\top}M d_t + a_t^{\top}R a_t)` under `d_{t+1}=A_d d_t + B a_t`,

$$P \;=\; A_d^{\top}P A_d \;-\; A_d^{\top}P B\,(R+B^{\top}P B)^{-1}B^{\top}P A_d \;+\; M,\qquad
K=(R+B^{\top}PB)^{-1}B^{\top}P A_d,$$

with optimal control `a = -K d` and value `V(d)=d^{\top}P d`. The two equations are **transpose-dual**
(`A Σ Aᵀ` vs `Aᵀ P A`), the classical estimation↔control duality.

**Consequence.** `A_d ↔ Σ` (Lyapunov, estimation) and `A_p ↔ P` (Riccati, control) are the dual pair;
`det(A_p)>0` (§4) **is** the LQR requirement `P ≻ 0`. The program's regularizers are precisely the LQG
preconditions: R15 affine action ⇒ linear plant; R16 smooth reward ⇒ quadratic cost; the svband ⇒
stabilizable contractive `A_d`; `det(A_p)>0` ⇒ `P≻0`. For state-dependent `A_d(d)` the regulator is
state-dependent-Riccati / iLQR [Cimen2008, Tassa2012], well-posed because the operator-smoothness
penalty keeps `A_d(\cdot)` slowly varying. A scalar reproduction of the duality:
[`../reproduction/lqr_riccati_duality.py`](../reproduction/lqr_riccati_duality.py).

---

### 8. The `Q`–policy coupling (why the policy is still learned)

Closing LQG offline fails because the innovation `Q` is **policy-coupled**: `Q` is the covariance of
the one-step residual over the *visited-state* distribution, which the policy generates. A fixed policy
narrows that distribution, degrading `Q`, the model, and hence the very cross-entropy being fit.
Therefore the behaviour learner is retained — not as the *controller* (the Riccati gives that closed
form) but as the **data-quality engine**, the on-policy generator of the distribution on which the
operator loss is meaningful [Hafner2020, Hafner2023]. This is the data-quality thesis of [Gunasekar2023]
in control form: representation quality is bounded by the quality of the visited data, and only an
adapting policy keeps that data informative. The behaviour objective is Dreamer λ-returns through the
model with an EMA target value net (deliberately not an off-policy `Q`-critic, which would orphan the
operator-field structure).

---

### 9. The critical entropy-exponent ratio

**Conjecture 1 (critical ratio). [conjectured; one-seed empirical support]**
The operator energy-retention ratio `r = |λ(A_d)|^2` has a *critical* value

$$r_\star \;=\; \tfrac15 \;=\; 1 - \tfrac45,$$

the complement of the Pareto/effective-rank fraction (`r_eff/d ≈ 0.8` regime). At `r_\star` the
controlled system undergoes a **sharp transition** rather than a gradual climb: the policy determinant
(`∝ e^{\mathrm{policy\ entropy}}`), the imagined return, and the realized return all break vertically
**together**. Empirically, an `A_d` initialized at `r=0.2` exhibited exactly this co-transition at
`≈220k` steps (policy entropy `−2.5→−0.3`, imagined return crossing zero, eval `−187→+540`) in one
seed; the Stein-consistency arm with no spectral initialization climbed gradually with no transition.
A one-shot initialization at `r_\star` reverts (the dynamics fit pulls `|λ|→1`); the reliable form holds
the ratio via a persistent descending/clamped spectral target. The detector and the reliability test
across seeds are in
[`../verification/verify_critical_transition.py`](../verification/verify_critical_transition.py). The
criticality framing (a control order parameter with a sharp threshold) is in the spirit of
self-organized criticality [Bak1987]; here the order parameter is the entropy exponent and the control
parameter the ratio `r`.

---

### Status summary

| § | result | status |
|---|---|---|
| 1 | `z_std=√⟨μ⟩`, `r_eff/d` const | proved (coefficient empirical) |
| 2 | Stein/Lyapunov closed form; normal diagonalization `q/(1−\|λ\|²)` | proved |
| 3 | covariance band ⟺ operator annulus (anti-freeze) | proved |
| 4 | `log det A` = entropy exponent; `det(A_p)>0` ⟺ `P≻0` finite | proved |
| 5 | dissipative `A_d` / conservative `A_p` asymmetry | empirical |
| 6 | operator cross-entropy = Stein's loss; fixed point `Σ̂=G`; unifies 3 terms | proved |
| 7 | LQG / Lyapunov–Riccati duality; regularizers = LQG preconditions | proved (control theory) |
| 8 | `Q`–policy coupling ⇒ retain online policy learning | argued |
| 9 | critical ratio `r⋆ = 1/5` | conjectured (1-seed support) |

*Rejected along the way (kept here to forestall rediscovery):* a golden-ratio fixed point for the
operator coupling, and the relations `g_p = 1+1/g_d` and `g_p = 1+\text{eff}/H` — all shown to be
coincidences of two independently band-pinned constants (numerical refutation in
[`../verification/verify_rejected_coincidences.py`](../verification/verify_rejected_coincidences.py)).
