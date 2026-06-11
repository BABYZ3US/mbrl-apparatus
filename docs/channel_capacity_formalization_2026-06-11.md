# Latent-as-Channel: an information-theoretic formalization

*Working draft, 2026-06-11. Goal: formalize spectral reward + VAE loss + policy loss
+ affine dynamics as one channel-coding problem, per the PM's hypothesis that the
latent space is a noisy channel (encoder = transmitter, decoder = receiver).*

Status tags: `[SHANNON]` standard info theory; `[PROJECT]` proved/measured in this repo
(R-number = founding doc); `[KNOWN]` established in the ML literature; `[GAP]` the
conjecture this formalization is meant to make precise and test.

---

## 0. The gap this closes

The repo proves two legs and never joins them:
- **Leg A (kernel):** the H² reward penalty ⇒ Matérn RKHS, covering numbers at s=d/2 (R3/R7);
  realized as RFF-ridge reward regression over the latent.
- **Leg B (filter):** the same penalty *is* a Wiener / rate-distortion low-pass filter,
  cutoff ω_c~λ^{−1/4}, water-filling at SNR=1 (R11/R14); σ*≈0.207 measured.

What is missing: the kernel is attached to the **reward surface**, not the **encoder-induced
latent metric** (R16 found that metric flat on a synthetic decoder; the VAE arm Run 10 is unrun).
The channel view below is the bridge: it makes the latent itself the object whose capacity is
managed, and shows leg B's water-filling *is* channel-capacity allocation.

---

## 1. Three coupled channels sharing the latent z

Model the apparatus as a cascade of noisy channels, all carrying the latent `z ∈ ℝ^k`.

### 1.1 Representation channel  x —[enc]→ z —[dec]→ x̂   (the PM's channel)
- **Transmitter (encoder):** `q_φ(z|x) = 𝒩(μ_φ(x), diag σ²_φ(x))`  [encoder.py:115] — a stochastic
  (noisy) input; the noise is the encoder variance σ²_φ.
- **Receiver (decoder):** `p_θ(x|z)` Gaussian ⇒ distortion `D_rep = 𝔼‖dec(z) − x‖²`  [encoder.py:117].
- **Prior / reference:** `p(z) = 𝒩(0, I)`  [encoder.py:118].
- **Rate** `[SHANNON]`: `R_rep = 𝔼_x KL(q_φ(z|x)‖p(z)) = I(x;z) + KL(q̄(z)‖p(z)) ≥ I(x;z)`,
  where `q̄(z)=∫q_φ(z|x)p(x)dx` is the aggregate posterior. So the VAE KL term is an **upper
  bound on the channel's mutual information** I(x;z) — the bits the latent carries about x.
- **β-VAE loss** `[loop.py:413]`  `L_VAE = D_rep + β·R_rep` is exactly the **rate–distortion
  Lagrangian** `[KNOWN: Alemi 2017 deep VIB; the RD view of β-VAE]`; β is the RD slope.

### 1.2 Reward read-out channel  z —[Wiener/RFF head]→ r̂
- `R(z,a) = Σ_j c_j √(2/M) cos(w_j·[z,a] + b_j)`  [spectral.py:421] — RFF features over the
  **detached** joint coord `[z,a,τ]`, targets in symlog space.
- Closed form `c = (ΦᵀΦ + diag θ + εI)⁻¹ Φᵀ y`  [spectral.py:451]; band weights
  `θ_j = (N/M)/SNR_j`  [spectral.py:277] ⇒ per-feature Wiener gain `SNR_j/(1+SNR_j)`, cutoff SNR=1.
- Read this as the **matched/Wiener decoder** that extracts the reward signal from the noisy z.

### 1.3 Dynamics channel  (z,a) —[Gaussian affine]→ z′
- `p(z'|z,a) = 𝒩(μ = z + f(z) + G(z)a,  diag σ²(z))`  [dynamics.py:47–58].
  The **mean is affine in a** (∂²/∂a²=0, R15) — a *linear* channel input; σ²(z) is state-only
  channel noise. NLL `= 0.5·𝔼[(z′−μ)²e^{−lv} + lv]` is the channel's log-loss.
- Imagination = **repeated use of this channel**; horizon-variance amplification (R15) = noise
  accumulating over channel uses, bandwidth-limited by the curvature penalty λ.

---

## 2. Capacity, and the one deep bridge: water-filling

### 2.1 Correct the objective: it's a *bottleneck*, not "maximize capacity"
The VAE loss **minimizes** the rate R_rep ≈ I(x;z) at a target distortion — it *compresses* x
into z. Maximizing I(x;z) outright is anti-compression (β→0, overfit). The right object for
**control** is the **Information Bottleneck** `[KNOWN: Tishby]`: make z a *minimal sufficient
statistic* of x for the task,
```
        min_φ   I(x;z)  −  γ · I(z ; r, z′)
                └ rate ┘     └ task-relevant capacity ┘
```
— minimize the nuisance rate the channel spends on x, **maximize the task-relevant capacity**
I(z; reward, next-state). The PM's instinct ("VAE loss is the channel knob") is right; the
refinement is the *direction*: the channel should carry the reward/dynamics bits, not the pixels.

This reframes three project findings as IB phenomena `[PROJECT]`:
- **encoder collapse** (z→const): rate R=0, task-info I(z;r)=0 — the degenerate IB corner; the
  spectral-aux fix forces I(z;r)>0 [claims_ledger encoder-collapse rule].
- **1×-latent-cap rule**: a hard cap on nuisance rate (wide latents over-resolve x) [ledger].
- **β=1e-3 + encoder_aux** (Run 10): an explicit point on the RD frontier.

### 2.2 The bridge: the H² Wiener penalty IS capacity water-filling  ⟵ the centerpiece
`[SHANNON]` Capacity of **parallel Gaussian sub-channels** with noises N_j under a power budget P:
```
   C = max_{P_j}  Σ_j ½ log(1 + P_j / N_j)   s.t. Σ_j P_j ≤ P
   ⇒  water-filling:  P_j = (ν − N_j)_+ .
```
`[PROJECT]` The spectral reward decomposes the latent into RFF frequency sub-bands indexed by
`|w_j|`, each with measured noise → SNR_j, and allocates penalty weights `θ_j = (N/M)/SNR_j`
with cutoff at SNR=1 [spectral.py:277, snr_band_weights]. **This is the same convex program.**
The optimal H²/Wiener penalty allocation across RFF bands *is* water-filling over the latent
channel's frequency decomposition; the SNR=1 crossing σ*≈0.207 is the **channel's noise floor**
in latent-frequency space, and ω_c~λ^{−1/4} (R11) is the **bandwidth** the penalty admits.

So the reward head doesn't just regress — it performs **capacity-achieving decoding** over the
latent channel viewed as parallel Gaussian sub-channels. Leg A (kernel) and leg B (filter) are
the **same object** seen through the channel: the Matérn RKHS is the channel's signal space, the
Wiener weights are its capacity-achieving power allocation.

---

## 3. The assembled objective, re-read as channel coding

Grounded in `loop.py:636–651`, the model loss is
```
 L_model =  D_dyn(NLL)                          ← dynamics-channel log-loss
          + D_reward (MLP rew_loss | spectral aux MSE)   ← reward-channel distortion (the true receiver)
          + λ_t · H²(reward)                    ← Wiener water-filling on the reward read-out (≡0 explicit in
          + β · KL(q_φ‖𝒩(0,I))                       spectral mode; baked into the ridge weights θ)
          + recon                               ← representation distortion (proxy; grounds z, prevents collapse)
```
with `λ_t = dg_gate · λ(step)`. Channel reading:
- `β·KL` = **rate** of the representation channel (bits x→z).
- `recon` = a *proxy* distortion; the **true** receivers are the reward head + dynamics model, so
  the real distortion is the **task** distortion `D_reward + D_dyn`. Reconstruction only grounds z
  (this is exactly why champion needs `encoder_aux`).
- `λ·H²` = the **rate-allocation** (water-filling) on the reward read-out channel.
- Behaviour loss (Dreamer λ-returns, `returns.py:16`) = the **controller** acting on the decoded
  signal; `R^λ_t = r_t + γ[(1−λ_ret)v_{t+1} + λ_ret R^λ_{t+1}]`.

**Why the penalty was moot on champion (this session's result), in channel terms:** the spectral
reward already self-regularizes (the ridge weights θ *are* the water-filling), so an *additional*
external λ on top is redundant — the channel is already decoded at capacity. The penalty earns
its keep only where the receiver is *not* self-regularizing (the plain MLP reward). That is the
channel-level statement of "dose is moot for champion."

---

## 4. Curved latents and the GFT (the arXiv 2605.00403 connection)

RFF is the **flat** (Euclidean) Fourier basis. If the encoder-induced metric g on the latent is
non-flat (R16 found it flat on a synthetic decoder — but that is exactly the thing to re-test for
pixels / harder tasks), the natural frequency basis of the channel is the **Laplace–Beltrami
eigenbasis** — precisely the Generalized Fourier Transform that 2605.00403 constructs (𝒰 unitary,
diagonalizes Δ_g; MASA of Killing-symmetry operators resolves spectral degeneracies). Then:
- reward kernel = **heat/Matérn kernel of Δ_g** on the latent manifold (the curved-space R7);
- H² penalty = the **biharmonic Δ_g² seminorm** (the curved-space R3/R13);
- water-filling runs over the **Δ_g spectrum** instead of |w_j|.

This makes "latent-space kernel" precise: the kernel is the heat kernel of the encoder-induced
Laplace–Beltrami operator, the reward lives in its RKHS, and the paper supplies the unitary
spectral transform that flat RFF approximates. `[GAP]` — contingent on the latent metric being
non-flat, which R16 has not yet confirmed.

---

## 5. Ledger: proved / known / gap

| Claim | Status |
|---|---|
| water-filling = capacity of parallel Gaussian channels | `[SHANNON]` |
| β-VAE loss = rate–distortion Lagrangian; KL ≥ I(x;z) | `[KNOWN]` (Alemi 2017; Hoffman/Alemi rate decomposition) |
| H² penalty ⇔ Matérn RKHS (s=d/2 covering numbers) | `[PROJECT]` R3/R7 |
| H² penalty ⇔ Wiener low-pass, ω_c~λ^{−1/4}, SNR=1 | `[PROJECT]` R11/R14; σ*≈0.207 measured |
| **spectral θ_j=(N/M)/SNR_j allocation = water-filling over RFF sub-channels** | **`[GAP→]` the new bridge — clean to prove, not yet written** |
| latent metric is the capacity-governing channel for control | `[GAP]` R16 found it flat; Run 10 (VAE) unrun |
| IB objective min I(x;z) − γI(z;r,z′) is the training target | `[GAP]` current VAE β·KL is only a proxy for the rate term |

---

## 6. Testable next steps (cheap, mostly already instrumented)

1. **Log the rate** `R̂ = 𝔼 KL(q_φ‖𝒩(0,I))` and a **task-info proxy** `Î(z;r) ≈ H(r) − ½log(2πe·MSE_reward)`
   per run; plot the (rate, task-distortion) point. Champion / spectral should sit on a frontier.
2. **Run the β-sweep (Run 10)** — it *is* the RD-frontier trace. The IB prediction: task return is
   flat across β until the rate drops below the task's minimal sufficient bits, then collapses.
3. **Prove the water-filling bridge (§2.2)** formally: show the SNR-band ridge `θ_j=(N/M)/SNR_j`
   minimizing penalized fit error is the Lagrange-dual of the parallel-Gaussian capacity program.
   This converts leg B from "Wiener analogy" to "capacity allocation" — a publishable identity.
4. **Curved-latent probe:** estimate the encoder Laplacian spectrum (graph-Laplacian on a z minibatch);
   if non-flat, swap RFF → Laplacian-eigenfeatures (a discrete GFT) and re-measure SNR bands.
