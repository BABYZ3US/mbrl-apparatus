# Related work — the spectral reward stack in context

Compiled 2026-06-08, after the spectral_ladder RL runs came up working. Maps each
ingredient of the stack (sigma ladder x lambda-polynomial band weights x closed-form
refits, models/spectral.py + training/loop.py) to its lineage. Bottom line: every
ingredient has strong precedent; the COMBINATION — band-structured random features
with a time-scheduled per-band filter function, solved in closed form inside the
model-learning loop of MBRL — does not appear in the literature we found.

## 1. Multi-scale Fourier features (the sigma ladder)

- Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi,
  Barron, Ng (NeurIPS 2020), *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains*. https://arxiv.org/abs/2006.10739
  — Fourier feature mappings turn the MLP's NTK into a stationary kernel with
  tunable bandwidth; fixes spectral bias. NeRF's positional encoding = deterministic
  log-spaced frequency octaves, i.e. a fixed sigma ladder. Our ladder is the
  random-feature version with bandwidth parameterized over the transform.
- Rahimi & Recht (NeurIPS 2007), *Random Features for Large-Scale Kernel Machines*
  — the RFF foundation.

## 2. Frequency-band curricula (the lambda polynomial / band equalizer)

- Lin, Ma, Torralba, Lucey (ICCV 2021), *BARF: Bundle-Adjusting Neural Radiance
  Fields*. https://arxiv.org/abs/2104.06405
  — smooth mask over positional-encoding frequency bands, opened low->high during
  optimization: a dynamic low-pass filter as curriculum. Direct ancestor of the
  per-degree schedule shifts ("wide cuts first, sharp cuts later"). FA-BARF (2025)
  refines the annealing. https://arxiv.org/abs/2503.12086
- Spectral bias / F-principle literature (Rahaman et al. 2019; Xu et al.) — networks
  fit low frequencies first; admitting high bands too early overfits. The RL-run
  overfitting episode (2026-06-07, ledger "spectral scheduling rule") is this
  phenomenon in closed form: with no optimizer inertia, lambda ~ 0 IS the
  unregularized interpolator.

## 3. Band-weighted closed-form solves (the theta(|w|) ridge weights)

- Regularization by spectral filtering, kernel-methods tradition (Lo Gerfo, Rosasco,
  Odone, De Vito, Verri, *Spectral Algorithms for Supervised Learning*; MIT 9.520
  notes https://www.mit.edu/~9.520/spring09/Classes/class07_spectral.pdf;
  https://en.wikipedia.org/wiki/Regularization_by_spectral_filtering)
  — theta(|w|) is a *filter function*; the H^2 quartic is the Sobolev penalty
  diagonalized in the Fourier basis. The ledger's Wiener-filter identity (Tier 1)
  lives in this family.

## 4. Closest RL relatives

- Li & Pathak (NeurIPS 2021), *Functional Regularization for Reinforcement Learning
  via Learned Fourier Features*. https://arxiv.org/abs/2112.03257
  — NTK analysis: tuning the Fourier basis's initial variance IS functional
  regularization controlling per-band over/underfit of value functions. Our stack
  makes that control explicit, multi-band, and time-varying.
- Konidaris, Osentoski, Thomas (AAAI 2011), *Value Function Approximation in RL
  using the Fourier Basis* — the classic.
- What we did NOT find: the band-weighted closed-form solve as the *reward head
  inside the MBRL model-learning loop*, refit from a rolling cache with a scheduled
  filter function. That, plus the no-inertia scheduling rule, is the apparent novelty.

## 5. The physics analogy ("spectral holography / resonance imaging")

Genuinely cognate at the level of band-weighted Fourier-domain inverse problems:

- Compressed-sensing MRI — reconstruction from k-space samples via regularized
  inverse problems, lambda trading data consistency vs band penalties.
  (Lustig et al. tradition; review: mriquestions.com CS review.)
- Fourier ptychography / synthetic-aperture holography — high-resolution images
  stitched from spectrum bands captured at different illumination angles; a
  reference-free extension of synthetic-aperture holography.
  https://en.wikipedia.org/wiki/Fourier_ptychography ;
  SAVI: https://www.science.org/doi/10.1126/sciadv.1602564
  — structurally parallel to the sigma ladder covering frequency annuli and the
  equalizer dosing them.

**Caveat (keep when citing):** the kinship is mathematical — band-weighted inverse
problems in a Fourier dual — not evidence that the reward model performs physical
imaging. State the analogy at the math level only.
