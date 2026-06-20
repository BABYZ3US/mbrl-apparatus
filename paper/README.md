# paper/ — publication draft materials

Draft materials for **Spectral Operator Regularization and the LQG Structure of Latent Model-Based
Control** (the operator-field latent-MBRL program).

## Layout

```
paper/
├── 00_draft.md            # the paper draft: abstract, narrative, results, reproduction pointers
├── references.bib         # all citations (BibTeX)
├── theory/
│   └── derivations.md     # ALL theoretical derivations, one document, with citations + status tags
├── reproduction/          # CAS (sympy) — symbolic reproduction of every [proved] result
│   ├── lyapunov_covariance.py      # §2  Stein equation; Σ_i = q/(1-|λ|²)
│   ├── operator_cross_entropy.py   # §6  cross-entropy = Stein's loss; fixed point Σ̂=G
│   ├── stein_loss_separation.py    # §6  separation over the generalized spectrum; rational reduction
│   ├── padic_exact_loss.py         # §6  2-adic (p=2) exact loss in fixed-width binary (conservative case)
│   ├── padic_lift_solve.py         # §6  Dixon 2-adic lift: exact rational solve in O(n^3)
│   ├── padic_gpu.py                # §6  GPU-vectorized 2-adic lift (torch int64 = Z/2^64), batched
│   ├── band_to_annulus.py          # §3  covariance band ⟺ eigenvalue annulus
│   ├── lqr_riccati_duality.py      # §7  DARE; Lyapunov–Riccati transpose-duality
│   └── entropy_exponent.py         # §4  log det A = Σ log|λ|; det>0
└── verification/          # numerical checks against experiment (numpy; some take a metrics path)
    ├── verify_zstd_identity.py            # §1  z_std = √⟨μ⟩
    ├── verify_effrank_linear.py           # §1  eff_rank/d constant
    ├── verify_cross_entropy_fixedpoint.py # §6  minimizer of L is G
    ├── verify_critical_transition.py      # §9  the co-transition detector (path optional)
    └── verify_rejected_coincidences.py    # §9  golden-ratio / g_p laws are coincidences
```

## Conventions

- Every result in `theory/derivations.md` is tagged **[proved]**, **[empirical]**, or
  **[conjectured]** (claims-ledger discipline). Do not promote a tag without the corresponding script.
- `reproduction/` = symbolic (CAS) proofs of the analytical claims. `verification/` = numerical checks,
  including against the recorded experimental numbers.

## Run everything

```bash
cd ..                                  # mbrl/ project root (uses .venv with sympy + numpy)
for f in paper/reproduction/*.py; do echo "== $f =="; .venv/bin/python "$f"; done
for f in paper/verification/*.py; do echo "== $f =="; .venv/bin/python "$f"; done
```

All `reproduction/` scripts and the self-contained `verification/` scripts assert their identities and
print `PASS`. `verify_critical_transition.py` accepts a `metrics.jsonl` path (otherwise it prints the
recorded A10fifth-s0 example).
