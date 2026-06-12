# Nightly ML log — MBRL project

Mirror of the math project's nightly verification stream. One cycle per
night, recorded honestly. Cycle order: adjudicate → supervised experiment →
research note.

## 2026-06-12

**Headline: gpu_spectral 6-arm RL validation adjudicated — NOT SUPPORTED,
and the batch is apparatus-confounded (the mlp-recipe anchor failed).**

- All 18 runs (6 arms × 3 seeds, HalfCheetah-v5) reached 200K env steps.
  Note: `status.py` shows model-update steps (40K) and stale idle times;
  `env_steps` in the JSONL is the ground truth. spec-auto-s2's jsonl is
  resume-reordered (last line 139K, max 200K) — worth a status.py fix.
- Pre-registered rule (improvement plan #1): spec-auto ≥ spec-ladder >
  spec-single with mlp-recipe reproducing +98 ± 23. Verdict: (i) FAILS,
  winrate 4/9; (ii) point estimate only, 6/9; (iii) anchor FAILED — −188.9
  ± 90.8, i.e. the original BASELINE band (−165 ± 41). Full RESULTS block
  appended to claims_ledger.md; supervised +48.3% does NOT become an RL
  claim.
- Discrepancies surfaced: (1) anchor regression despite the λ schedule
  verifiably executing and DreamSmooth on — candidate suspects recorded
  (smoothing σ 1.5 vs original 1.0, eval protocol, Hutchinson probe count);
  (2) champion (= spec-auto + gaussian dynamics) was the WORST arm (7–9/9
  seedwise losses to every other arm) — run 9's gaussian "mechanism without
  payoff" may be a return TAX at MuJoCo scale; requalify before it stays in
  the champion config; (3) encoder_aux fix held up: z_std 0.50–0.93 across
  all spectral arms, no collapse.
- Run 10 (vae_ablation) still has NO arms in results/runs — remains pending.
- Session constraint, recorded honestly: `make test` could not run in this
  sandbox (repo .venv is macOS-native, sandbox is Linux; `make sync` would
  have clobbered the user's venv — declined). Tonight's changes are
  docs-only (ledger, improvement plan, this log), so the
  test-before-and-after-code-changes rule is not violated.

**Next-night pickup:** root-cause the mlp-recipe anchor regression (top
priority per the updated improvement plan — start by diffing the recipe
arm's effective config against original report §2: smoothing.sigma,
probe count, eval episodes). The cycle-2 supervised queue (leverage-score
sampling; Φ-SVD shrinkage) stays queued behind it.
