# Nightly ML log — MBRL project

Mirror of the math project's nightly verification stream. One cycle per
night, recorded honestly. Cycle order: adjudicate → supervised experiment →
research note.

## 2026-06-14

**Headline: ran cycle-2 supervised experiment (run 12B — Φ-SVD shrinkage /
adaptive TSVD), pre-registered then adjudicated NOT SUPPORTED; the corrected
candidate B is closed.**

- Orient: `status.py` + `ledger_check.py` both clean (ledger-check PASS).
  Nothing to ADJUDICATE — gpu_spectral 6-arm was already done (2026-06-12);
  run 10 vae_ablation still has NO arms in results/runs (remains pending, no
  data). So the cycle fell to priority 2, supervised experiment.
- Picked the cycle-2 queue item "shrinkage in the Φ-SVD basis (adaptive TSVD
  / Rosasco spectral filtering — the corrected form of candidate B)".
  Web-searched first (Rosasco spectral-filtering family, MIT 9.520; DJ 1994
  universal threshold + MAD/0.6745). Implemented `spectral.svd_shrink_fit`
  (SVD of the RFF design Φ=USVᵀ; β=Uᵀy has iid noise; DJ soft-threshold;
  Tikhonov filter s/(s²+λ) for stable inversion). NOTE/decision: the
  penalty-whitening route (Φ·diag(w)^{-1/2}) was implemented first but
  REJECTED — it blows up numerically when poly weights → 0 (Pinv ~1e4), so
  kappa=0 failed to reproduce the ridge; switched to the raw-Φ-SVD form,
  which is stable at any conditioning and kappa=0 ≡ scalar Tikhonov ridge
  (pinned in test).
- PRE-REGISTERED criteria in the ledger and COMMITTED them (c1b9144) before
  running. Then ran the 20-cell harness (chunked, sha-scoped
  results/bridge/c1b9144/), adjudicated, committed results (ccc526f).
- Verdict: NOT SUPPORTED — 6/20 wins, mean −69.4%, worst −945%, fails all
  three bars. Discrepancy/insight surfaced: **kappa=0 (no shrinkage) was the
  validation pick in 20/20 cells** — the DJ threshold never helped, because
  smooth/resonant rewards aren't sparse in the Φ-SVD basis. The orthonormal
  frame DOES fix shrink_coefs' ringing (test: ×3000 → bounded), but sparsity
  (the property DJ needs) is absent. Per-target: smooth 0/10 (−140%),
  resonant 6/10 (+1.0%), resonant n=2048 5/5 (+10.8%) — plain spectral
  filtering only ties/edges the poly recipe in the data-rich resonant regime.
  4th orthonormal-frame instance (runs 4, 6/8, 12, 12B); corrected candidate
  B CLOSED.
- Env, recorded honestly: repo `.venv` is a broken macOS symlink on this
  Linux sandbox (as on 2026-06-12). This time I installed the EXACT locked
  versions (torch 2.5.1, numpy 2.2.6, gymnasium 1.3.0, + omegaconf/hydra/
  onnx/joblib at their pins) into the ephemeral sandbox — a faithful repro of
  the LOCAL CPU harness env (not the cloud image; the supervised harness is
  CPU-only by the execution rules). `make test` (fast) green before AND after
  the change: 333 passed.
- GIT HYGIENE ISSUE for the user: the mount denies `unlink` on `.git`, so
  `git commit` left stale lock files it couldn't clean up. Both commits DID
  land (ledger HEAD has the RESULTS block; `git diff HEAD` empty). But three
  stale locks remain — **on your Mac, run `rm -f .git/index.lock
  .git/HEAD.lock .git/refs/heads/main.lock`** (and the working `.git/index`
  is stale, so the first `git status` may show claims_ledger.md as modified
  until you `git reset` / it refreshes; the committed content is correct).

**Next-night pickup:** (1) the mlp-recipe anchor regression is STILL the top
science priority (improvement plan #1 — diff the recipe arm's effective
config vs original report §2: smoothing.sigma 1.5 vs 1.0, eval protocol,
probe count) before any spectral relaunch; (2) cycle-2 supervised queue now =
leverage-score feature sampling (Bach 2017) as the sole remaining candidate;
(3) run 10 vae_ablation still needs arms pulled before it can be adjudicated.

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
