# MBRL project — setup & nightly-cycle memory (auto-memory)

Durable facts for the nightly research cycle (mirrors the math project's nightly
verification stream). Update this only when a RULE, ENV quirk, or CYCLE-STATE changes —
not every night. Created 2026-06-16.

## Where things live
- Repo: `…/math/mbrl` (git; the parent `…/math` holds the math-side notes).
- Publishability contract: `docs/claims_ledger.md` — tiers, pre-registered runs + RESULTS,
  and the research cycle queue at the BOTTOM. `docs/original_findings_report.md` = original
  empirical record (thermodynamic-consistency / clamp +41 vs −40; λ=0.5 on HalfCheetah §7).
- Plan: `docs/improvement_plan.md`. Nightly log: `docs/nightly_ml_log.md` (newest entry on
  top, under the 5-line header).
- Orient EVERY night: `python scripts/status.py` + `python scripts/ledger_check.py`
  (the latter must print `PASS`).

## Execution rules (HARD — violating these has cost batches)
- Training/inference is CLOUD-ONLY via the sealed image (`docs/remote_execution.md`); Colab
  retired 2026-06-08. LOCALLY only: `make test` / `test-all`, the supervised closed-form
  harness scripts (CPU, seconds/cell — `bridge_experiment.py` / `orf_shrinkage_test.py` /
  `svd_shrink_test.py` / `leverage_sample_test.py` style), and analysis over results/runs
  JSONL + W&B artifacts. **NEVER launch RL training locally.**
- One change per experiment; matched budgets; PRE-REGISTER criteria in the ledger (and
  COMMIT them) BEFORE any results exist. Default supervised bar: majority of 20 cells,
  mean rel test-MSE > +2%, worst cell > −20% vs the champion arm.
- `make test` (fast set) before AND after any CODE change; never commit red. Docs-only
  nights do not trigger it (precedent: 2026-06-12).
- Spectral path: no zero-touching λ schedules (smooth floored decay only, floor 1e-5);
  latent cap 1× obs_dim; encoder needs a grounding loss (encoder_aux or VAE). Per-component
  statistics need an orthogonalized/incremental measurement frame (runs 4, 12B).
- Commit each completed step as `Julian Pandelakis <julian@zpandas.com>`; do NOT push.

## Sandbox env quirks (recurring — do not re-derive each night)
- The repo `.venv` is a broken macOS symlink on the Linux sandbox. For a CODE night, install
  the EXACT locked versions into the ephemeral sandbox (torch 2.5.1, numpy 2.2.6,
  gymnasium 1.3.0, omegaconf/hydra/joblib at their pins, + onnx>=1.16 for the policy-export
  tests) — a faithful repro of the LOCAL CPU harness env. Never `make sync` / pip-install
  ad hoc into the user's venv. Docs-only nights need none of this.
- Git on this mount DENIES `unlink` on `.git`, so stale `.git/index.lock`, `*.lock`, and
  `tmp_obj_*` files accumulate. Commit WITHOUT touching the real index:
  `GIT_INDEX_FILE=$(mktemp); git read-tree HEAD; git add <only your files>;
  TREE=$(git write-tree); git commit-tree $TREE -p HEAD -m msg` → then advance
  `refs/heads/main` (`git update-ref`, or a direct loose-ref write if `main.lock` is stuck).
  Cleanup command for the user's Mac: `rm -f .git/index.lock .git/HEAD.lock
  .git/refs/heads/main.lock .git/objects/maintenance.lock && rm -f .git/objects/*/tmp_obj_*`.

## Cycle state (as of 2026-06-16)
- Cycle 1 (approx-theory): ORF (run 12) sub-threshold; DJ shrinkage dropped pre-run.
- Cycle 2: Φ-SVD shrinkage (run 12B) NOT SUPPORTED; leverage-score sampling (run 13) NOT
  SUPPORTED ⇒ cycle-2 supervised queue EXHAUSTED.
- Meta-result (runs 6/8/12/12B/13): at matched M=512 the linear multi-scale frame +
  validated poly recipe is sufficient; smarter feature SELECTION does not beat it. New
  candidates must change a DIFFERENT axis.
- Cycle 3 queue (refilled 2026-06-16; see ledger bottom): C3-1 reduced-budget feature-
  efficiency M-sweep (QMC + Gaussian-quadrature features) [TOP]; C3-2 second-moment
  eff_rank/CV penalty [med, structural translation of Appendix_C §C.3]; C3-3 reverse-water-
  filling band allocation [low — run-4 duplication risk]. Stein/LogDet loss is PARKED on the
  RL/dynamics-calibration axis, not the supervised queue.
- Standing pending (NOT supervised; need cloud RL): mlp-recipe anchor regression (TOP
  science priority — improvement plan #1); run 10 vae_ablation (no arms pulled yet);
  gaussian-dynamics MuJoCo requalification (R15 ruled decorative at Pendulum; champion was
  the worst arm in gpu_spectral).

## Math-side bridge discipline
- Skim NEW parent-folder notes for STRUCTURAL candidates only; translate at the
  optimization/structural-math level. NEVER import RH / number-theory claims — the RL↔Nyman–
  Beurling correspondence is explicitly NON-transferring (Appendix B.5d; the spine test
  confirmed it empirically). The Weil-positivity bridge already has no remaining closed-form
  candidate (run 7); it lives in the RL loop or retires.
