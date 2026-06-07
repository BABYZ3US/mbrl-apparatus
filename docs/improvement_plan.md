# Improvement plan — 2026-06-08

Ranked by expected value per unit work. Status tags: [ready] = can start now,
[blocked-on-RL] = needs the colab_spectral readout first.

## A. Science (the model itself)

1. **[blocked-on-RL] Adjudicate the 5-arm spectral RL validation.** Everything
   below re-ranks on this readout. Decision rules are pre-registered in the
   ledger (runs 3/5). If spec-auto ≥ spec-ladder > spec-single with the MLP
   anchor band reproduced, the supervised +48.3% becomes an RL claim.

2. **[SHIPPED 2026-06-08] Recalibration-on-drift for sigma_w=auto.** sigma* is measured on
   early-policy cache data and frozen; the reward landscape's frequency content
   shifts as the policy improves. Cheap fix: re-run the probe every N refits,
   log sigma*_t, and recalibrate if it moves > x2 (new lineage not needed — the
   basis can be rebuilt at a refit boundary since c re-anchors anyway). The
   spectral/sigma_star time series from the first RL runs will say whether
   drift is real before we build this.

3. **[SHIPPED 2026-06-08] Weight decay on learned log_s.** The learned-sigma gradient has an
   incentive to widen bandwidths toward cache-fitting (same failure family as
   the zero-touching schedule). A small L2 on log_s toward its init (elastic
   anchor) bounds the drift; one config knob + 5-line change + one test.

4. **[SHIPPED 2026-06-08] Per-band SNR as a logged diagnostic everywhere.** weights_mode=snr
   lost as a *penalty* but the band-SNR profile is measurement-grade. Log it on
   every refit regardless of mode (~free: one snr_band_weights call per refit
   on head 0). Gives: drift detection for (2), an overfit early-warning (live
   SNR collapse), and the per-env sigma* table for the writeup.

5. **[SHIPPED 2026-06-08 — result: NOT SUPPORTED, Spearman −0.12 at fixed bandwidth (ledger run 2b)] Deconfound angle-vs-bandwidth.** Anisotropic W
   draws (stretch along/across data curvature directions) move the diag-vs-Gram
   angle at fixed sigma_w. Settles whether "maximize the transversality angle"
   is causal or a bandwidth proxy. Supervised-only, ~1 day.

6. **Gaussian-dynamics value test.** NLL beats MSE only if the stochasticity is
   used: compare affine vs gaussian on Pendulum 3 seeds (imagined-return
   variance, final return, dyn calibration: predicted sigma vs realized error).
   Also the honest R15 ablation: a full-MLP mean arm to test whether the
   curvature floor actually binds (one-line class; run as its own arm).

7. **The clamp question is still open.** Bridge runs showed pointwise
   positivity is NOT the clamp's operative property in closed form. The
   remaining candidate: sign-coherence as a CONE constraint (project c onto
   {c : per-sample curvature products >= 0}) — nonlinear, but solvable as one
   QP per refit at M=512. This is the last cheap shot at the ledger's bridge
   prediction before it needs the RL loop.

## B. Apparatus (usability / reliability)

8. **[SHIPPED 2026-06-08] Quarantine integration tests.** The suite is creeping past 30s
   (smoke + trainer tests dominate). Mark them `@pytest.mark.slow`; default
   `make test` runs the fast set (<5s), CI/pre-push runs all. Keeps the
   verify-before-anything-else habit cheap.

9. **[SHIPPED 2026-06-08 — configs/presets.yaml] Preset registry dedup.** parallel_runs.py PRESETS has grown
   organically (3 blocks, interleaved constants). Move to
   `configs/presets.yaml` consumed by parallel_runs — presets become data,
   diffable, and the notebook can list them.

10. **[SHIPPED 2026-06-08 — scripts/status.py, make status] One-command experiment status.** `scripts/status.py`: read
    results/runs JSONL + W&B (auto fallback), print per-group step counts,
    last checkpoint, and whether each ledger-pending validation has its arms
    running. The "what is actually running and what's missing" question
    currently takes dashboard + W&B + memory.

11. **[SHIPPED 2026-06-08 — Trainer warns on known-bad combos] Single source of truth for the recipe rules.** The spectral rules
    (no zero-touching schedules, cap 1x, smooth floor) are enforced by
    convention in presets/configs but nothing stops
    `spectral.enabled=true penalty.schedule.kind=step` from the CLI. Add a
    config validator in Trainer.__init__ that *warns loudly* (not errors —
    ablations must stay possible) when a known-bad combination is composed.

12. **[SHIPPED 2026-06-08 — fail-fast install cell] Colab cell-2 install check.** The MuJoCo failure mode (pip line silently
    truncated by `| tail -1`) cost a 9-run batch. Replace with an assert-import
    cell that fails fast and says what to do.

## C. Writeup readiness

13. **[SHIPPED 2026-06-08 — results/{bridge,bench}/<sha>/] Results provenance.** bridge_experiment/recipe results overwrite in
    place (rm + rerun). Move to results/bridge/<git-sha>/ so every ledger
    number stays reproducible from its commit. Small change in two scripts.

14. **[SHIPPED 2026-06-08 — scripts/ledger_check.py, make ledger-check] Auto-generated ledger table.** The per-run numbers in claims_ledger.md
    are hand-copied today; a `make ledger-check` that regenerates the bridge
    tables from results JSON and diffs against the prose would catch rot.
