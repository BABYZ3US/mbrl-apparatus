# Ablation-1 (`abl1`) — adjudication block

*Ledger-ready. Fold this verbatim into `claims_ledger.md` under the RL-loop results once the
other uncommitted ledger edits are settled. Full analysis: [`analysis_ablation1.md`](./analysis_ablation1.md);
registered follow-up: [`abl1_followup_a6a7_prereg.md`](./abl1_followup_a6a7_prereg.md).*

---

**Ablation-1 (2026-06-16, `run_ablation1.sh`, adjudicated from W&B
`pandelak-boston-college/mbrl-curvature`): operator-spectrum / Lyapunov levers on the
cf22 canonical baseline — NO lever cleanly beats A0 at this power; the campaign is
underpowered and bimodal; ONE fully-powered NEGATIVE mechanism result and ONE suggestive
positive worth a registered follow-up.** Arms A0–A19 (A20excite never launched),
HalfCheetah-v5, target 500K env steps, primary metric `eval/return_det` (3-episode
deterministic mean; final = mean of last-3 evals). Pre-registered campaign question
(`run_ablation1.sh` header): *which lever moves `eval/return_det` vs A0 → what to explore
next.* Verdict, with no softening:

- **(i) Cannot adjudicate any lever vs A0 at n=2 with fire/collapse bimodality.** Every
  arm with a positive mean (A1 modelfit +782, A4 dblvalue +524, A5 latent32 +545, A19
  riddown +886) is a *single* seed that fired to a sustained 1100–3400 return paired with
  one flat seed — seed SDs ±570–995. The baseline A0 *itself* has a seed that peaks to 312
  before reverting, so single-seed fires are not separable from the apparatus's intrinsic
  stochastic firing. Per the matched-budget read, at 300K env steps **every** arm is still
  negative; firing is a late-training (>300K) event. The campaign as designed ranks nothing.

- **(ii) FULLY-POWERED NEGATIVE — the radius-pinning sub-program (A9–A13, A18, A19) is
  mechanistically inert.** `op/radius_d` converges to **≈1.0 in all ~25 seeds** regardless
  of penalty strength (w_svband=15), initialization (init_shift 0.447 / 0.707 / 0.99), or
  anneal schedule. A12 (w_svband=15, target |λ|=0.447, "sit at the critical point from
  t=0") never holds below 0.83 and ends at 1.00–1.02; A10/A19 never dip below 0.98; A18
  (init 0.707 + w=15) drifts to 0.955; A9 (init 0.707) shoots *above* 1.0 immediately. The
  band/svband penalty cannot move the dynamics operator's dominant singular value off the
  unit circle — so the "control the energy/entropy ratio by pinning |λ|²" mechanism does
  not actuate the quantity it targets, and those arms' return numbers are NOT tests of the
  ratio hypothesis. The A12/A13 "critical separatrix → convergent basin" hypothesis is NOT
  SUPPORTED: seeds snap to a peak then revert hard (A12 drawdown up to 788; A13 nudge
  0.20→0.21 over 8 seeds bought ≈0/8 held, 1/8 transient). A19's own design note
  ("the data says |λ| WANTS the edge") is confirmed. **CLOSED**: radius-pinning as an
  actuator of |λ| is dead; reviving the ratio idea needs a different actuator (constrain the
  parametrization, not add a stronger band penalty), pre-registered anew.

- **(iii) SUGGESTIVE POSITIVE (not yet a claim) — the Stein consistency term stabilizes
  firing.** A7lyap (svband + Stein consistency, d=16) is the ONLY arm where BOTH seeds
  fired and held (+486 ± 172 final; peaks 597 / 996; full 500K), vs its own control A6
  antifreeze (svband only) at 1/2. A6→A7 differ by exactly `model.dual_latent.lyap_weight`
  (0.0 → 0.3) — the one clean within-campaign contrast. At n=2 this is hypothesis-tier
  only; promoted to the registered follow-up (A6 vs A7, ≥8 seeds, 500K) before any claim.

- **(iv) DEAD ARMS (record so they are not re-tried blind):** env-dim² latent failed
  computationally — A15 (d=289) logged 0 evals, A14 (d=256) logged 1 (the O(d³) operator
  SVD is infeasible at full resolution); the amortized phased variants A16/A17 (d=289 with
  `struct_every`) ran to 500K but never fired (peaks 135 / 96). The env-dim² latent
  direction is closed. A20excite was never launched.

**Caveats on record:** (a) absolute returns sit on the known weak anchor — the mlp-recipe
anchor regression (gpu_spectral, 2026-06-12) is still unresolved, so abl1 numbers are
A6-vs-A7 *relative* reads, never absolute HalfCheetah results. (b) Run-length confound:
arms reached 155K–500K (A11/A18 ≈155–160K, A10 ≈250K, A9 ≈310K, A8 ≈345K vs A6/A7/A16/A17/A19
= 500K), and paired-identical stop times suggest per-wave wall-clock/crash cutoffs, not
convergence — the short arms (A8–A11, A18) may have been killed before any firing window
(`results/gridlogs/` is empty, cloud run, so the termination cause is not in the metrics).
(c) `eval/return_det` averages only 3 deterministic episodes; the bimodality is nonetheless
a real training-dynamics property (corroborated by `imagine/return_mean`→∞ collapse flags
and the `op/radius_d` trajectories), not pure eval noise.

**Next:** run the registered A6-vs-A7 follow-up; retire radius-pinning and env-dim² latent;
fix the eval-variance/run-length protocol (≥5 seeds, matched budget all arms reach, ≥10
eval episodes) before any further abl1-style adjudication. A20excite (phase-kick to break
the bimodality) is only worth running *after* the protocol fix, since the bimodality it
targets is exactly this campaign's core measurement problem.
