#!/usr/bin/env bash
# ABLATION 1 (PM 2026-06-15): each analysis-derived improvement added INDIVIDUALLY and in
# ISOLATION to the cf22 canonical baseline, 2 seeds each. Theory-respecting set — NO model-free
# SAC import (which would orphan the operator-field/curvature program); only the tractable
# ingredients that stay in-framework + the in-framework levers the correlation analysis flagged.
# det-eval (eval/return_det = deterministic MEAN policy return, the benchmark convention) is
# logged on ALL arms — it recalibrates every number and is a measurement, not its own arm.
#   A0 baseline   — cf22 canonical (control): band + variance bound + leaky gate + ratchet
#                   horizon + near-zero init, curvature lambda pinned at its 1e-4 floor.
#   A1 modelfit   — model.hidden 256->384 (more capacity; targets imagine/return_mean, the +0.66
#                   eval driver).
#   A2 pconsist   — dual_latent.p_consistency_weight 1->3 (the dual/p_consistency +0.53 driver).
#   A3 autoalpha  — optim.auto_alpha.enabled: SAC auto-tuned entropy temperature (tractable
#                   ingredient; alpha=exp(log_alpha) tuned to a target entropy).
#   A4 dblvalue   — optim.clipped_double_value: TD3 twin-value MIN bootstrap (tractable ingredient).
# 5 arms x 2 seeds = 10 @ 500k. Fresh 'abl1-' prefix. Verdict: which lever moves eval/return_det
# vs A0 -> what to explore next (incl. folding in the heat-flow annealing as a later arm).
set -uo pipefail
cd "$(dirname "$0")/.."
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"; [ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((5 * NGPU))}"; STEPS="${STEPS:-500000}"; SEEDS="${SEEDS:-0 1}"
PY=".venv/bin/python"; mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then export WANDB_API_KEY="$(cat .wandb_key)"; fi

# cf22 canonical baseline (A0). Arms append ONE override each.
BASE="model.latent_dim=16 model.hidden=256 model.policy_init_scale=0.01 \
model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.w_compress=0.0 model.dual_latent.rank2_frame.w_band=5.0 \
model.dual_latent.rank2_frame.band_ceiling=1.0 model.dual_latent.rank2_frame.band_floor=0.1 \
model.dual_latent.rank2_frame.band_floor_shape=sigmoid \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false eval.deterministic=true penalty.auto_dose.enabled=false checkpoint.push_wandb=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=0 penalty.lambda_min=1e-4 \
penalty.return_gate.enabled=true penalty.return_gate.ratchet=true penalty.return_gate.shape=leaky_relu \
penalty.return_gate.leak=0.1 penalty.return_gate.mid=0.0 penalty.return_gate.scale=100.0 penalty.return_gate.floor=0.1 \
reward_adapt.mid=0.0 reward_adapt.scale=1000.0 reward_adapt.entropy_anneal=true reward_adapt.entropy_floor.enabled=false \
reward_adapt.logstd_floor.enabled=true reward_adapt.logstd_floor.hi=-1.0 reward_adapt.logstd_floor.lo=-4.0 \
reward_adapt.actor_clip_adapt.enabled=true reward_adapt.actor_clip_adapt.min_frac=0.1 \
smoothing.enabled=false imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 \
optim.skip_nonfinite=true optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=true \
imagination.adaptive_horizon.h_min=15 imagination.adaptive_horizon.h_max=25 \
imagination.adaptive_horizon.ratchet=true imagination.adaptive_horizon.ratchet_base=15"

declare -a ARMS=(
  "A0baseline|"
  "A1modelfit|model.hidden=384"
  "A2pconsist|model.dual_latent.p_consistency_weight=3.0"
  "A3autoalpha|optim.auto_alpha.enabled=true"
  "A4dblvalue|optim.clipped_double_value=true"
  "A5latent|model.latent_dim=32"                 # PM: 2x latent — watch the emergent eff_rank
  # A6 anti-freeze (PM 2026-06-15): the Lyapunov-derived operator-spectrum band on op_d.
  # svband pulls every singular value of A_d below radius_max=0.9<1 — the free zone is
  # ENTIRELY contractive, so a mode can shed penalty ONLY by crossing the gap from σ>1
  # (frozen/marginal, |λ|→1) to σ<0.9. Tests: does un-freezing op_d lift eval/return_det?
  "A6antifreeze|model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05"
  # A7 full-Lyapunov (PM 2026-06-15): svband (=A6) PLUS the Stein consistency term (c) of the
  # unified loss — ‖G − A_d G A_dᵀ − Q̂‖² forces the empirical d covariance to be op_d's
  # stationary covariance. A6 is the svband-only control; A7 adds consistency on top.
  "A7lyap|model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.lyap_weight=0.3"
  # A8 det(op_p)>0 (PM 2026-06-15): svband (=A6) PLUS the policy-operator determinant barrier —
  # require det(op_p)≥0.05>0 so op_p stays invertible + orientation-preserving (GL⁺), the
  # conservative-op_p counterpart to svband's dissipative-op_d. The flow invariant (entropy
  # exponent finite) vs A7's Stein term (which only sets the starting covariance level).
  "A8detpos|model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.detpos_weight=5.0"
  # A9 half-energy init (PM 2026-06-15): the STARTING ratio is everything (A7's Stein term was
  # powerful because it set the initial energy/entropy ratio). Skip the running penalty — just
  # init op_d at |λ|=1/√2 so |λ|²=½ (half latent energy retained per step). Pure init test:
  # cf22 canonical + init_shift only, NO svband/lyap/detpos. Watch if op/radius_d holds ~0.707.
  "A9halfinit|model.operator.init_shift=0.70710678"
  # A10 fifth-energy init (PM 2026-06-15): |λ|²=1/5=0.2 (= 1−4/5, complement of the Pareto/
  # eff_rank ~0.8 ratio) — op_d retains 1/5 of its energy per step, sheds 4/5. init_shift=1/√5.
  # Direct 0.2-vs-0.5 comparison against A9. Strongly contractive ⇒ sharpest path-dependence test.
  "A10fifth|model.operator.init_shift=0.4472136"
  # A11 annealed ratio (PM 2026-06-15): the ratio is NOT constant — it should START at 1 (the
  # natural |λ|, where the dynamics fit wants it) and DECAY toward |λ|²=0.2 (|λ|=√0.2=0.447)
  # WITHOUT reaching. svband ceiling radius_max anneals 1.0→0.4472 on exp(−step/30000); a
  # persistent descending target the dynamics fit can't undo (unlike the one-shot init). Strong
  # w_svband=15 so op_d actually tracks the ceiling. Watch op/radius_d follow op/radius_ceil down.
  "A11anneal|model.operator.w_svband=15 model.operator.radius_max=0.4472136 model.operator.radius_min=0.05 model.operator.radius_anneal_start=1.0 model.operator.radius_anneal_tau=30000"
  # A12 critical (PM 2026-06-15): 0.2 IS the critical entropy-exponent ratio — A10's 0.2-init
  # snapped (policy det + imagined return + eval all vertical at ~220k) but REVERTED. A12 =
  # init op_d AT |λ|²=0.2 (init_shift=0.4472) AND HOLD it there (svband ceiling pinned 0.4472,
  # w=15, NO anneal). Sit at the critical point from t=0. 4 seeds to test if the snap is reliable.
  "A12critical|model.operator.init_shift=0.4472136 model.operator.w_svband=15 model.operator.radius_max=0.4472136 model.operator.radius_min=0.05"
  # A13 just-above-critical (PM 2026-06-16): A12 at r=|λ|²=0.2 shows the ~50/50 converge/collapse
  # split = the SIGNATURE of sitting on the critical separatrix. Nudge r to 0.21 (|λ|=√0.21≈0.4583)
  # to ride just inside the convergent basin. 8 seeds to estimate the convergence FRACTION vs A12's
  # 50/50. Same config as A12, only the ratio moved 0.2->0.21.
  "A13ratio021|model.operator.init_shift=0.4582576 model.operator.w_svband=15 model.operator.radius_max=0.4582576 model.operator.radius_min=0.05"
  # A14 = the VALIDATED A7 (svband+Stein consistency, det_m4=487) at dim^2 latent (16^2=256), PM
  # 2026-06-16. Self-consistency: latent 256 but operator LOW-RANK (rank=32 > intrinsic dynamics dim
  # ~17 and eff_rank ~15) -- full-rank would be 35M params with d-scaled structural penalties. The
  # band/svband/lyap/radius weights are dimension-invariant (per-element / decoupled-per-eigenvalue),
  # SiLU + per-element MSE unchanged. = A7 overrides + latent_dim=256 + operator.rank=32.
  "A14d256|model.latent_dim=256 model.operator.rank=32 model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.lyap_weight=0.3"
  # A15 = A7 at ENVIRONMENT-dim^2 latent: HalfCheetah obs_dim=17 -> latent_dim=17^2=289 (PM clarified
  # "dim" = env dim, not the network/latent dim). Rationale: 289 = dim of the env's 17x17 operator/
  # covariance structure. Same self-consistency as A14 (operator rank=32 low-rank, dim-invariant weights).
  "A15d289|model.latent_dim=289 model.operator.rank=32 model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.lyap_weight=0.3"
  # A16/A17 = PHASED-SVD A/B at env-dim^2 latent (d=289, PM 2026-06-16): the O(d^3) operator SVD
  # amortized to once/episode (struct_every), lyap lever kept every-update. A16 = SVD-only (safe:
  # keep 200 updates/iter). A17 = full phased (also cut updates 200->50 = PM's "duty cycle too short").
  "A16svdonly|model.latent_dim=289 model.operator.rank=32 model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.lyap_weight=0.3 model.operator.struct_every=200"
  "A17phased|model.latent_dim=289 model.operator.rank=32 model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.lyap_weight=0.3 model.operator.struct_every=50 training.model_updates_per_iter=50"
  # A18 entropy-ratio-0.5 (PM 2026-06-16): pin the entropy-ratio coefficient |λ|²=0.5 at env-dim²
  # latent (d=289, rank=32, the Q1-chosen architecture) and TEST the prediction z_std→√0.5≈0.707
  # (the 0.5 replaces the rejected eff_rank/d≈0.91 band law; eff_rank/d stays rank-capped ~0.2 here,
  # so the coefficient is read off z_std, NOT eff_rank). Design = A9's half-energy init (init_shift=
  # 1/√2 ⇒ |λ|²=½) + A12-style HOLD (strong w_svband=15, radius_max=1/√2 ⇒ |λ|≤0.707) so the
  # operator SITS at the ratio rather than drifting up (A16 drifted radius_d→1.0 under w=2). Keeps
  # A16's lyap=0.3 Stein term for comparability; struct_every=50 (4 SVD fires/iter at 200 updates)
  # for a TIGHTER pin than A16's struct_every=200 — the experiment's validity needs the pin to hold.
  "A18ratiohalf|model.latent_dim=289 model.operator.rank=32 model.operator.w_svband=15 model.operator.radius_max=0.70710678 model.operator.radius_min=0.05 model.operator.init_shift=0.70710678 model.dual_latent.lyap_weight=0.3 model.operator.struct_every=50"
  # A19 ride-down (PM 2026-06-16): the data says |λ| WANTS the edge (every arm drifts to ~1.0;
  # low pins A12/A18 fight it and over-damp/die) and z_std self-locks to ~0.80 regardless. So
  # don't pin low — INIT at the edge (init_shift=0.99, where it wants to be = also near-I, the
  # 2-adic precondition) and RIDE THE CEILING DOWN gently (radius_anneal 0.99→0.80 on exp(−t/τ),
  # band-gap/Arrhenius shape) so |λ| descends with an on-average dissipative (positive entropy-
  # flow) gradient instead of being slammed low. = A7's winning recipe (svband 2.0 + Stein lyap
  # 0.3, d=16 full operator) but ceiling ANNEALS 0.99→0.80 instead of static 0.9. Tests: does
  # riding down from the edge track (radius_d follow the ceiling) and beat A7's static-0.9 (=487)?
  # Float solve is edge-safe HERE because we anneal AWAY from the |λ|=1 singularity + skip_nonfinite.
  # (Stochastic-excitation p=0.2 knob + 2-adic head-refit solve = next layer, A20.)
  "A19riddown|model.latent_dim=16 model.operator.w_svband=2.0 model.operator.radius_max=0.8 model.operator.radius_min=0.05 model.operator.init_shift=0.99 model.operator.radius_anneal_start=0.99 model.operator.radius_anneal_tau=80000 model.dual_latent.lyap_weight=0.3"
  # A20 excite (PM 2026-06-16): A7's VALIDATED winner (svband 2.0 + Stein lyap 0.3, d=16, =487)
  # PLUS the discrete stochastic-excitation gate. The system is a marginal oscillator pinned at
  # |λ|≈1 (radius_d=1.0 in every arm; uncontrollable by the band penalty); A19 confirmed you can't
  # MOVE |λ|, so instead PHASE-KICK it: each behaviour update, with prob 0.2, AND only at the
  # operating point (ema z_std∈[0.7,0.9]), inject process noise ε~N(0,(1.0·innov)²) at every
  # imagined-rollout step (innov = EMA of the Stein innovation RMS). Tests: does the gated Q-drive
  # break the fire/collapse bimodality (make BOTH seeds fire) and beat A7=487? excite/gate +
  # excite/noise_std are logged. (2-adic head-refit solve = separate follow-up; not in this arm.)
  "A20excite|model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05 model.dual_latent.lyap_weight=0.3 model.operator.excite_enabled=true model.operator.excite_p=0.2 model.operator.excite_zstd_anchor=0.8 model.operator.excite_zstd_band=0.1 model.operator.excite_scale=1.0"
)
# ONLY="A5latent" launches just that subset (alongside an already-running campaign); empty = all.
throttle(){ while [ "$(jobs -rp|wc -l)" -ge "$JOBS" ]; do sleep 5; done; }
pids=(); idx=0
for entry in "${ARMS[@]}"; do
  arm="${entry%%|*}"; extra="${entry#*|}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $arm "* ]]; then continue; fi
  for seed in $SEEDS; do
    throttle
    tag="abl1-${arm}-s${seed}"; log="results/gridlogs/${tag}.log"; gpu=$(( (idx + ${GPU_BASE:-0}) % NGPU )); idx=$((idx+1))
    echo "launching ${tag} (extra: ${extra:-none}) on GPU ${gpu} -> ${log}"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
      $PY scripts/train.py $BASE $extra seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" > "$log" 2>&1 &
    pids+=($!); sleep 2
  done
done
n=${#pids[@]}; echo "all ${n} abl1 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "abl1 done: $((n-fail))/${n} succeeded"; [ "$fail" -eq 0 ]
