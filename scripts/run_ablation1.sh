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
logging.video.enabled=false eval.deterministic=true penalty.auto_dose.enabled=false \
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
)
throttle(){ while [ "$(jobs -rp|wc -l)" -ge "$JOBS" ]; do sleep 5; done; }
pids=(); idx=0
for entry in "${ARMS[@]}"; do
  arm="${entry%%|*}"; extra="${entry#*|}"
  for seed in $SEEDS; do
    throttle
    tag="abl1-${arm}-s${seed}"; log="results/gridlogs/${tag}.log"; gpu=$((idx % NGPU)); idx=$((idx+1))
    echo "launching ${tag} (extra: ${extra:-none}) on GPU ${gpu} -> ${log}"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
      $PY scripts/train.py $BASE $extra seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" > "$log" 2>&1 &
    pids+=($!); sleep 2
  done
done
n=${#pids[@]}; echo "all ${n} abl1 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "abl1 done: $((n-fail))/${n} succeeded"; [ "$fail" -eq 0 ]
