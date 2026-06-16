#!/usr/bin/env bash
# A6-vs-A7 FOLLOW-UP (2026-06-16) — pre-registered: docs/abl1_followup_a6a7_prereg.md
# Tests Ablation-1 finding (iii): does the Stein consistency term (lyap_weight=0.3, the ONLY
# difference between A6 and A7) raise the rate at which a seed reaches+holds a firing policy?
# Powered replication of abl1 A6/A7: 2 arms x 8 seeds (0-7) @ 500k. Distinct 'fu-a6a7-' W&B
# group prefix so it does NOT contaminate the original 'abl1-' group. The cf22 canonical BASE
# below is copied verbatim from run_ablation1.sh (kept self-contained to avoid drift); the two
# arms differ ONLY in model.dual_latent.lyap_weight (A6=absent/0.0, A7=0.3).
#   A6antifreeze  — svband only (control): pulls op_d's singular values below radius_max=0.9.
#   A7lyap        — A6 + Stein consistency term (treatment).
# Verdict rule is PRE-REGISTERED in the prereg doc — do not change it after the first launch.
set -uo pipefail
cd "$(dirname "$0")/.."
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"; [ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((5 * NGPU))}"; STEPS="${STEPS:-500000}"; SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
PY=".venv/bin/python"; mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then export WANDB_API_KEY="$(cat .wandb_key)"; fi

# cf22 canonical baseline (identical to run_ablation1.sh's BASE). Arms append ONE override block.
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

SVBAND="model.operator.w_svband=2.0 model.operator.radius_max=0.9 model.operator.radius_min=0.05"
declare -a ARMS=(
  "A6antifreeze|${SVBAND}"
  "A7lyap|${SVBAND} model.dual_latent.lyap_weight=0.3"
)
throttle(){ while [ "$(jobs -rp|wc -l)" -ge "$JOBS" ]; do sleep 5; done; }
pids=(); idx=0
for entry in "${ARMS[@]}"; do
  arm="${entry%%|*}"; extra="${entry#*|}"
  for seed in $SEEDS; do
    throttle
    tag="fu-a6a7-${arm}-s${seed}"; log="results/gridlogs/${tag}.log"; gpu=$(( (idx + ${GPU_BASE:-0}) % NGPU )); idx=$((idx+1))
    echo "launching ${tag} (extra: ${extra}) on GPU ${gpu} -> ${log}"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
      $PY scripts/train.py $BASE $extra seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" > "$log" 2>&1 &
    pids+=($!); sleep 2
  done
done
n=${#pids[@]}; echo "all ${n} fu-a6a7 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "fu-a6a7 done: $((n-fail))/${n} succeeded"; [ "$fail" -eq 0 ]
