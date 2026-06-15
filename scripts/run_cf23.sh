#!/usr/bin/env bash
# LR-TIED-TO-LAMBDA (PM 2026-06-15): tie the MODEL learning rate to the curvature lambda so
# they AGREE IN EXPONENT. lambda(t) = lam0*(t0/(t0+t))^(1/3) (cuberoot, R12); set the model lr
# to the SAME cuberoot form with the SAME t0 (optim.lr_schedule.kind=cuberoot, t0=10000) so
# lr(t) ∝ lambda(t) — regularization strength and model step size anneal on a matched
# timescale. Base = cf21 (curvature lambda SCHEDULE active, lam0=1e-3 — needed so there is an
# exponent to tie to; cf22 had dropped it), + the variance bound (reward-adaptive log_std
# floor) + leaky_relu gate + band(sigmoid floor) + ratchet horizon + near-zero init. 6 seeds.
# NOTE inherits cf21's sigma-only variance bound (no mean bound), so the tanh-mean-saturation
# spread carries over — this run tests the LR/lambda exponent tie, not the spread. WIN = matched
# annealing climbs cleaner/earlier than cf21 (fixed lr). Verify optim/model_lr decays cuberoot
# and tracks penalty/lambda. Fresh 'cf23-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
SHAPES="${SHAPES:-sigmoid}"
LSF_HI="${LSF_HI:--1.0}"
LSF_LO="${LSF_LO:--4.0}"
LR_T0="${LR_T0:-10000}"              # match penalty.schedule.t0 => lr ∝ lambda
WBAND="${WBAND:-5.0}"
LATENT="${LATENT:-16}"
HID="${HID:-256}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf21 base (curvature lambda cuberoot ACTIVE) + the LR tied to lambda's exponent.
BASE="model.latent_dim=${LATENT} model.hidden=${HID} model.policy_init_scale=0.01 \
model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.w_compress=0.0 model.dual_latent.rank2_frame.w_band=${WBAND} \
model.dual_latent.rank2_frame.band_ceiling=1.0 model.dual_latent.rank2_frame.band_floor=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.lambda_min=1e-4 penalty.return_gate.enabled=true penalty.return_gate.ratchet=true \
penalty.return_gate.shape=leaky_relu penalty.return_gate.leak=0.1 penalty.return_gate.mid=0.0 \
penalty.return_gate.scale=100.0 penalty.return_gate.floor=0.1 \
optim.lr_schedule.kind=cuberoot optim.lr_schedule.t0=${LR_T0} optim.lr_schedule.floor=0.0 \
reward_adapt.mid=0.0 reward_adapt.scale=1000.0 \
reward_adapt.entropy_anneal=true reward_adapt.entropy_floor.enabled=false \
reward_adapt.logstd_floor.enabled=true reward_adapt.logstd_floor.hi=${LSF_HI} reward_adapt.logstd_floor.lo=${LSF_LO} \
reward_adapt.actor_clip_adapt.enabled=true reward_adapt.actor_clip_adapt.min_frac=0.1 \
smoothing.enabled=false imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 \
optim.skip_nonfinite=true optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=true \
imagination.adaptive_horizon.h_min=15 imagination.adaptive_horizon.h_max=25 \
imagination.adaptive_horizon.ratchet=true imagination.adaptive_horizon.ratchet_base=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for shape in $SHAPES; do
	for seed in $SEEDS; do
		throttle
		tag="cf23-${shape}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (LR tied to lambda exponent: cuberoot t0=${LR_T0}, band=${shape}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.band_floor_shape="$shape" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf23 runs launched (max ${JOBS} concurrent, ${STEPS} steps, LR tied to lambda's exponent)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf23 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
