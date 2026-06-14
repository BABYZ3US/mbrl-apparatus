#!/usr/bin/env bash
# HORIZON x DREAMSMOOTH (PM 2026-06-14). Diagnosis from the metrics: the climb-then-
# collapse is caused by the ADAPTIVE IMAGINATION HORIZON collapsing to h_min=5 at
# convergence — policy converges -> reward sharpens -> curvature pen_val rises -> the
# curvature-certified horizon shortens -> the policy can no longer see the gait's
# distributed reward -> collapse (pol_loss sign-flip at ret_mean crossing 0 is the
# trigger). cf6's return_clip bounded the variance (ret_var <15, not cf3's 5e8) yet it
# STILL collapsed (+130 -> -434) — so the horizon, not the variance, is the root cause.
# The adaptive horizon is now OBSOLETE: the cf4 stabilization stack does targeted
# stability; the horizon restriction just truncates the gait reward at the worst moment.
# cf7 = the cf6 dissipativity base x {adaptive horizon | fixed H=15} x {DreamSmooth off |
# on}. fixed-H = the root-cause fix (keep seeing the gait reward post-convergence);
# DreamSmooth (PM's idea) = spread returns over time (variance + short-horizon amplifier).
#   adapt-dsoff = cf6 baseline (reproduces +130->collapse)   fixed-dsoff = the horizon fix
#   adapt-dson  = DreamSmooth alone                          fixed-dson  = both
# seed 0, 4 runs @ 500k. WIN = the climb to +130 HOLDS (no collapse) and pushes toward
# +569, imagine/horizon stays ~15 in the fixed arms. Fresh 'cf7-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0}"
HORIZONS="${HORIZONS:-adapt fixed}"
SMOOTHS="${SMOOTHS:-dsoff dson}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# the cf6 dissipativity base (twin + rank-2 + light clamp + soft dissipativity).
# horizon + smoothing are set per arm below.
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.1 model.dual_latent.rank2_frame.w_lyap=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false imagination.reward_clip=1000 \
imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true optim.policy_ema_decay=0.0"

horizon_cfg() {   # $1 = adapt | fixed
	case "$1" in
		adapt) echo "imagination.adaptive_horizon.enabled=true" ;;
		fixed) echo "imagination.adaptive_horizon.enabled=false imagination.horizon=15" ;;
	esac
}
smooth_cfg() {    # $1 = dsoff | dson
	case "$1" in
		dsoff) echo "smoothing.enabled=false" ;;
		dson)  echo "smoothing.enabled=true smoothing.sigma=1.5" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for hz in $HORIZONS; do
	for ds in $SMOOTHS; do
		for seed in $SEEDS; do
			throttle
			tag="cf7-${hz}-${ds}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE $(horizon_cfg "$hz") $(smooth_cfg "$ds") \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} cf7 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf7 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
