#!/usr/bin/env bash
# TWIN v2 (PM 2026-06-13): the dual-latent TWIN arm (separate smoothness — smooth
# dynamics latent d / rough policy latent p) WON the lambda sweep: dltw λ=1e-3
# peaked +569 vel .86 (2.7x founding) — but with two handicaps: DreamSmooth was
# OFF (a confound) and the return-gate was the OLD (sign-blind, near-zero-spike)
# form. This rerun fixes both: DreamSmooth ON + the corrected sign-aware gate, and
# asks two questions:
#   - does the (corrected) return-gate help?          -> gate ON vs OFF at λ=1e-3
#   - is the curvature penalty even needed for twin?  -> λ=1e-3 vs λ OFF
#
# Matrix: TWIN + DreamSmooth ON (sigma=1), 3 seeds x:
#   l1em3-goff : lam0=1e-3, return-gate OFF
#   l1em3-gon  : lam0=1e-3, return-gate ON  (corrected: sign-aware, floor 0.1, slew)
#   loff       : penalize_reward=false (lambda OFF; operator smoothness + DreamSmooth only)
# = 9 runs @ 500k HalfCheetah. NOTE: "λ-off × gate-on" is omitted on purpose — the
# gate is a multiplier on lambda, so with λ off it is identical to "λ-off × gate-off".
# Judge by PEAK (the sweep showed a universal peak-then-collapse); watch eval/return,
# eval/x_velocity, penalty/return_gate (does the gate relax λ as it learns to run?).
#
# Usage: bash scripts/run_twin_v2.sh        (STEPS/JOBS/NGPU/SEEDS/ARMS tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-l1em3-goff l1em3-gon loff}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# the winning twin config + DreamSmooth ON (the confound fix)
TWIN="model.dynamics=operator model.operator.structure=normal model.operator.w_normal=0.05 \
model.operator.w_smooth=0.1 model.operator.w_radius=0.1 model.dual_latent.enabled=true \
model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 model.dual_latent.smooth_p=false \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot smoothing.enabled=true smoothing.sigma=1.0"

arm_cfg() {
	case "$1" in
		l1em3-goff) echo "model.dual_latent.penalize_reward=true penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false" ;;
		l1em3-gon)  echo "model.dual_latent.penalize_reward=true penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=true" ;;
		loff)       echo "model.dual_latent.penalize_reward=false penalty.return_gate.enabled=false" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for arm in $ARMS; do
	for seed in $SEEDS; do
		throttle
		tag="twinv2-${arm}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $TWIN $(arm_cfg "$arm") \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} twin-v2 runs launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "twin-v2 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
