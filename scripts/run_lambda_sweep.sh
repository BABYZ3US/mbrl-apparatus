#!/usr/bin/env bash
# LAMBDA-DOSE SWEEP (PM 2026-06-13): find the curvature-penalty dose on the arms
# where lambda actually matters (plain MLP reward — the dose-validation finding
# showed lambda is moot on the self-regularizing spectral champion). PM suspects
# the optimum is LOW, ~1e-3. Sweep lam0 DIRECTLY (auto_dose OFF, return_gate OFF
# so nothing modulates it) on a MONOTONIC cuberoot decay.
#
# Matrix: lam0 {1e-4, 1e-3, 1e-2} x 3 config arms x seeds {0,1} = 18 runs @ 500k:
#   found : the validated +98 recipe (MLP + Frobenius + DreamSmooth + affine)
#   dlsh  : dual-latent SHARED  — one operator on z (CONJOINED smoothness)
#   dltw  : dual-latent TWIN    — op_d smooth / op_p rough (SEPARATE smoothness)
# dual arms: operator dynamics + MLP reward (1 head) + Frobenius penalty in p-coords,
# NO DreamSmooth; operator priors fixed (structure=normal, w_normal .05 / w_smooth .1 /
# w_radius .1 on op_d / the shared op); twin op_p left rough (operator_p all-zero) +
# couple_weight .1. Watch eval/return + eval/x_velocity (conservatism check: a do-
# nothing policy scores ~0 return AND ~0 velocity — a real runner has v>0).
#
# Usage: bash scripts/run_lambda_sweep.sh        (STEPS/JOBS/NGPU/SEEDS/LAMS tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
LAMS="${LAMS:-1e-4 1e-3 1e-2}"
ARMS="${ARMS:-found dlsh dltw}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

COMMON="env=halfcheetah training.total_env_steps=${STEPS} logging.video.enabled=false \
penalty.auto_dose.enabled=false penalty.return_gate.enabled=false penalty.schedule.kind=cuberoot"
DUAL="model.dynamics=operator model.operator.structure=normal model.operator.w_normal=0.05 \
model.operator.w_smooth=0.1 model.operator.w_radius=0.1 model.dual_latent.enabled=true \
model.dual_latent.penalize_reward=true model.reward_heads=1 smoothing.enabled=false penalty.form=frobenius"

arm_cfg() {   # $1 = arm, $2 = lam. NB: +experiment=founding must LEAD (Hydra
	            # requires the defaults-list append before the value overrides).
	case "$1" in
		found) echo "penalty.schedule.lam0=$2" ;;
		dlsh)  echo "$DUAL model.dual_latent.mode=shared penalty.schedule.lam0=$2" ;;
		dltw)  echo "$DUAL model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false penalty.schedule.lam0=$2" ;;   # smooth d / rough p
	esac
}
arm_prefix() { [ "$1" = "found" ] && echo "+experiment=founding"; }   # leads the overrides
lam_tag() { echo "$1" | sed 's/e-/em/; s/\.//g'; }   # 1e-3 -> 1em3

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); arm_idx=0
for arm in $ARMS; do
	for lam in $LAMS; do
		for seed in $SEEDS; do
			throttle
			tag="lamsweep-${arm}-l$(lam_tag "$lam")-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((arm_idx % NGPU)); arm_idx=$((arm_idx + 1))
			echo "launching ${tag} on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $(arm_prefix "$arm") $COMMON $(arm_cfg "$arm" "$lam") \
				seed="$seed" experiment.name="$tag" \
				hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} runs launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "lambda-sweep done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
