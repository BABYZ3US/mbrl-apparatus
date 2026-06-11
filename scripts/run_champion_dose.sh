#!/usr/bin/env bash
# CHAMPION DOSE VALIDATION (2026-06-11): does the FIXED auto_dose (b77b0f1, now
# referenced against the reward fit instead of the gaussian-dynamics NLL) find a
# sane champion dose on its own — and how does it compare to the hand-tuned dose?
#
# 3 dose sources x 3 seeds = 9 arms, champion, everything else held at the good
# config (cuberoot + floor, n_features=1024, gate OFF — gate is dead for champion):
#   cd-auto   : penalty.auto_dose ON  -> auto picks lam0 (the fix under test)
#   cd-fixed  : auto_dose OFF, lam0=0.41 (the hand-tuned winner / reference)
#   cd-off    : auto_dose OFF, lam0=0   (penalty-OFF control — should reproduce
#               the old broken champion -247..-310, confirming the diagnosis)
# Reads: (a) is cd-auto's lam0 positive at real scale, (b) does it match cd-fixed
# or under-dose, (c) does cd-off reproduce the unpenalized baseline.
#
# Usage: bash scripts/run_champion_dose.sh   (STEPS/SEEDS/JOBS/NGPU tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((4 * NGPU))}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1 2}"
PY=".venv/bin/python"
arm_idx=0
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# per-arm dose overrides (everything else is shared/held below)
dose_override() {
	case "$1" in
		auto)  echo "penalty.auto_dose.enabled=true" ;;
		fixed) echo "penalty.auto_dose.enabled=false penalty.schedule.lam0=0.41" ;;
		off)   echo "penalty.auto_dose.enabled=false penalty.schedule.lam0=0.0" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for dose in auto fixed off; do
	ov="$(dose_override "$dose")"
	for s in $SEEDS; do
		throttle
		tag="cd-${dose}"
		log="results/gridlogs/${tag}-s${s}.log"
		gpu=$((arm_idx % NGPU)); arm_idx=$((arm_idx + 1))
		echo "launching ${tag} seed ${s} (${STEPS} steps) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py +experiment=champion env=halfcheetah seed="$s" \
			experiment.name="$tag" \
			penalty.schedule.kind=cuberoot penalty.schedule.floor=1e-5 \
			spectral.n_features=1024 \
			penalty.disagreement_gate.enabled=false \
			$ov \
			training.total_env_steps="$STEPS" \
			logging.video.enabled=false \
			hydra.run.dir="outputs/${tag}-s${s}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} arms launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "champion-dose done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
