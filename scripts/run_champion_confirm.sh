#!/usr/bin/env bash
# CHAMPION CONFIRMATION (2026-06-11): nail down the search sweep's standout
# (lam0~0.41, n_features=1024, cuberoot, gate ON -> eval -32 at 1 seed, vs every
# prior champion run -206..-310). Confirm at 3 seeds and factor out the two
# knobs the user flagged: gate x n_features.
#   gate {off, on} x n_features {256 (low), 1024 (high)} x seeds {0,1,2} = 12 arms.
# Held fixed at the winning recipe: +experiment=champion, lam0 (default 0.41),
# auto_dose OFF, cuberoot + floor 1e-5. 250k steps (the real-verdict length, up
# from the 150k screen). The gate-on/nf-1024 cell is the -32 reproduction.
#
# Usage: bash scripts/run_champion_confirm.sh    (LAM0/STEPS/SEEDS/JOBS/NGPU tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((4 * NGPU))}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1 2}"
LAM0="${LAM0:-0.41}"
PY=".venv/bin/python"
arm_idx=0
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for gate in off on; do
	flag=$([ "$gate" = on ] && echo true || echo false)
	for nf in 256 1024; do
		for s in $SEEDS; do
			throttle
			tag="cc-g${gate}-nf${nf}"
			log="results/gridlogs/${tag}-s${s}.log"
			gpu=$((arm_idx % NGPU)); arm_idx=$((arm_idx + 1))
			echo "launching ${tag} seed ${s} (lam0=${LAM0}, ${STEPS} steps) on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py +experiment=champion env=halfcheetah seed="$s" \
				experiment.name="$tag" \
				penalty.auto_dose.enabled=false \
				penalty.schedule.kind=cuberoot penalty.schedule.floor=1e-5 \
				penalty.schedule.lam0="$LAM0" \
				spectral.n_features="$nf" \
				penalty.disagreement_gate.enabled="$flag" \
				training.total_env_steps="$STEPS" \
				logging.video.enabled=false \
				hydra.run.dir="outputs/${tag}-s${s}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} arms launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "champion-confirm done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
