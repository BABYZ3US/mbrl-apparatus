#!/usr/bin/env bash
# DISAGREEMENT-GATE CAMPAIGN — the gate x reward stack, longer training.
#
# 3 reward stacks x {gate off (baseline), gate on} x seeds, HalfCheetah-v5.
#   mlp       : base defaults (MLP reward heads — gate signal = the regularized object)
#   spectral  : spectral_ladder (RFF sigma-ladder + poly; gate signal = aux MLP heads)
#   champion  : champion (gaussian dynamics + spectral reward; the headline recipe)
# Longer STEPS (default 250k vs campaign-1's 100k — champion/spectral need
# convergence time). Each arm: distinct experiment.name -> own W&B group + mirror;
# checkpoint-resumable.
#
# Usage: bash scripts/run_dgate_campaign.sh
#   STEPS=500000 SEEDS="0 1 2" JOBS=4 bash scripts/run_dgate_campaign.sh
set -uo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-4}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# stack -> the +experiment override ("none" = base MLP defaults)
declare -A EXP=( [mlp]="none" [spectral]="spectral_ladder" [champion]="champion" )

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for stack in mlp spectral champion; do
	exp="${EXP[$stack]}"
	for gate in off on; do
		flag=$([ "$gate" = on ] && echo true || echo false)
		for s in $SEEDS; do
			throttle
			tag="dg-${stack}-${gate}"
			log="results/gridlogs/${tag}-s${s}.log"
			echo "launching ${tag} seed ${s} (steps=${STEPS}) -> ${log}"
			ov=""; [ "$exp" != none ] && ov="+experiment=${exp}"
			OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
			$PY scripts/train.py $ov env=halfcheetah seed="$s" \
				experiment.name="$tag" \
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
echo "all ${n} arms launched (max ${JOBS} concurrent, steps=${STEPS}) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "dgate campaign done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
