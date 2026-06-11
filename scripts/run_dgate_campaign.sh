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
# Multi-GPU: arms round-robin across the visible GPUs via CUDA_VISIBLE_DEVICES
# (train.py device=auto sees only its assigned GPU as cuda:0 — no code change);
# default JOBS = 4 per GPU (a 3090 is compute-bound at ~4 of these small-net
# arms; more per GPU just slows each).
#
# Usage: bash scripts/run_dgate_campaign.sh
#   STEPS=500000 SEEDS="0 1 2" JOBS=8 NGPU=2 bash scripts/run_dgate_campaign.sh
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((4 * NGPU))}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1}"
STACKS="${STACKS:-mlp spectral champion}"
# Network capacity (24GB VRAM, ~2GB used -> huge headroom). DEPTH/HIDDEN
# default to "" = the per-stack config's own values (comparable to prior runs);
# set them to deepen/widen every arm uniformly.
DEPTH="${DEPTH:-}"
HIDDEN="${HIDDEN:-}"
PY=".venv/bin/python"
arm_idx=0
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# stack -> the +experiment override ("none" = base MLP defaults)
declare -A EXP=( [mlp]="none" [spectral]="spectral_ladder" [champion]="champion" )

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for stack in $STACKS; do
	exp="${EXP[$stack]}"
	for gate in off on; do
		flag=$([ "$gate" = on ] && echo true || echo false)
		for s in $SEEDS; do
			throttle
			tag="dg-${stack}-${gate}"
			log="results/gridlogs/${tag}-s${s}.log"
			# PIN_GPU forces every arm onto one physical GPU (e.g. add a
			# seed batch on the idle GPU beside a running campaign); else
			# round-robin across NGPU.
			gpu="${PIN_GPU:-$((arm_idx % NGPU))}"; arm_idx=$((arm_idx + 1))
			echo "launching ${tag} seed ${s} (steps=${STEPS}) on GPU ${gpu} -> ${log}"
			ov=""; [ "$exp" != none ] && ov="+experiment=${exp}"
			cap=""; [ -n "$DEPTH" ] && cap="$cap model.depth=$DEPTH"
			[ -n "$HIDDEN" ] && cap="$cap model.hidden=$HIDDEN"
			OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $ov $cap env=halfcheetah seed="$s" \
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
echo "all ${n} arms launched (max ${JOBS} concurrent across ${NGPU} GPU(s), steps=${STEPS}) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "dgate campaign done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
