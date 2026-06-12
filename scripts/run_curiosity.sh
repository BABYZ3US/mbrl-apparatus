#!/usr/bin/env bash
# CURIOSITY RUN (2026-06-11): the clean-eval "winner" (champion + VAE β=0.1,
# the most-compressed channel) + the transformer planner as the policy, scaled
# UP (deeper/wider nets, bigger transformer) and run LONG. Exploratory — the
# IB result was a conservatism artifact (best arms ≈ do-nothing); the open
# question is whether more capacity + time actually learns to RUN.
#
# THE diagnostic: eval/x_velocity (now logged). A passive policy → return ≈ 0
# AND velocity ≈ 0; a real runner → velocity > 0 (and return goes POSITIVE on
# HalfCheetah). That's how we tell "it works" from "it sits still."
#
# Config: +experiment=champion (spectral reward + gaussian dynamics) · VAE
# encoder β=0.1 · planner ON (d_model 256, 4 layers, 8 heads) · depth 4, hidden
# 512 · n_features 1024 · clean obs · 1M steps. 2 seeds across the GPUs.
#
# Usage: bash scripts/run_curiosity.sh    (STEPS/SEEDS/JOBS/NGPU tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$NGPU}"
STEPS="${STEPS:-1000000}"
SEEDS="${SEEDS:-0 1}"
# NOISE = obs-channel input σ. The PM's hypothesis: the big-VAE+planner model
# needs the DENOISING regime (a real channel bottleneck) to do something other
# than copy x. >0 gives the VAE compression a job. Default 0.5 (the IB sweep level).
NOISE="${NOISE:-0.5}"
PY=".venv/bin/python"
arm_idx=0
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for s in $SEEDS; do
	throttle
	nz=$(echo "$NOISE" | tr -d '.')
	tag="curio-bigvae-plan-n${nz}"
	log="results/gridlogs/${tag}-s${s}.log"
	gpu=$((arm_idx % NGPU)); arm_idx=$((arm_idx + 1))
	echo "launching ${tag} seed ${s} (${STEPS} steps, noise=${NOISE}) on GPU ${gpu} -> ${log}"
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
	$PY scripts/train.py +experiment=champion env=halfcheetah seed="$s" \
		experiment.name="$tag" \
		model.encoder=vae model.vae.beta=0.1 \
		planner.enabled=true planner.d_model=256 planner.layers=4 planner.nhead=8 \
		model.depth=4 model.hidden=512 spectral.n_features=1024 \
		+env.obs_noise="$NOISE" \
		training.total_env_steps="$STEPS" \
		logging.video.enabled=false \
		hydra.run.dir="outputs/${tag}-s${s}" \
		> "$log" 2>&1 &
	pids+=($!)
	sleep 2
done

n=${#pids[@]}
echo "all ${n} arms launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "curiosity run done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
