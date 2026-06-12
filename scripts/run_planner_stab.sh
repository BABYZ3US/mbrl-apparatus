#!/usr/bin/env bash
# TRANSFORMER PLANNER STABILIZATION (2026-06-12): the curiosity runs were the
# ONLY configs that reached positive return (peaks +201..+754 = the cheetah
# actually ran) but collapsed catastrophically after the peak. The stability
# audit pinned collapse, not capacity, as the problem. This isolates the two
# fixes on the exact config that peaked highest (clean affine + VAE β0.1 +
# transformer planner): a 2x2 of imagination HORIZON x imagination-latent
# ALIGNMENT (arXiv 2507.16450 — keep long plans on the encoder manifold).
#
#   stab-h15-a0 : horizon 15, align 0   (the collapse CONTROL — reproduces it)
#   stab-h15-a1 : horizon 15, align 1   (does alignment alone stop the collapse?)
#   stab-h30-a0 : horizon 30, align 0   (does a longer horizon alone help?)
#   stab-h30-a1 : horizon 30, align 1   (combined)
# Watch eval/return (does the peak HOLD vs collapse?), eval/x_velocity (does it
# run?), imagine/align (the alignment term), actor/grad_norm. 1 seed each (screen).
#
# Usage: bash scripts/run_planner_stab.sh   (STEPS/JOBS/NGPU tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-700000}"
PY=".venv/bin/python"
arm_idx=0
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

BASE="+experiment=champion env=halfcheetah model.dynamics=affine model.encoder=vae model.vae.beta=0.1 planner.enabled=true planner.d_model=256 planner.layers=4 planner.nhead=8 model.depth=4 model.hidden=512 spectral.n_features=1024 logging.video.enabled=false"

# tag -> (horizon, align_weight)
arm_cfg() {
	case "$1" in
		h15-a0) echo "imagination.horizon=15 imagination.align_weight=0.0" ;;
		h15-a1) echo "imagination.horizon=15 imagination.align_weight=1.0" ;;
		h30-a0) echo "imagination.horizon=30 imagination.align_weight=0.0" ;;
		h30-a1) echo "imagination.horizon=30 imagination.align_weight=1.0" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for arm in h15-a0 h15-a1 h30-a0 h30-a1; do
	throttle
	tag="stab-${arm}"
	log="results/gridlogs/${tag}-s0.log"
	gpu=$((arm_idx % NGPU)); arm_idx=$((arm_idx + 1))
	echo "launching ${tag} (${STEPS} steps) on GPU ${gpu} -> ${log}"
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
	$PY scripts/train.py $BASE seed=0 experiment.name="$tag" $(arm_cfg "$arm") \
		training.total_env_steps="$STEPS" hydra.run.dir="outputs/${tag}-s0" \
		> "$log" 2>&1 &
	pids+=($!)
	sleep 2
done

n=${#pids[@]}
echo "all ${n} arms launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "planner-stab done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
