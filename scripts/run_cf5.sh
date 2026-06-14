#!/usr/bin/env bash
# RANK-2 REWARD⊥ENERGY FRAME + CLAMP SWEEP (PM 2026-06-14). cf4 verdict: the
# stabilization FIXED the NaN (0 nan/skips, radius_p pinned, clean to 365k where cf3
# died) but FROZE learning — i0 stuck at peak −9 (never climbed to +569), and the new
# order-parameter readout shows why: eff_rank≈2 but cond(G)≈1e12 (a degenerate, ill-
# conditioned ~2-mode latent). Two fixes, tested together here:
#   (1) the rank-2 reward⊥energy FRAME — turn the emergent (but unaligned) rank-2 into
#       reward-ascent ⊥ energy-descent. energy_mode {lyapunov, contractive}.
#   (2) LIGHTER CLAMPS — the heavy radius_p=0.1 pin (σ_max≤1) likely blocks the operator
#       from AMPLIFYING into a running gait (you must climb energy to run); relax toward
#       the edge and lean on skip_nonfinite as the safety net.
#         heavy = the cf4 stack (radius_p .1 / reward_clip 100 / return_clip 1k / value_clip 100)
#         light = radius_p .02 / reward_clip 1k / return_clip 10k / value_clip 1k
#       skip_nonfinite=true in BOTH (free safety net, no over-damping cost).
# Grid: energy_mode {lyapunov, contractive} x clamp {heavy, light} x seed {0} = 4 runs
# @ 500k. cf5-heavy vs cf4-i0 isolates the FRAME; cf5-light vs cf5-heavy isolates the
# CLAMP. Watch: does peak climb off −9? does eff_rank→2 with cond bounded? Fresh 'cf5-'.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0}"
MODES="${MODES:-lyapunov contractive}"
CLAMPS="${CLAMPS:-heavy light}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# twin + rank-2 operators + the rank-2 frame; inertia OFF (isolate the frame vs cf4-i0).
# Clamp-specific knobs are appended per arm by clamp_cfg().
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.1 \
model.dual_latent.rank2_frame.w_rank2=0.01 model.dual_latent.rank2_frame.w_lyap=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 smoothing.enabled=false penalty.return_gate.enabled=false \
optim.policy_ema_decay=0.0"

clamp_cfg() {   # $1 = heavy | light
	case "$1" in
		heavy) echo "model.dual_latent.radius_p=0.1 imagination.reward_clip=100 \
imagination.return_clip=1000 optim.value_clip=100 optim.skip_nonfinite=true" ;;
		light) echo "model.dual_latent.radius_p=0.02 imagination.reward_clip=1000 \
imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for mode in $MODES; do
	for clamp in $CLAMPS; do
		for seed in $SEEDS; do
			throttle
			tag="cf5-${mode}-${clamp}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE $(clamp_cfg "$clamp") \
				model.dual_latent.rank2_frame.energy_mode="$mode" \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} cf5 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf5 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
