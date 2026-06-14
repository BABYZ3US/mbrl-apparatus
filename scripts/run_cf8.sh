#!/usr/bin/env bash
# COMBINED FIX (PM 2026-06-14). Two findings folded in:
#  (1) the climb-then-collapse root cause is the ADAPTIVE HORIZON collapsing to h_min at
#      convergence -> FIX = fixed H=15 (the policy keeps seeing the gait's distributed
#      reward post-convergence). cf6 climbed to +130 then died this way.
#  (2) the energy term was VACUOUS: the learned E head collapsed to a constant (dissip +
#      lyap == exactly 0), so cf6's +130 came from rank-2+clamp, NOT the energy. FIX =
#      anchor E = anchor*½||d||² + tanh(head) (non-collapsible kinetic floor).
# cf8 = the cf6 dissipativity base + FIXED H=15 x energy_anchor {1, 0}:
#   anchor=1 = the FULL fix (fixed horizon + a real, non-vacuous energy/dissipativity)
#   anchor=0 = horizon fix alone (energy stays vacuous = the rank-2+clamp+horizon base)
# -> isolates whether the now-functional energy adds anything on top of the horizon fix.
# energy_anchor {1, 0} x seeds {0, 1} = 4 runs @ 500k. anchor=1 runs first. WIN = the
# +130 climb HOLDS (no collapse) and pushes toward +569; watch imagine/horizon==15 and
# frame/dissip_resid now NON-zero (energy live) in the anchor=1 arms. Fresh 'cf8-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
ANCHORS="${ANCHORS:-1 0}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf6 dissipativity base + FIXED horizon (the collapse fix). energy_anchor set per arm.
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.1 model.dual_latent.rank2_frame.w_lyap=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false smoothing.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for a in $ANCHORS; do
	for seed in $SEEDS; do
		throttle
		tag="cf8-anchor${a}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.energy_anchor="$a" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf8 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf8 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
