#!/usr/bin/env bash
# QUEUED LEVERS on the combined fix (PM 2026-06-14). cf8 verdict: fixed H=15 + anchored
# energy STILL collapses — the policy loss still crosses negative (the policy succeeds in
# imagination, returns cross 0+) and nothing brakes the latent runaway that follows,
# because the energy/dissipativity penalty, though no longer vacuous, is too small to
# bite. So bring in the queued levers, led by the one that ACTIVATES the energy:
#   balance  = equilibrium-couple alignment(couple) vs energy(dissip): normalizes each by
#              its running magnitude, so the small dissipativity gets upweighted to equal
#              influence -> energy growth actually gets constrained (the brake).
#   DreamSmooth = spread imagined returns over time (variance + distant-reward visibility).
#   quadratic gate = tie lambda weakly to eval return (relax as return climbs).
# cf9 = cf8 base (fixed H=15 + energy_anchor=1 + dissipativity) x arms:
#   bal = + balance (isolate the energy-activation; the direct test of 'energy was the issue')
#   all = + balance + DreamSmooth + quadratic gate (everything queued)
# arms x seeds {0,1} = 4 runs @ 500k. WIN = policy loss can go negative (policy succeeds)
# WITHOUT the latent runaway/collapse -> the +130 climb holds and pushes toward +569.
# Watch frame/dissip_resid + frame/bal_w_energy (energy now ACTIVE), imagine/horizon==15.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
ARMS="${ARMS:-bal all}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf8 base: fixed horizon + anchored (non-vacuous) energy + dissipativity. Arms add levers.
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.1 model.dual_latent.rank2_frame.w_lyap=0.1 \
model.dual_latent.rank2_frame.energy_anchor=1.0 model.reward_heads=1 penalty.form=frobenius \
env=halfcheetah training.total_env_steps=${STEPS} logging.video.enabled=false \
penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

arm_cfg() {   # $1 = bal | all
	case "$1" in
		bal) echo "model.dual_latent.rank2_frame.balance=true smoothing.enabled=false \
penalty.return_gate.enabled=false" ;;
		all) echo "model.dual_latent.rank2_frame.balance=true smoothing.enabled=true smoothing.sigma=1.5 \
penalty.return_gate.enabled=true penalty.return_gate.shape=quadratic penalty.return_gate.mid=0.0 \
penalty.return_gate.scale=300.0" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for seed in $SEEDS; do
	for arm in $ARMS; do
		throttle
		tag="cf9-${arm}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE $(arm_cfg "$arm") \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf9 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf9 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
