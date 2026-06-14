#!/usr/bin/env bash
# DISSIPATIVITY — SOFT CONSTRAINT (PM 2026-06-14). cf5 verdict: the HARD reward⊥energy
# frame (rigid orthogonality + one-sided rank-2 pressure) OVER-constrained — eff_rank
# collapsed to ~1.3 (below the rank-2 target, toward rank-1), peaks −38/−55 (worse than
# cf4's −9), grad skips from the double-backward. The rigid constraint is the problem.
# cf6 replaces it with the SOFT thermodynamic inequality (PM's idea):
#   relu(E(d') − E(d) − reward)  —  energy may climb only as much as reward earns it.
# One-sided ⇒ it LETS a running gait BUILD energy for reward (the autonomous reward=0
# case is a pure Lyapunov descent). First-order (no grad skips).
#   drop: w_ortho=0 (hard orthogonality), w_rank2=0 (broken one-sided rank-2 pressure)
#   add:  w_dissip=0.1 (the dissipativity), keep w_lyap=0.1 (autonomous grounding)
#   keep: rank=2 operator, LIGHT clamp (cf5's better-conditioned arm), lyapunov energy.
# energy_mode=lyapunov (dissipativity needs the scalar E head). seeds {0,1} = 2 runs @
# 500k. WIN = peak climbs off −9 toward +569 with eff_rank→2 and cond bounded. Fresh 'cf6-'.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# twin + rank-2 operator + LIGHT clamp + the SOFT dissipativity (no hard ortho/rank2).
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.1 model.dual_latent.rank2_frame.w_lyap=0.1 \
model.dual_latent.rank2_frame.supply=reward model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 smoothing.enabled=false penalty.return_gate.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for seed in $SEEDS; do
	throttle
	tag="cf6-lyapunov-light-s${seed}"
	log="results/gridlogs/${tag}.log"
	gpu=$((idx % NGPU)); idx=$((idx + 1))
	echo "launching ${tag} on GPU ${gpu} -> ${log}"
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
	$PY scripts/train.py $BASE \
		seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
		> "$log" 2>&1 &
	pids+=($!)
	sleep 2
done

n=${#pids[@]}
echo "all ${n} cf6 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf6 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
