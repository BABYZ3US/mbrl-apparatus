#!/usr/bin/env bash
# RANK-2 REWARD⊥ENERGY FRAME (PM 2026-06-14). Hypothesis: the controllable essence is
# RANK-2 — two orthogonal axes with opposed senses: reward-ascent (∇_z R, the p/control
# axis) ⊥ energy-descent (the d/dynamics axis). 'We really only need rank-2-ness' (a
# running gait is a 2D limit cycle). cf5 = the cf4-STABILIZED TWIN + rank-2 operators +
# the frame, A/B-ing the two definitions of "energy":
#   energy_mode=lyapunov    a learned E(d), axis −∇_z E, grounded by the autonomous
#                           drift descending it (relu(E(op_d(d,0))−E(d))).
#   energy_mode=contractive op_d's smallest right-singular vector (most-contracted
#                           dynamics direction), pulled back to z. No head.
# Isolated vs cf4: SAME twin + SAME stabilization stack + inertia OFF (i0) — the ONLY
# new variable is the rank-2 frame. Watch latent/gram_eff_rank -> 2 (the order
# parameter), frame/ortho_cos -> 0 (square frame), and whether a rank-2 latent still
# runs HalfCheetah at the +569 level. energy_mode {lyapunov, contractive} x seeds {0,1}
# = 4 runs @ 500k. Fresh 'cf5-' prefix. (Frame default-off ⇒ cf4 etc. byte-unchanged.)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
MODES="${MODES:-lyapunov contractive}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf4-stabilized twin + rank-2 operators + the rank-2 frame (energy_mode set per arm).
# Inertia OFF so cf5 vs cf4-i0 isolates the frame.
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.1 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.1 \
model.dual_latent.rank2_frame.w_rank2=0.01 model.dual_latent.rank2_frame.w_lyap=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 smoothing.enabled=false penalty.return_gate.enabled=false \
imagination.reward_clip=100 imagination.return_clip=1000 optim.value_clip=100 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for mode in $MODES; do
	for seed in $SEEDS; do
		throttle
		tag="cf5-${mode}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.energy_mode="$mode" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf5 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf5 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
