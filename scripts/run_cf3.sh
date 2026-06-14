#!/usr/bin/env bash
# COLLAPSE-FIX ROUND 3 (PM 2026-06-14). Gates are a dead end (round-1 quadratic
# +237 didn't hold; round-2 bump +60, both << plain twin +569 — gates perturb λ
# off the 1e-3 sweet spot at the wrong moment). So drop the gate entirely and test
# the one untested-clean lever — POLICY INERTIA — on the WINNING config:
#   plain twin (smooth-d/rough-p), DreamSmooth OFF, λ=1e-3, NO gate, best-ckpt.
#   inertia {i0=off, i1=on} x seeds {0,1,2} = 6 runs @ 500k HalfCheetah.
# i0 reproduces the sweep's +569 config (baseline); i1 adds the two-timescale
# policy inertia (slow EMA used for acting + a soft weight anchor). Judge by
# FINAL/peak — does inertia HOLD the gait where nothing else has?
# Fresh 'cf3-' prefix => no collision with cfix-/cf2-/cf2b-.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2}"
INERTIAS="${INERTIAS:-i0 i1}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# the sweep's winning config: twin + DreamSmooth OFF + λ=1e-3, NO gate, NO align.
BASE="model.dynamics=operator model.operator.structure=normal model.operator.w_normal=0.05 \
model.operator.w_smooth=0.1 model.operator.w_radius=0.1 model.dual_latent.enabled=true \
model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 model.dual_latent.smooth_p=false \
model.dual_latent.penalize_reward=true model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 smoothing.enabled=false \
penalty.return_gate.enabled=false"

inertia_cfg() {   # $1 = i0 | i1
	case "$1" in
		i0) echo "optim.policy_ema_decay=0.0" ;;
		i1) echo "optim.policy_ema_decay=0.99 optim.policy_ema_act=true optim.policy_inertia=0.1" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for inr in $INERTIAS; do
	for seed in $SEEDS; do
		throttle
		tag="cf3-${inr}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE $(inertia_cfg "$inr") \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf3 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf3 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
