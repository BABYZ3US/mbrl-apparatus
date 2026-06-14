#!/usr/bin/env bash
# COLLAPSE-FIX ROUND 2 (PM 2026-06-13). Round 1 verdict: align_weight=1.0 was too
# weak — NaN-prone, didn't hold the gait; QUADRATIC gate > cuberoot (settled). This
# round applies the STRONGER collapse levers on the winning config (twin smooth-d/
# rough-p, DreamSmooth OFF, λ=1e-3, best-ckpt) with the BUMP reward-gate (PM): λ
# MAXED at the return≈0 "phase transition", released both ways — brace the policy
# where it's unstable (crossing 0, incl. on the way DOWN through a collapse).
#   align {5, 10}  x  inertia {i0=off, i1=on}  x  seeds {0,1,2}  = 12 runs @ 500k.
# inertia i1 = slow policy EMA used for ACTING (policy_ema_act) + a soft weight
# anchor (policy_inertia) — two-timescale stabilizer: the policy gets extra inertia
# vs the faster operator so it can't lunge at transient model errors.
# Judge by FINAL/peak (did the gait HOLD) and NaN rate (did stabilization help).
# Fresh 'cf2b-' prefix => no collision with cfix-* (round 1) or cf2-* (quadratic try).
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2}"
ALIGNS="${ALIGNS:-5 10}"
INERTIAS="${INERTIAS:-i0 i1}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# winner config + quadratic gate + best-ckpt (best-ckpt is automatic in train.py)
BASE="model.dynamics=operator model.operator.structure=normal model.operator.w_normal=0.05 \
model.operator.w_smooth=0.1 model.operator.w_radius=0.1 model.dual_latent.enabled=true \
model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 model.dual_latent.smooth_p=false \
model.dual_latent.penalize_reward=true model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 smoothing.enabled=false \
penalty.return_gate.enabled=true penalty.return_gate.shape=bump penalty.return_gate.mid=0.0 \
penalty.return_gate.scale=400.0"

inertia_cfg() {   # $1 = i0 | i1
	case "$1" in
		i0) echo "optim.policy_ema_decay=0.0" ;;
		i1) echo "optim.policy_ema_decay=0.99 optim.policy_ema_act=true optim.policy_inertia=0.1" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for align in $ALIGNS; do
	for inr in $INERTIAS; do
		for seed in $SEEDS; do
			throttle
			tag="cf2b-a${align}-${inr}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE imagination.align_weight="$align" $(inertia_cfg "$inr") \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} cf2 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf2 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
