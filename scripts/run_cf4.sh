#!/usr/bin/env bash
# COLLAPSE/NaN-FIX ROUND 4 (PM 2026-06-14). cf3 diagnosis (cf3-i0-s2 @100k): the
# i0 baseline is 1 collapse + 2 NaN / 3 seeds. The NaN is NOT a large total loss
# (loss/total stayed ~1e-3) — it's the ACTOR GRADIENT exploding (grad_norm 0.9 ->
# 46,293 -> NaN) because the UNREGULARIZED policy operator op_p (radius_p≈1.06>1)
# makes imagined p-rollouts diverge as the policy crosses return≈0. cf4 = the
# winning twin config + a layered NaN defence (PM-approved):
#   model.dual_latent.radius_p=0.1   bound op_p's spectral radius at the source
#                                    (keeps p ROUGH but not EXPANSIVE)
#   imagination.reward_clip=100      cap per-step imagined reward
#   imagination.return_clip=1000     cap the imagined λ-returns (last line before the loss)
#   optim.value_clip=100             grad-clip the (previously unclipped) value optimizer
#   optim.skip_nonfinite=true        skip the opt step on a non-finite grad (no poisoning)
# Grid: inertia {i0=off, i1=on} x seeds {0,1,2} = 6 runs @ 500k. i0 tests whether the
# stabilization ALONE holds the +569 gait without NaN; i1 adds policy inertia on top.
# Fresh 'cf4-' prefix. (radius_p/clips/skip all default-off ⇒ cf3 etc. byte-unchanged.)
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

# winning twin config + the cf4 NaN-stabilization stack
BASE="model.dynamics=operator model.operator.structure=normal model.operator.w_normal=0.05 \
model.operator.w_smooth=0.1 model.operator.w_radius=0.1 model.dual_latent.enabled=true \
model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 model.dual_latent.smooth_p=false \
model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.1 model.reward_heads=1 \
penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} logging.video.enabled=false \
penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 \
smoothing.enabled=false penalty.return_gate.enabled=false \
imagination.reward_clip=100 imagination.return_clip=1000 optim.value_clip=100 optim.skip_nonfinite=true"

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
		tag="cf4-${inr}-s${seed}"
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
echo "all ${n} cf4 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf4 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
