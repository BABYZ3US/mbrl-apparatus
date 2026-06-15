#!/usr/bin/env bash
# EMERGENT RANK — log-det ALONE, no rank enforcement (PM 2026-06-14). The shell did two
# jobs: (1) anti-collapse [keep modes off zero — necessary] and (2) rank-k selection
# [crush the tail — a HAND-SET integer we ended up sweeping]. We only needed (1). cf12
# drops (2) entirely: NO shell (w_shell=0), FULL operator (rank=0), and the ONLY
# structural pressure is the log-det / KL volume barrier −mean ln(λ_i+eps) that keeps
# every eigenvalue OFF zero. The task's own compression then chooses how many modes to
# grow above the floor -> the RANK EMERGES (and incidentally answers 'is a gait rank-2?'
# — if so the emergent eff_rank settles near 2 on its own; if it wants more, it takes
# more). Bigger latent (16) gives it room. Sweep w_logdet {0.1, 0.5} (the anti-collapse
# strength — too weak collapses, too strong goes full-rank) x seeds {0,1} = 4 runs @ 500k.
# WATCH: latent/gram_eff_rank = the EMERGENT rank (what the task picks), and does it climb
# past cf10's +95. Fresh 'cf12-' prefix. (Flip w_shell back >0 / set operator.rank for
# the original cond-tame-on-shell variant.)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
LOGDETS="${LOGDETS:-0.1 0.5}"
LATENT="${LATENT:-16}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# NO rank enforcement: w_shell=0, operator.rank=0 (full A). ONLY the log-det barrier acts
# as anti-collapse; the rank emerges. Bigger latent for room. cf10 fixed-horizon + light clamp.
BASE="model.latent_dim=${LATENT} model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.logdet_eps=0.01 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false smoothing.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for ld in $LOGDETS; do
	for seed in $SEEDS; do
		throttle
		tag="cf12-logdet${ld}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (latent=${LATENT}, NO rank enforcement, w_logdet=${ld}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.w_logdet="$ld" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf12 runs launched (max ${JOBS} concurrent, ${STEPS} steps, latent=${LATENT}, emergent rank)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf12 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
