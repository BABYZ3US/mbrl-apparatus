#!/usr/bin/env bash
# DOUBLE-WALLED POTENTIAL — log-det floor + shell ceiling (PM 2026-06-14). The idea: give
# each eigenvalue a HARD FLOOR (log-det −mean ln(λ+eps): blows up as λ->0, can't collapse)
# AND a HARD CEILING (the two-sided shell (target-λ)²: pulls λ down to target, can't run
# away), with a steep/SENSITIVE MIDDLE confining λ into the target band. Combo: w_logdet=0.1
# (floor) + 0.99 energy shell (w_shell=1.0, shell_target=0.99, the ceiling + rank-2 structure).
# Built to ISOLATE the combo vs cf10 (same rank-2 shell at k=4, fixed H=15, light clamp) —
# the only changes are shell_target 1.0->0.99 and the added log-det floor. cf10 (pure rank-2
# shell) held eff_rank=2 but capped at +95; this tests whether the floor+ceiling band (a
# better-conditioned rank-2 rep) moves that — NOTE it is still RANK-2, so it polishes the
# representation, it does not add capacity (cf13/cf12 test rank). seeds {0,1} = 2 runs @ 500k.
# WATCH: latent/gram_cond (should be bounded ~target/floor), frame/shell, frame/logdet_barrier,
# eff_rank (~2), and whether peak moves off +95. Fresh 'cf14-' prefix.
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

# cf10 rank-2 shell stack (k=4 default), + shell_target=0.99 (ceiling) + w_logdet=0.1 (floor).
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=1.0 model.dual_latent.rank2_frame.shell_target=0.99 \
model.dual_latent.rank2_frame.target_rank=2 model.dual_latent.rank2_frame.w_logdet=0.1 \
model.dual_latent.rank2_frame.logdet_eps=0.01 model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false \
smoothing.enabled=false imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 \
optim.skip_nonfinite=true optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for seed in $SEEDS; do
	throttle
	tag="cf14-combo-s${seed}"
	log="results/gridlogs/${tag}.log"
	gpu=$((idx % NGPU)); idx=$((idx + 1))
	echo "launching ${tag} (0.99 shell ceiling + 0.1 logdet floor, rank-2) on GPU ${gpu} -> ${log}"
	OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
	$PY scripts/train.py $BASE \
		seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
		> "$log" 2>&1 &
	pids+=($!)
	sleep 2
done

n=${#pids[@]}
echo "all ${n} cf14 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf14 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
