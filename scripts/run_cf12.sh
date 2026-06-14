#!/usr/bin/env bash
# COND-TAMING via the KL/LOG-DET barrier (PM 2026-06-14). cf10's two-sided shell holds
# eff_rank at 2 but cond(G) still blew to 1.1e12 (shell pushes the tail->0, so
# cond=λ_max/λ_min->∞ by construction — a metric artifact, but a near-singular rep is
# fragile). Fix = a log-det volume barrier −mean ln(λ_i+eps) (the spectrum term of
# KL(N(0,Σ)||N(0,I)); MCR2 total coding rate): push every eigenvalue OFF zero. Paired
# with the shell, the tail settles at a small floor instead of 0 -> cond bounded, tunably.
# cf12 = cf10 shell base (fixed H=15 + rank-2 + light clamp + w_shell=1.0) x w_logdet
# {0.02, 0.1} x seeds {0,1} = 4 runs @ 500k. cf10 (w_logdet=0) is the cond-blows-up
# baseline. WIN = cond(G) BOUNDED (no 1e12) while eff_rank stays ~2 and the climb holds.
# Watch latent/gram_cond (should drop) + frame/logdet_barrier + eff_rank + frame/shell.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
LOGDETS="${LOGDETS:-0.02 0.1}"
WSHELL="${WSHELL:-1.0}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.0 model.dual_latent.rank2_frame.w_lyap=0.0 \
model.dual_latent.rank2_frame.balance=false model.dual_latent.rank2_frame.w_shell=${WSHELL} \
model.dual_latent.rank2_frame.shell_target=1.0 model.dual_latent.rank2_frame.logdet_eps=0.01 \
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
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.w_logdet="$ld" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf12 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf12 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
