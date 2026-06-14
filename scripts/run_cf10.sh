#!/usr/bin/env bash
# TWO-SIDED ENERGY SHELL (PM 2026-06-14). Diagnosis chain: cf9's dissipativity was inert
# (raw_energy==0) because it only penalizes energy GROWTH, but the failure is the latent
# CONTRACTING / collapsing (eff_rank->1) — which trivially satisfies it. No reweighting
# (cap, balance, -ln) can activate a structural zero. Fix = a TWO-SIDED rank-2 energy
# SHELL (Ginzburg-Landau double-well, PM): hold the top-2 Gram eigenvalues at a setpoint
# and push the rest to 0 -> Σ_{i<=2}(target-λ_i)² + Σ_{i>2}λ_i². Penalizes collapse to
# rank-1 AND norm collapse AND explosion, in every mode. Operates on the Gram directly,
# so there is no learned energy head to go vacuous.
# cf10 = fixed H=15 (horizon fix) + rank-2 operator + light clamp + ONLY the shell
# (dissipativity / balance / energy head all dropped). Sweep w_shell {0.3, 1.0} x seeds
# {0,1} = 4 runs @ 500k. WIN = eff_rank HOLDS at 2 (cond bounded, no collapse) and the
# policy climbs/holds toward +569. Watch frame/shell + latent/gram_eff_rank (->2) +
# imagine/horizon==15. Fresh 'cf10-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
SHELLS="${SHELLS:-0.3 1.0}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# fixed horizon + rank-2 operator + light clamp + the two-sided shell (w_shell per arm).
# All other frame terms OFF (dissipativity/balance/head dropped); energy head unused.
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.0 model.dual_latent.rank2_frame.w_lyap=0.0 \
model.dual_latent.rank2_frame.balance=false model.dual_latent.rank2_frame.shell_target=1.0 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false smoothing.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for ws in $SHELLS; do
	for seed in $SEEDS; do
		throttle
		tag="cf10-shell${ws}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.w_shell="$ws" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf10 runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf10 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
