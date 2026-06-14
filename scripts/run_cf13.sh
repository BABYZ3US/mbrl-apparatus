#!/usr/bin/env bash
# REDUCE COMPRESSION — target_rank sweep (PM 2026-06-14). The shell holds eff_rank at 2,
# but rank-2 may be TOO TIGHT to run (cf3 climbed to +569 at k=4 with NO rank constraint,
# i.e. effectively full rank-4). 'Reduce compression' = more ACTIVE modes = higher
# target_rank (NOT just a bigger latent — with the shell, more latent dims at the same
# rank just adds crushed tail). This sweep does both: a BIGGER latent (latent_dim=16, up
# from 4 — gives the higher ranks room) x RANK {2, 4}, where RANK sets BOTH the operator
# rank (dynamics expressiveness) AND the shell target_rank (active latent modes) coherently.
#   rank=2 = the hypothesis (2 active modes / a limit cycle)
#   rank=4 = ~2x the capacity (closer to cf3's full-rank regime that hit +569)
# Base = the cf10 shell stack (fixed H=15 + light clamp + w_shell=1.0). RANK {2,4} x seeds
# {0,1} = 4 runs @ 500k. WIN = a higher rank CLIMBS where rank-2 stalls (rank-2 was too
# tight) — or rank-2 holds, confirming the limit-cycle hypothesis. Watch eval climb +
# latent/gram_eff_rank (~RANK) + horizon=15. Fresh 'cf13-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
RANKS="${RANKS:-2 4}"
LATENT="${LATENT:-16}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf10 shell stack at a BIGGER latent; operator.rank + rank2_frame.target_rank set per arm.
BASE="model.latent_dim=${LATENT} model.dynamics=operator model.operator.structure=normal \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.0 model.dual_latent.rank2_frame.w_lyap=0.0 \
model.dual_latent.rank2_frame.balance=false model.dual_latent.rank2_frame.w_shell=1.0 \
model.dual_latent.rank2_frame.shell_target=1.0 model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false \
smoothing.enabled=false imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 \
optim.skip_nonfinite=true optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for r in $RANKS; do
	for seed in $SEEDS; do
		throttle
		tag="cf13-rank${r}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (latent=${LATENT}, op+shell rank=${r}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.operator.rank="$r" \
			model.dual_latent.rank2_frame.target_rank="$r" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf13 runs launched (max ${JOBS} concurrent, ${STEPS} steps, latent=${LATENT})"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf13 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
