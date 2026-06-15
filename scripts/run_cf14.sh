#!/usr/bin/env bash
# SPECTRAL BAND — bound the floor and ceiling, let the rank EMERGE (PM 2026-06-14).
# The lesson of cf10-cf13: (1) demanding a hand-set rank is the wrong lever — forced
# rank-2 (cf10, +95) AND forced rank-4 (cf13, +67) both peaked-then-collapsed; and
# (2) a ONE-SIDED barrier alone is vacuous — the energy/dissipativity barrier only saw
# growth (inert) and the log-det barrier alone (cf12) only pushes eigenvalues UP, never
# caps them, never confines. cf14 drops BOTH mistakes: NO rank enforcement (operator
# rank=0 full, no shell, no target_rank) and a TWO-SIDED spectral BAND that bounds EVERY
# Gram eigenvalue between a hard floor and a hard ceiling, FREE in between:
#     band = Σ_i relu(λ_i − ceiling)²  +  Σ_i relu(floor − λ_i)²
# Nothing collapses to 0 (floor wall), nothing runs away (ceiling wall) -> cond(G) ≤
# ceiling/floor is bounded -> and the NUMBER of active modes (the effective rank) is
# chosen by the task inside the band, not imposed. Bigger latent (16) gives the rank room
# to emerge. Sweep w_band {1.0, 5.0} (wall STRENGTH — the floor wall relu(floor−λ)² is
# gentle, grad ~0.2 at λ→0, so 'how HARD is the hard floor' is the live question) x seeds
# {0,1} = 4 runs @ 500k, band fixed at [0.1, 1.0] (cond ≤ 10). cf10 stabilization stack
# (fixed H=15 + light clamp + lam=1e-3). WATCH: latent/gram_eff_rank = the EMERGENT rank
# (does it settle on its own?), latent/gram_cond (bounded ~ceiling/floor?), frame/band,
# loss/policy (does the band AVOID the policy-side collapse cf10-cf13 all hit?), and does
# eval climb past +95. Fresh 'cf14-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
WBANDS="${WBANDS:-1.0 5.0}"
FLOOR="${FLOOR:-0.1}"
LATENT="${LATENT:-16}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# NO rank enforcement (operator.rank=0 full, w_shell=0, no target_rank). The ONLY
# structural pressure is the two-sided band on the Gram spectrum. ceiling=1.0, floor per-run.
BASE="model.latent_dim=${LATENT} model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.band_ceiling=1.0 model.dual_latent.rank2_frame.band_floor=${FLOOR} \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false smoothing.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for wb in $WBANDS; do
	for seed in $SEEDS; do
		throttle
		tag="cf14-band${wb}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (band [${FLOOR},1.0] w_band=${wb}, NO rank demand, latent=${LATENT}) on GPU ${gpu} -> ${log}"
		# ${EXTRA} = extra Hydra overrides (e.g. RESUME: EXTRA='~model.dual_latent.rank2_frame.w_compress'
		# deletes the post-launch w_compress key so the resolved-config hash matches the original
		# pre-w_compress cf14 lineage and checkpoint.resume=auto picks up the existing checkpoints).
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE ${EXTRA:-} model.dual_latent.rank2_frame.w_band="$wb" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf14 runs launched (max ${JOBS} concurrent, ${STEPS} steps, band-bounded emergent rank)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf14 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
