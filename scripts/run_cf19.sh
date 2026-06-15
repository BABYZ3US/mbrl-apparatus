#!/usr/bin/env bash
# COMBINED ADAPTIVE LEVERS on the band-alone winner (PM 2026-06-15). cf18 showed sigmoid
# converged quickest then stabilized (relu1 close); cf19 takes the two best floor walls and
# stacks ALL the new performance-adaptive levers to attack the two remaining gaps (late
# convergence + seed spread), over 6 SEEDS so the spread is actually measurable:
#   - near-zero policy init (model.policy_init_scale=0.01): same start every seed.
#   - lambda-gate RATCHET (return_gate.enabled + ratchet): lambda relaxes as return climbs
#     and LOCKS — can't re-tighten (horizon-ratchet logic for lambda; needs the gate on).
#   - reward_adapt: entropy_anneal (bonus*=1-rf) + entropy_floor (H>=h_high*(1-rf): explore
#     low-return, commit as it rises) + actor_clip_adapt (grad-clip tightens as return rises).
# Base = the band-alone +1344 family (w_compress=0, w_band=5, ceiling=1, floor=0.1, RATCHET
# horizon h_min=15, hidden=256, latent 16). SWEEP band_floor_shape {relu1, sigmoid} x seeds
# {0..5} = 12 arms @ 500k. WIN = the seed SPREAD tightens (fewer dead seeds vs cf17's 2/3)
# AND/OR a cleaner/higher climb. WATCH: per shape the spread of peak eval across 6 seeds
# (vs cf18's same shapes WITHOUT levers), policy/entropy (held up by the floor early, fading),
# penalty/lambda (ratcheted down, not re-tightening), latent/gram_cond (floor shape binds it).
# Fresh 'cf19-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
SHAPES="${SHAPES:-relu1 sigmoid}"
WBAND="${WBAND:-5.0}"
LATENT="${LATENT:-16}"
HID="${HID:-256}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# band-alone base + ALL combined adaptive levers. floor_shape per-arm.
BASE="model.latent_dim=${LATENT} model.hidden=${HID} model.policy_init_scale=0.01 \
model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.w_compress=0.0 model.dual_latent.rank2_frame.w_band=${WBAND} \
model.dual_latent.rank2_frame.band_ceiling=1.0 model.dual_latent.rank2_frame.band_floor=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=true penalty.return_gate.ratchet=true \
penalty.return_gate.shape=quadratic penalty.return_gate.mid=0.0 penalty.return_gate.scale=100.0 \
penalty.return_gate.floor=0.1 reward_adapt.mid=0.0 reward_adapt.scale=1000.0 \
reward_adapt.entropy_anneal=true reward_adapt.entropy_floor.enabled=true \
reward_adapt.entropy_floor.h_high=1.0 reward_adapt.entropy_floor.coef=0.01 \
reward_adapt.actor_clip_adapt.enabled=true reward_adapt.actor_clip_adapt.min_frac=0.1 \
smoothing.enabled=false imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 \
optim.skip_nonfinite=true optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=true \
imagination.adaptive_horizon.h_min=15 imagination.adaptive_horizon.h_max=25 \
imagination.adaptive_horizon.ratchet=true imagination.adaptive_horizon.ratchet_base=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for shape in $SHAPES; do
	for seed in $SEEDS; do
		throttle
		tag="cf19-${shape}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (band-alone + combined levers, floor=${shape}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.band_floor_shape="$shape" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf19 runs launched (max ${JOBS} concurrent, ${STEPS} steps, combined levers x {relu1,sigmoid} x 6 seeds)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf19 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
