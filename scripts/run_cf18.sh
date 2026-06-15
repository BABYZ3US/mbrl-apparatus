#!/usr/bin/env bash
# FLOOR-WALL SHAPE at the CAS optimum (PM 2026-06-15). The CAS (scripts/cas_spectral_optimum.py)
# showed WHY cond is stuck at 1e7-1e12: the relu² floor wall's lift 2·w_band·(floor−λ) VANISHES
# at the floor, so the deadest mode sinks and cond blows up when drift > 2·w_band·floor. Fixes:
# relu1 (constant lift, hard edge) or softplus (smooth logistic lift, no kink) — both bind ⇒
# cond → ceiling/floor. cf18 tests all three on the BAND-ALONE winner (cf14 +1244 family,
# w_compress=0, gate off) at the CAS-optimal point: ceiling=1.0, floor=0.1 (cond target 10),
# w_band=5 (active modes within ~10% of ceiling), RATCHET horizon (h_min=15 -> always >=15,
# the gap fixed), lambda_min=1e-4 (base default). latent 16, hidden 256.
#   SWEEP band_floor_shape {relu2(baseline), relu1, softplus} x seeds {0,1} = 6 arms @ 500k.
# WIN = relu1/softplus realize cond ~10 (vs relu2's 1e7-1e12) AND the band-alone climb holds
# (toward +1244) cleaner. Open question (PM): is the relu HARD edge (zero force inside the band)
# better than the softplus soft shoulder? WATCH: latent/gram_cond (does it drop to ~10 for
# relu1/softplus?), latent/eig* (full spectrum now logged -> heatmap), eval climb, frame/band.
# Fresh 'cf18-' prefix. (band_floor_beta=20 default for softplus.)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
SHAPES="${SHAPES:-relu2 relu1 softplus}"
WBAND="${WBAND:-5.0}"
LATENT="${LATENT:-16}"
HID="${HID:-256}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# band-ALONE at the CAS optimum: w_band=5, ceiling=1, floor=0.1, ratchet horizon h_min=15.
# floor_shape per-arm. lambda_min=1e-4 inherited from base.yaml.
BASE="model.latent_dim=${LATENT} model.hidden=${HID} model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
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
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false smoothing.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=true imagination.adaptive_horizon.h_min=15 \
imagination.adaptive_horizon.h_max=25 imagination.adaptive_horizon.ratchet=true \
imagination.adaptive_horizon.ratchet_base=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for shape in $SHAPES; do
	for seed in $SEEDS; do
		throttle
		tag="cf18-floor${shape}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (band-alone w_band=${WBAND}, floor_shape=${shape}, ratchet H[15,25]) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.band_floor_shape="$shape" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf18 runs launched (max ${JOBS} concurrent, ${STEPS} steps, floor-shape sweep)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf18 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
