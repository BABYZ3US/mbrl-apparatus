#!/usr/bin/env bash
# THRESHOLD-BARRIER combo (PM 2026-06-15): pair a leaky_relu return-GATE with a sigmoid
# entropy-FLOOR — both "firm during exploration, release/catch at the knee" barriers, one on
# lambda, one on entropy. Motivated by cf19's relu1-s0 collapsing to a deterministic policy
# (entropy -> -21, log_std pinned at the -5 clamp) DESPITE the relu entropy floor: at coef=0.01
# the relu's constant lift was too weak. The two levers here:
#   - leaky_relu GATE (penalty.return_gate.shape=leaky_relu, leak=0.1): hold lambda ~rigid
#     while return < mid (exploring), then release the spectral penalty SHARPLY once the policy
#     crosses into positive return. A threshold on lambda (vs cf19's convex quadratic gate).
#   - sigmoid entropy FLOOR (reward_adapt.entropy_floor.shape=sigmoid, beta=8, coef=0.05): a
#     bounded barrier whose lift PEAKS at the entropy target and vanishes below — catches the
#     entropy AS it tries to cross the floor (prevention), instead of the relu's weak constant
#     lift. coef/beta set so the lift at the target (coef*beta/4 = 0.1) is ~10x cf19's 0.01.
# Everything else = the cf19 combined-lever base (band-alone +1344 family + near-zero init +
# lambda-gate ratchet + entropy_anneal + actor_clip_adapt + ratchet horizon). Band floor =
# sigmoid (cf18's fastest converger). WIN = the deterministic-collapse seeds disappear (every
# seed holds entropy near the floor early) AND the spread tightens vs cf19. Fresh 'cf20-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
SHAPES="${SHAPES:-sigmoid}"          # band_floor_shape; cf19's faster converger
WBAND="${WBAND:-5.0}"
LATENT="${LATENT:-16}"
HID="${HID:-256}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf19 combined-lever base, but: gate shape=leaky_relu (leak 0.1) + entropy floor shape=sigmoid
# (beta 8, coef 0.05 — a real catch at the target). floor_shape (the BAND floor) is per-arm.
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
penalty.return_gate.shape=leaky_relu penalty.return_gate.leak=0.1 penalty.return_gate.mid=0.0 \
penalty.return_gate.scale=100.0 penalty.return_gate.floor=0.1 reward_adapt.mid=0.0 reward_adapt.scale=1000.0 \
reward_adapt.entropy_anneal=true reward_adapt.entropy_floor.enabled=true \
reward_adapt.entropy_floor.shape=sigmoid reward_adapt.entropy_floor.beta=8.0 \
reward_adapt.entropy_floor.h_high=1.0 reward_adapt.entropy_floor.coef=0.05 \
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
		tag="cf20-${shape}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (leaky_relu gate + sigmoid entropy floor, band=${shape}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.band_floor_shape="$shape" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf20 runs launched (max ${JOBS} concurrent, ${STEPS} steps, leaky_relu gate + sigmoid entropy floor)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf20 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
