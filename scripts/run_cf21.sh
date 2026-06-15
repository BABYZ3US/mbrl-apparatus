#!/usr/bin/env bash
# THE VARIANCE BOUND (PM 2026-06-15): replace the failed SOFT entropy floor with a reward-
# adaptive HARD log_std floor. cf19 (relu) and cf20 (sigmoid) both showed a SOFT penalty on the
# entropy ESTIMATE can't stop a source-level collapse (σ→0, policy mean saturates tanh, log-prob
# blows up); cf20 still had a violently unstable spread (s1 +705 then -154, 4/6 flat). The fix is
# STRUCTURAL: clamp log_std from below so σ has a hard minimum and the policy CANNOT collapse to a
# deterministic point mass — which keeps COLLECTION exploratory (act()/eval sample stochastically),
# so no seed freezes before it finds the climb. Driven by reward: floor HIGH (explore) at low
# return, relaxing to `lo` (commit) as return climbs — the relaxation is what recovers near-
# deterministic peak return under the stochastic eval. NO policy-mean bound (redundant: a floored σ
# keeps the policy from committing into the saturated corner in the first place).
#   - reward_adapt.logstd_floor {enabled, hi=-1.0 (σ≥0.37 explore), lo=-4.0 (σ≥0.018 commit)}.
#   - entropy_floor DISABLED (the soft floor it replaces); entropy_anneal + actor_clip_adapt KEPT.
# Base = the cf20 combined-lever family (band-alone +1344 + near-zero init + leaky_relu gate +
# lambda-gate ratchet + ratchet horizon, band floor=sigmoid). 6 SEEDS {0..5}=6 arms @ 500k. WIN =
# the seed SPREAD tightens (more than 1-2/6 climb) AND the climb is STABLE (no s1-style +705→-154
# whipsaw). Fresh 'cf21-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
SHAPES="${SHAPES:-sigmoid}"          # band_floor_shape
LSF_HI="${LSF_HI:--1.0}"             # log_std floor at rf=0 (explore)
LSF_LO="${LSF_LO:--4.0}"             # log_std floor at rf=1 (commit)
WBAND="${WBAND:-5.0}"
LATENT="${LATENT:-16}"
HID="${HID:-256}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf20 base, but entropy_floor OFF + the reward-adaptive HARD log_std floor ON. floor_shape per-arm.
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
reward_adapt.entropy_anneal=true reward_adapt.entropy_floor.enabled=false \
reward_adapt.logstd_floor.enabled=true reward_adapt.logstd_floor.hi=${LSF_HI} reward_adapt.logstd_floor.lo=${LSF_LO} \
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
		tag="cf21-${shape}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (reward-adaptive HARD log_std floor [${LSF_HI},${LSF_LO}], band=${shape}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.band_floor_shape="$shape" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf21 runs launched (max ${JOBS} concurrent, ${STEPS} steps, reward-adaptive hard log_std floor)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf21 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
