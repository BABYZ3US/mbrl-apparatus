#!/usr/bin/env bash
# CURVATURE-LAMBDA ABLATION (PM 2026-06-15): drop the reward-Hessian curvature SCHEDULE
# (penalty.schedule.lam0=0) and pin lambda at its minimal floor (penalty.lambda_min=1e-4 =>
# lambda=1e-4 constant), letting the BAND carry the regularization. NOTE the floor is 1e-4,
# NOT exactly 0 — a small nonzero lambda is a valid condition, not a bug; if it climbs that
# refutes 'lambda must be substantial'. Rationale + framing from the loss/RH audit: the band's floor
# wall Φ(f−λ) is the Tikhonov conditioning of the latent Gram — it holds λ_min(G) ≥ f, the
# ridge that keeps the Gram off singular — so "keep the Tikhonov regularization, drop the
# curvature λ" = keep the band, remove ONLY the Hutchinson ∇²R penalty (the founding R4/R16
# mechanism). The curvature λ is already gated to ~1e-3..1e-4 in the winning runs, so this
# tests whether the residual it carries matters at all. lam0=0 is the sanctioned ablation
# (CLAUDE.md). lambda is pinned at the 1e-4 floor (schedule ~0), so the gate/ratchet are ~moot.
#
# WHAT THIS DOES *NOT* FIX: it inherits cf21's policy setup (sigma-only hard log_std floor,
# NO mean bound), so the tanh-MEAN-saturation entropy blowups + shaky seed spread carry over
# untouched — that is the orthogonal mean-bound lever. So READ cf22 as: "is the curvature λ
# vestigial?" — i.e. does the BEST seed still reach +1344-class WITHOUT the ∇²R penalty? — not
# "is the spread fixed?". Everything else = the cf21 base (band floor=sigmoid, reward-adaptive
# log_std variance bound, leaky_relu gate, entropy_anneal, actor_clip_adapt, ratchet horizon,
# near-zero init). 6 SEEDS {0..5}=6 arms @ 500k. Fresh 'cf22-' prefix.
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

# cf21 base, but the curvature penalty is ZEROED: penalty.schedule.lam0=0 AND penalty.lambda_min=0.
# The band (w_band, floor wall) is the Tikhonov conditioner that survives.
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
penalty.schedule.lam0=0 penalty.lambda_min=1e-4 penalty.return_gate.enabled=true penalty.return_gate.ratchet=true \
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
		tag="cf22-${shape}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (curvature lambda DROPPED, band=Tikhonov, band_floor=${shape}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.band_floor_shape="$shape" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf22 runs launched (max ${JOBS} concurrent, ${STEPS} steps, curvature lambda dropped)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf22 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
