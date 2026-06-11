#!/usr/bin/env bash
# LATENT-AS-CHANNEL / IB FRONTIER (2026-06-11): test whether managing the
# representation channel's RATE governs control performance.
# (docs/channel_capacity_formalization_2026-06-11.md)
#
# The VAE encoder makes the latent an explicit noisy channel q_φ(z|x)=N(μ,σ²);
# β (the KL weight) is its rate knob: small β = high-rate channel (z carries
# everything), large β = low-rate channel (z → prior, carries nothing). Sweeping
# β traces the rate–distortion / information-bottleneck frontier. The IB
# prediction: control return is FLAT at high rate, then COLLAPSES once the rate
# drops below the task's minimal sufficient bits — the elbow is the optimal
# operating point, and (the win condition) it should match or beat the
# deterministic champion at a fraction of the rate.
#
# Stack: champion (spectral reward = the capacity-achieving read-out; gaussian
# dynamics) + VAE encoder. encoder_aux + recon both ground z (Run 10 parity).
# Logged per step: info/rate_nats (= E KL ≥ I(x;z)), info/task_reward_nats,
# info/task_dyn_nats, info/ib_efficiency.
#
#   arms: ctl (deterministic, rate=∞ reference) + β∈{0,1e-4,1e-3,1e-2,1e-1}
#         × seeds {0,1,2} = 18 arms @250k.
# Verdict = plot eval/return vs info/rate_nats (the IB curve) across β; find the
# elbow; does the IB-optimal β beat ctl?
#
# Usage: bash scripts/run_ib_frontier.sh   (STEPS/SEEDS/JOBS/NGPU tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((4 * NGPU))}"
STEPS="${STEPS:-250000}"
SEEDS="${SEEDS:-0 1 2}"
# NOISE = obs-channel input noise σ. 0 = clean (representation channel is NOT a
# bottleneck → frontier likely flat); >0 (e.g. 0.5) manufactures a real channel
# bottleneck → the IB elbow should appear (the R14 denoising regime).
NOISE="${NOISE:-0.0}"
PY=".venv/bin/python"
arm_idx=0
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# arm -> (tag, encoder/beta override). ctl = deterministic MLP encoder (no rate).
arm_override() {
	case "$1" in
		ctl)   echo "model.encoder=mlp" ;;
		b00)   echo "model.encoder=vae model.vae.beta=0.0" ;;
		b1em4) echo "model.encoder=vae model.vae.beta=1e-4" ;;
		b1em3) echo "model.encoder=vae model.vae.beta=1e-3" ;;
		b1em2) echo "model.encoder=vae model.vae.beta=1e-2" ;;
		b1em1) echo "model.encoder=vae model.vae.beta=1e-1" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=()
for arm in ctl b00 b1em4 b1em3 b1em2 b1em1; do
	ov="$(arm_override "$arm")"
	for s in $SEEDS; do
		throttle
		nz=$(echo "$NOISE" | tr -d '.')
		tag="ib-n${nz}-${arm}"
		log="results/gridlogs/${tag}-s${s}.log"
		gpu=$((arm_idx % NGPU)); arm_idx=$((arm_idx + 1))
		echo "launching ${tag} seed ${s} (${STEPS} steps) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py +experiment=champion env=halfcheetah seed="$s" \
			experiment.name="$tag" \
			$ov \
			+env.obs_noise="$NOISE" \
			training.total_env_steps="$STEPS" \
			logging.video.enabled=false \
			hydra.run.dir="outputs/${tag}-s${s}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} arms launched (max ${JOBS} concurrent across ${NGPU} GPU(s), ${STEPS} steps) — waiting…"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "ib-frontier done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
