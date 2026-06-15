#!/usr/bin/env bash
# COMPRESSION x GATE grid (PM 2026-06-14). cf15 only ever ran band+gate (wc=0.0, failed)
# and one band+compress+gate point (wc=0.1); we never swept compression, and never got a
# clean GATE-OFF negative. cf16 does both: on top of the band (w_band=5, [0.1,1.0], latent
# 16, op rank=0), sweep w_compress {0.03, 0.1, 0.3} x return_gate {on, off}. gate-OFF is the
# clean control (isolates whether the gate does anything); the w_compress sweep finds whether
# ANY compression strength helps vs the band-alone winner (cf14, +1244). Fixed H=15 here so
# the only moving parts are compression + gate (the horizon ratchet is the SEPARATE cf17
# band-alone track). cf10 stabilization stack (light clamp, lam0=1e-3). 3 wc x 2 gate x
# SEEDS{0} = 6 arms @ 500k by default (bump SEEDS for robustness — seeds vary a LOT here:
# cf14 s0=+535 vs s1=+1244). TARGET METRIC to watch: frame/compress + frame/band DECREASING
# into a low-amplitude high-frequency jitter near 0 (spectrum settled in the band). WATCH:
# eval (past +95? toward cf14's +1244? or worse, confirming the additions hurt), eff_rank
# (compression lowering it from band-alone ~12?), penalty/return_gate (relaxing as return
# rises, on the gate-on arms), loss/policy. Fresh 'cf16-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1 2}"
WCS="${WCS:-0.03 0.1 0.3}"
GATES="${GATES:-on off}"
LATENT="${LATENT:-16}"
HID="${HID:-256}"          # net width
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# band (w_band=5, [0.1,1.0]) + fixed H=15. w_compress + return_gate.enabled set per-arm.
BASE="model.latent_dim=${LATENT} model.hidden=${HID} model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.w_band=5.0 model.dual_latent.rank2_frame.band_ceiling=1.0 \
model.dual_latent.rank2_frame.band_floor=0.1 model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 penalty.return_gate.shape=quadratic \
penalty.return_gate.mid=0.0 penalty.return_gate.scale=100.0 penalty.return_gate.floor=0.1 \
smoothing.enabled=false imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 \
optim.skip_nonfinite=true optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for wc in $WCS; do
	for g in $GATES; do
		[ "$g" = "on" ] && gate=true || gate=false
		for seed in $SEEDS; do
			throttle
			tag="cf16-wc${wc}-gate${g}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} (band w5 + compress=${wc} + gate=${g}) on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE model.dual_latent.rank2_frame.w_compress="$wc" \
				penalty.return_gate.enabled="$gate" \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} cf16 runs launched (max ${JOBS} concurrent, ${STEPS} steps, compression x gate grid)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf16 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
