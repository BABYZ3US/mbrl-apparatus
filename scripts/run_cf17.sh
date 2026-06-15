#!/usr/bin/env bash
# BAND-ALONE + RATCHET HORIZON + energy-bound sweep (PM 2026-06-14). Band-alone IS the winner
# (cf14-band1.0 hit +1244 resumed, ~2x the all-time best) with NO compression and NO gate.
# cf17 keeps it pure (w_compress=0, return_gate off) and changes two things:
#  (1) HORIZON: drop the fixed H=15; use the adaptive horizon with the new RATCHET floor
#      (imagination.adaptive_horizon.ratchet) — H roams in [h_min=15, h_max=25] but, once it
#      reaches ratchet_base=15, locks a running-max floor: it can climb but NEVER fall below
#      its peak. A relu/step floor that enforces stability at peak convergence and forecloses
#      the penalty-spike -> horizon-collapse failure (cf6/cf7). Always >=15 (>1), monotone up.
#  (2) BAND PARAMETERS: sweep band_ceiling {0.99, 1.0} (the '0.99/1 hard energy bound' — does
#      pinning eigenvalues just under 1 give the cleaner low-amplitude-near-0 band residual?)
#      x w_band {1.0, 2.0} (1.0 = the cf14 winner; 2.0 = a harder wall). band_floor=0.1.
# latent 16, op rank=0, cf10 stabilization stack (light clamp, lam0=1e-3). 2 ceil x 2 w_band
# x SEEDS{0} = 4 arms @ 500k by default (bump SEEDS — seeds vary a LOT). TARGET METRIC:
# frame/band DECREASING into a low-amplitude high-frequency jitter near 0 (spectrum parked at
# the ceiling). WATCH: eval (hold/climb toward +1244 like cf14, now with a ratcheted horizon?),
# imagine/horizon (climbs to ~25 and HOLDS, never collapses?), eff_rank (~12 emergent),
# frame/band + latent/gram_cond. Fresh 'cf17-' prefix. (cf14 winner used fixed H=15;
# cf17-wb1.0-ceil1.0 = that winner but with the ratcheted adaptive horizon — direct A/B.)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0}"
WBANDS="${WBANDS:-1.0 2.0}"
CEILINGS="${CEILINGS:-0.99 1.0}"
LATENT="${LATENT:-16}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# band-ALONE (w_compress=0, gate off, no shell/logdet). RATCHET adaptive horizon (NOT fixed
# H=15): h_min=15 floor, h_max=25, ratchet locks the running max. w_band + band_ceiling per-arm.
BASE="model.latent_dim=${LATENT} model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.w_compress=0.0 model.dual_latent.rank2_frame.band_floor=0.1 \
model.reward_heads=1 penalty.form=frobenius env=halfcheetah training.total_env_steps=${STEPS} \
logging.video.enabled=false penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot \
penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=false smoothing.enabled=false \
imagination.reward_clip=1000 imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true \
optim.policy_ema_decay=0.0 imagination.adaptive_horizon.enabled=true imagination.adaptive_horizon.h_min=15 \
imagination.adaptive_horizon.h_max=25 imagination.adaptive_horizon.ratchet=true \
imagination.adaptive_horizon.ratchet_base=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for wb in $WBANDS; do
	for ceil in $CEILINGS; do
		for seed in $SEEDS; do
			throttle
			tag="cf17-wb${wb}-ceil${ceil}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} (band-alone w_band=${wb} ceiling=${ceil} + ratchet H[15,25]) on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE model.dual_latent.rank2_frame.w_band="$wb" \
				model.dual_latent.rank2_frame.band_ceiling="$ceil" \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} cf17 runs launched (max ${JOBS} concurrent, ${STEPS} steps, band-alone + ratchet horizon)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf17 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
