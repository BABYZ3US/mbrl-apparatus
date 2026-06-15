#!/usr/bin/env bash
# BAND + COMPRESSION + RETURN-GATE (PM 2026-06-14). cf14 diagnosis: the band gives the
# spectrum hard walls but a FREE interior — once the policy lands in a decent basin there
# is no internal gradient to commit, so it drifts (band-alone eff_rank≈12 but peaked low,
# stalled, no momentum). PM's fix — two COMPLEMENTARY pressures:
#   (1) COMPRESSION (rank2_frame.w_compress, Σ√λ nuclear norm): the inward representation
#       pressure the free band lacks — √ concave ⇒ rewards CONCENTRATING variance into the
#       modes that earn it, compressing the rest toward the floor. Rank still emerges, just
#       lower/more decisively. (band floor catches the compressed modes at 0.1, not 0.)
#   (2) RETURN-GATE (penalty.return_gate, quadratic): release the curvature-penalty brake
#       as the policy nears a solution — λ held high while return≤mid(0), relaxed to floor
#       as return climbs toward scale(100). Lets a near-converged policy SHARPEN/commit.
# Band fixed at w_band=5.0 (the harder floor — cf14 w_band=1 left a stray near-zero mode;
# 5 also holds against the compression's downward pull) [0.1,1.0], latent 16, operator
# rank=0 (full). SWEEP w_compress {0.0, 0.1} x seeds {0,1} = 4 runs @ 500k: wc=0.0 ISOLATES
# the gate (band+gate, vs cf14 band-alone), wc=0.1 is the FULL combo (band+gate+compress).
# WATCH: eval (does it now climb PAST cf10's +95 and HOLD, not stall?), latent/gram_eff_rank
# (lower than band-alone's ~12? compression biting?), frame/compress, penalty/return_gate
# (relaxing as return rises?), loss/policy (still finite — no collapse?). Fresh 'cf15-' prefix.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
WCOMPRS="${WCOMPRS:-0.0 0.1}"
WBAND="${WBAND:-5.0}"
LATENT="${LATENT:-16}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# band (w_band=5, [0.1,1.0]) + return_gate ON (quadratic, defaults mid=0/scale=100/floor=0.1).
# NO rank demand (operator.rank=0, no shell/target_rank). w_compress set per-arm.
BASE="model.latent_dim=${LATENT} model.dynamics=operator model.operator.structure=normal model.operator.rank=0 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.w_ortho=0.0 \
model.dual_latent.rank2_frame.w_rank2=0.0 model.dual_latent.rank2_frame.w_dissip=0.0 \
model.dual_latent.rank2_frame.w_lyap=0.0 model.dual_latent.rank2_frame.balance=false \
model.dual_latent.rank2_frame.w_shell=0.0 model.dual_latent.rank2_frame.w_logdet=0.0 \
model.dual_latent.rank2_frame.w_band=${WBAND} model.dual_latent.rank2_frame.band_ceiling=1.0 \
model.dual_latent.rank2_frame.band_floor=0.1 model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 penalty.return_gate.enabled=true \
penalty.return_gate.shape=quadratic penalty.return_gate.mid=0.0 penalty.return_gate.scale=100.0 \
penalty.return_gate.floor=0.1 smoothing.enabled=false imagination.reward_clip=1000 \
imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true optim.policy_ema_decay=0.0 \
imagination.adaptive_horizon.enabled=false imagination.horizon=15"

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for wc in $WCOMPRS; do
	for seed in $SEEDS; do
		throttle
		tag="cf15-wc${wc}-s${seed}"
		log="results/gridlogs/${tag}.log"
		gpu=$((idx % NGPU)); idx=$((idx + 1))
		echo "launching ${tag} (band[0.1,1.0]w${WBAND} + gate + compress=${wc}, latent=${LATENT}) on GPU ${gpu} -> ${log}"
		OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
		$PY scripts/train.py $BASE model.dual_latent.rank2_frame.w_compress="$wc" \
			seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

n=${#pids[@]}
echo "all ${n} cf15 runs launched (max ${JOBS} concurrent, ${STEPS} steps, band+compression+gate)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf15 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
