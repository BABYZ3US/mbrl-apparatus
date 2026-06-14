#!/usr/bin/env bash
# ENERGY SHELL x GATE x DREAMSMOOTH (PM 2026-06-14). cf10 established the two-sided
# Ginzburg-Landau shell as the first energy that holds eff_rank at 2 (anti-collapse in
# every mode). cf11 layers the two remaining levers onto that base, as a clean 2x2:
#   quadratic gate {off, on} — tie lambda weakly to eval return (relax as return climbs)
#   DreamSmooth  {off, on}    — spread imagined returns over time (variance + horizon-reward)
# Base = the cf10 shell config (fixed H=15 + rank-2 + light clamp + w_shell). WSHELL env
# defaults to 1.0 (the smoke-verified value); set it to cf10's winning w_shell at launch.
# gate {nogate, gate} x dreamsmooth {nods, ds} x seed {0} = 4 runs @ 500k. WIN = eff_rank
# stays ~2 AND the climb holds further toward +569 with the added lever(s). Fresh 'cf11-'.
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0}"
GATES="${GATES:-nogate gate}"
DREAMS="${DREAMS:-nods ds}"
WSHELL="${WSHELL:-1.0}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# cf10 shell base: fixed horizon + rank-2 + light clamp + the two-sided shell (w_shell=WSHELL).
BASE="model.dynamics=operator model.operator.structure=normal model.operator.rank=2 \
model.operator.w_normal=0.05 model.operator.w_smooth=0.1 model.operator.w_radius=0.1 \
model.dual_latent.enabled=true model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 \
model.dual_latent.smooth_p=false model.dual_latent.penalize_reward=true model.dual_latent.radius_p=0.02 \
model.dual_latent.rank2_frame.enabled=true model.dual_latent.rank2_frame.energy_mode=lyapunov \
model.dual_latent.rank2_frame.w_ortho=0.0 model.dual_latent.rank2_frame.w_rank2=0.0 \
model.dual_latent.rank2_frame.w_dissip=0.0 model.dual_latent.rank2_frame.w_lyap=0.0 \
model.dual_latent.rank2_frame.balance=false model.dual_latent.rank2_frame.w_shell=${WSHELL} \
model.dual_latent.rank2_frame.shell_target=1.0 model.reward_heads=1 penalty.form=frobenius env=halfcheetah \
training.total_env_steps=${STEPS} logging.video.enabled=false penalty.auto_dose.enabled=false \
penalty.schedule.kind=cuberoot penalty.schedule.lam0=1e-3 imagination.reward_clip=1000 \
imagination.return_clip=10000 optim.value_clip=1000 optim.skip_nonfinite=true optim.policy_ema_decay=0.0 \
imagination.adaptive_horizon.enabled=false imagination.horizon=15"

gate_cfg() {    # $1 = nogate | gate
	case "$1" in
		nogate) echo "penalty.return_gate.enabled=false" ;;
		gate)   echo "penalty.return_gate.enabled=true penalty.return_gate.shape=quadratic \
penalty.return_gate.mid=0.0 penalty.return_gate.scale=300.0" ;;
	esac
}
ds_cfg() {      # $1 = nods | ds
	case "$1" in
		nods) echo "smoothing.enabled=false" ;;
		ds)   echo "smoothing.enabled=true smoothing.sigma=1.5" ;;
	esac
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for g in $GATES; do
	for ds in $DREAMS; do
		for seed in $SEEDS; do
			throttle
			tag="cf11-${g}-${ds}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} (w_shell=${WSHELL}) on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE $(gate_cfg "$g") $(ds_cfg "$ds") \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} cf11 runs launched (max ${JOBS} concurrent, ${STEPS} steps, w_shell=${WSHELL})"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "cf11 done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
