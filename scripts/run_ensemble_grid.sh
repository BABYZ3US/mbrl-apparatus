#!/usr/bin/env bash
# PLAN row 11 validation — the ensemble-pessimism grid (first RunPod campaign).
#
#   pessimism {0.0, 0.25, 0.5, 1.0} x seeds {0,1,2}, members=5 (the yaml),
#   HalfCheetah-v5, 100k env steps each = 12 runs, throttled to $JOBS at once
#   (3090: GPU is ample for these nets; 32 vCPUs carry the MuJoCo stepping).
#
# Usage:  bash scripts/run_ensemble_grid.sh            # the full grid
#         JOBS=3 bash scripts/run_ensemble_grid.sh     # gentler
#         SHAKEDOWN=1 bash scripts/run_ensemble_grid.sh  # 2-min Pendulum first
#
# Every run: distinct experiment.name => its own W&B group + run mirror;
# checkpoints resume bitwise if the pod restarts. Logs: results/gridlogs/.
set -uo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-4}"
PY=".venv/bin/python"
# container restarts wipe ~/.netrc — the volume-persisted key file restores auth
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi
mkdir -p results/gridlogs

if [ "${SHAKEDOWN:-0}" = "1" ]; then
	echo "== shakedown: champion Pendulum (minutes, ~free) =="
	$PY scripts/train.py +experiment=champion env=pendulum seed=0 \
		training.total_env_steps=5000 || { echo "shakedown FAILED"; exit 1; }
	echo "== shakedown OK =="
fi

pids=()
throttle() {
	while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
}

for p in 0.0 0.25 0.5 1.0; do
	tag="ens-p$(echo "$p" | tr -d '.')"
	for s in 0 1 2; do
		throttle
		log="results/gridlogs/${tag}-s${s}.log"
		echo "launching ${tag} seed ${s} -> ${log}"
		OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
		$PY scripts/train.py +experiment=ensemble env=halfcheetah seed="$s" \
			experiment.name="$tag" algo.ensemble_pessimism="$p" \
			hydra.run.dir="outputs/${tag}-s${s}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2   # stagger W&B inits
	done
done

echo "all 12 arms launched (max ${JOBS} concurrent) — waiting…"
fail=0
for pid in "${pids[@]}"; do
	wait "$pid" || fail=$((fail + 1))
done
echo "grid done: $((12 - fail))/12 succeeded"
[ "$fail" -eq 0 ]
