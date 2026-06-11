#!/usr/bin/env bash
# CAMPAIGN 2 — ensemble_spectral (the champion bridge): does the epistemic
# discount's direction replicate on the SPECTRAL reward stack?
#   p in {0.0, 1.0 (campaign-1 winner)} x seeds {0,1,2} = 6 arms,
#   HalfCheetah-v5 @100k. The spectral path has NO symexp, so the variance
#   metric is de-confounded here by construction.
# Usage: bash scripts/run_campaign2.sh    (JOBS=n to override; default 4)
set -uo pipefail
cd "$(dirname "$0")/.."

JOBS="${JOBS:-4}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

pids=()
throttle() {
	while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
}

for p in 0.0 1.0; do
	tag="ens2-p$(echo "$p" | tr -d '.')"
	for s in 0 1 2; do
		throttle
		log="results/gridlogs/${tag}-s${s}.log"
		echo "launching ${tag} seed ${s} -> ${log}"
		OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
		$PY scripts/train.py +experiment=ensemble_spectral env=halfcheetah seed="$s" \
			experiment.name="$tag" algo.ensemble_pessimism="$p" \
			hydra.run.dir="outputs/${tag}-s${s}" \
			> "$log" 2>&1 &
		pids+=($!)
		sleep 2
	done
done

echo "all 6 arms launched (max ${JOBS} concurrent) — waiting…"
fail=0
for pid in "${pids[@]}"; do
	wait "$pid" || fail=$((fail + 1))
done
echo "campaign 2 done: $((6 - fail))/6 succeeded"
[ "$fail" -eq 0 ]
