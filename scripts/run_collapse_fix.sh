#!/usr/bin/env bash
# COLLAPSE-FIX (PM 2026-06-13): the campaign winner is the dual-latent TWIN
# (smooth-d / rough-p), DreamSmooth OFF, λ=1e-3 — it peaked +569/+353 but every
# run peaks-then-COLLAPSES (value/imagination instability, orthogonal to λ). This
# run applies the two levers the campaign says will turn a +569 PEAK into a +569
# RESULT, on that winning config:
#   - best-checkpoint retention (train.py now saves ckpt_best.pt on every new-best
#     eval) — keeps the peak policy even if training later collapses;
#   - imagination-alignment stabilizer (imagination.align_weight=1.0, 2507.16450)
#     — pulls imagined latents to the encoder manifold to HOLD the gait.
#
# Matrix: TWIN + DreamSmooth OFF + align=1.0 + best-ckpt, 2 seeds x
#   lambda {1e-3, 5e-4, 1e-4} x  GATE-CURVATURE {gquad = quadratic (p=2, convex),
#                                                 gcube = cuberoot (p=1/3, concave)}
# = 12 runs @ 500k HalfCheetah. BOTH gates are full-range reward-tied (enabled,
# mid=0, scale=400 ≈ the twin's return span) — they relax λ as return climbs but
# with OPPOSITE curvature (quad holds λ high then drops late; cuberoot drops λ fast
# then flattens). Tests whether the gate's curvature/degree matters. The cuberoot
# TIME-DECAY schedule stays on underneath both. Judge by PEAK and FINAL/peak ratio.
#
# Usage: bash scripts/run_collapse_fix.sh   (STEPS/JOBS/NGPU/SEEDS/LAMS/GATES tunable)
set -uo pipefail
cd "$(dirname "$0")/.."

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$((2 * NGPU))}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-0 1}"
LAMS="${LAMS:-1e-3 5e-4 1e-4}"
GATES="${GATES:-gquad gcube}"
PY=".venv/bin/python"
mkdir -p results/gridlogs
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .wandb_key ]; then
	export WANDB_API_KEY="$(cat .wandb_key)"
fi

# the campaign-winning config + the collapse fix
BASE="model.dynamics=operator model.operator.structure=normal model.operator.w_normal=0.05 \
model.operator.w_smooth=0.1 model.operator.w_radius=0.1 model.dual_latent.enabled=true \
model.dual_latent.mode=twin model.dual_latent.couple_weight=0.1 model.dual_latent.smooth_p=false \
model.dual_latent.penalize_reward=true model.reward_heads=1 penalty.form=frobenius \
env=halfcheetah training.total_env_steps=${STEPS} logging.video.enabled=false \
penalty.auto_dose.enabled=false penalty.schedule.kind=cuberoot smoothing.enabled=false \
imagination.align_weight=1.0"

GATE_ON="penalty.return_gate.enabled=true penalty.return_gate.mid=0.0 penalty.return_gate.scale=400.0"
gate_cfg() {   # $1 = gquad | gcube  (both full-range reward-tied gates; opposite curvature)
	case "$1" in
		gquad) echo "$GATE_ON penalty.return_gate.shape=quadratic" ;;
		gcube) echo "$GATE_ON penalty.return_gate.shape=cuberoot" ;;
	esac
}
lam_tag() { echo "$1" | sed 's/e-/em/; s/\.//g'; }   # 1e-3->1em3, 5e-4->5em4

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done; }

pids=(); idx=0
for lam in $LAMS; do
	for gate in $GATES; do
		for seed in $SEEDS; do
			throttle
			tag="cfix-l$(lam_tag "$lam")-${gate}-s${seed}"
			log="results/gridlogs/${tag}.log"
			gpu=$((idx % NGPU)); idx=$((idx + 1))
			echo "launching ${tag} on GPU ${gpu} -> ${log}"
			OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
			$PY scripts/train.py $BASE $(gate_cfg "$gate") penalty.schedule.lam0="$lam" \
				seed="$seed" experiment.name="$tag" hydra.run.dir="outputs/${tag}" \
				> "$log" 2>&1 &
			pids+=($!)
			sleep 2
		done
	done
done

n=${#pids[@]}
echo "all ${n} collapse-fix runs launched (max ${JOBS} concurrent, ${STEPS} steps)"
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
echo "collapse-fix done: $((n - fail))/${n} succeeded"
[ "$fail" -eq 0 ]
