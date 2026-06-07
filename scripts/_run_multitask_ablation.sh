#!/usr/bin/env bash
# Driver: regularized arm (lam0=1e-3) vs ablation arm (lam0=0), seeds 0..2.
# Ablation arm gets a distinct experiment.name so JSONL mirrors / checkpoints
# don't collide with the regularized arm (same family+seed otherwise).
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=offline
PY=.venv/bin/python
LOG=results/ablation_driver.log
: > "$LOG"

for SEED in 0 1 2; do
  echo "=== regularized seed=$SEED $(date) ===" | tee -a "$LOG"
  $PY scripts/train_multitask.py device=cpu seed=$SEED \
      checkpoint.push_wandb=false checkpoint.resume=none \
      hydra.run.dir=outputs/multitask_reg_s$SEED >>"$LOG" 2>&1

  echo "=== lam0=0 seed=$SEED $(date) ===" | tee -a "$LOG"
  $PY scripts/train_multitask.py device=cpu seed=$SEED \
      experiment.name=multitask_lam0 penalty.schedule.lam0=0 \
      checkpoint.push_wandb=false checkpoint.resume=none \
      hydra.run.dir=outputs/multitask_lam0_s$SEED >>"$LOG" 2>&1
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
