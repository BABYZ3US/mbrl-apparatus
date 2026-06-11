#!/usr/bin/env bash
# One-shot RunPod bootstrap for the apparatus (W3/RunPod, 2026-06-11).
# Usage on a fresh pod:   bash scripts/runpod_setup.sh
# Idempotent; safe to re-run. Expects to run from the repo root.
set -euo pipefail

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

echo "== env sync (lock-exact, fallback re-resolve) =="
uv sync --extra dev --frozen || uv sync --extra dev

echo "== CUDA check =="
if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
	echo "lock wheels are CPU-only here — overlaying CUDA torch (cu124)"
	uv pip install --index-url https://download.pytorch.org/whl/cu124 torch --upgrade
	.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
fi
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "== MuJoCo envs =="
uv pip install -q "gymnasium[mujoco]"   # uv venvs ship no pip (pod run 1)
.venv/bin/python -c "import gymnasium as g; e=g.make('HalfCheetah-v5'); e.reset(seed=0); e.close(); print('HalfCheetah OK')"

echo "== W&B =="
if [ -n "${WANDB_API_KEY:-}" ]; then
	.venv/bin/wandb login --relogin "$WANDB_API_KEY" >/dev/null && echo "wandb: logged in from env"
else
	echo "wandb: set WANDB_API_KEY or run .venv/bin/wandb login manually"
fi

echo "== smoke gate (the rule: never spend GPU before it) =="
.venv/bin/python -m pytest tests/test_smoke.py -q

echo "setup complete — next: bash scripts/run_ensemble_grid.sh"
