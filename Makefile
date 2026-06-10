# Common entry points. All targets are safe to re-run (benchmarks are
# cell-resumable; tests are hermetic).
.PHONY: test test-all lint ci hooks-install audit bench bridge recipe angle dashboard figures spectral-rl status ledger-check image seal-check lock sync clean

test:              ## fast set (excludes @slow integration tests)
	python -m pytest tests/ -q -m "not slow"

ci:                ## run the CI gate locally (lint + fast tests + seal-check), mirrors .github/workflows/ci.yml
	@command -v uvx >/dev/null 2>&1 && uvx ruff@0.15.16 check . \
	    || python -m pyflakes src/mbrl scripts
	python -m pytest tests/ -q -m "not slow"
	$(MAKE) seal-check

hooks-install:     ## install git pre-push hook (core.hooksPath=hooks) for this repo
	bash scripts/install_hooks.sh

audit:             ## run the nightly deterministic codebase audit over both repos (appends to math/docs)
	bash ../scripts/nightly_codebase_audit.sh

test-all:
	python -m pytest tests/ -q

lint:
	python -m pyflakes src/mbrl scripts || true

bench:            ## supervised spectral benchmark (3 arms, chunked)
	python scripts/spectral_benchmark.py --budget 60

bridge:           ## bridge experiment cells (ledger runs 1-2)
	python scripts/bridge_experiment.py --budget 60

recipe:           ## recipe head-to-head incl. calibrated ladders (runs 3-5)
	python scripts/bridge_experiment.py --recipe --budget 60

angle:            ## transversality angle sweep (run 2's sigma_w sweep)
	python scripts/bridge_experiment.py --angle-sweep

dashboard:
	python scripts/make_dashboard.py

status:           ## what is running / finished / missing
	python scripts/status.py

ledger-check:     ## recompute headline numbers, verify the ledger quotes them
	python scripts/ledger_check.py

figures:
	python scripts/make_figures.py

spectral-rl:      ## the 5-arm spectral RL validation (GPU recommended)
	python scripts/parallel_runs.py --preset gpu_spectral \
	    --overrides env=halfcheetah --seeds 0 1 2

lock:             ## re-resolve deps + regenerate the sealed export (DELIBERATE
	uv lock        ## act: rerun test-all + the mlp-recipe anchor before committing)
	uv export --extra mujoco --no-default-groups --no-emit-project \
	    --no-hashes --format requirements.txt -o requirements-core.txt

sync:             ## local dev env from the lock (exact, incl. analysis+dev)
	uv sync --extra mujoco --extra analysis --extra dev

image:            ## sealed training image, tagged with the current sha
	docker build -t mbrl-curvature:$$(git rev-parse --short HEAD) .

seal-check:       ## training entrypoints must import only core deps
	@! grep -nE "^(import|from) (matplotlib|seaborn|tensorboard|tqdm|plotly|umap)" \
	    scripts/train.py scripts/train_multitask.py scripts/collect.py \
	    src/mbrl/training/*.py src/mbrl/models/*.py src/mbrl/regularization/*.py \
	    && echo "seal OK: training code imports core deps only"

clean:            ## scratch outputs only — never touches results/ or checkpoints/
	rm -rf outputs/parallel .pytest_cache pytest-cache-files-*
