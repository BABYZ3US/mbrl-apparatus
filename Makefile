# Common entry points. All targets are safe to re-run (benchmarks are
# cell-resumable; tests are hermetic).
.PHONY: test test-all lint bench bridge recipe angle dashboard figures spectral-rl status ledger-check clean

test:              ## fast set (excludes @slow integration tests)
	python -m pytest tests/ -q -m "not slow"

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
	python scripts/parallel_runs.py --preset colab_spectral \
	    --overrides env=halfcheetah --seeds 0 1 2

clean:            ## scratch outputs only — never touches results/ or checkpoints/
	rm -rf outputs/parallel .pytest_cache pytest-cache-files-*
