"""Launch a grid of training runs in parallel, one process per core.

All runs log to the SAME W&B project (mbrl-curvature) as separate runs; the
dashboard aggregates them by `group` (= experiment.name), so seeds within an
arm form one mean±CI band and different arms sit side by side in the same
panels. Local JSONL mirrors land in results/runs/ as usual, so
`make_figures.py` aggregates the same way offline.

Usage:
  # the multitask science grid: 3 arms x seeds, 6 processes
  python scripts/parallel_runs.py --preset multitask_ablation --seeds 0 1 --jobs 6

  # generic: any script + overrides, fanned over seeds
  python scripts/parallel_runs.py --script train.py --seeds 0 1 2 3 4 \\
      --overrides env=pendulum experiment.name=baseline --jobs 5

  # see the commands without running
  python scripts/parallel_runs.py --preset multitask_ablation --seeds 0 1 --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "results" / "logs" / "parallel"

# Presets live in configs/presets.yaml (improvement plan #9): data, not code.
# Each arm = name + optional use: [constants...] (prepended in order) + overrides.
import yaml

with open(ROOT / "configs" / "presets.yaml") as _f:
    _REG = yaml.safe_load(_f)
_CONST = _REG.get("constants", {})
PRESETS = {}
for _name, _spec in _REG["presets"].items():
    _arms = []
    for _arm in _spec["arms"]:
        _ovr = []
        for _u in _arm.get("use", []):
            _ovr += list(_CONST[_u])
        _ovr += list(_arm.get("overrides", []))
        _arms.append((_arm["name"], _ovr))
    PRESETS[_name] = (_spec["script"], _arms)


def build_commands(args) -> list[tuple[str, list[str]]]:
    jobs = []
    if args.preset:
        script, arms = PRESETS[args.preset]
        for arm, ovr in arms:
            for s in args.seeds:
                name = f"{arm}-s{s}"
                jobs.append((name, [sys.executable, str(ROOT / "scripts" / script),
                                    *ovr, *args.overrides,
                                    f"experiment.name={arm}", f"seed={s}",
                                    f"hydra.run.dir=outputs/parallel/{name}"]))
    else:
        for s in args.seeds:
            name = f"run-s{s}"
            jobs.append((name, [sys.executable, str(ROOT / "scripts" / args.script),
                                *args.overrides, f"seed={s}",
                                f"hydra.run.dir=outputs/parallel/{name}"]))
    return jobs


def run_one(name: str, cmd: list[str]) -> tuple[str, int]:
    """One training process. Single-threaded math libs: parallelism comes from
    process count, not intra-op threads (avoids 8 runs x 8 threads thrash)."""
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1"}
    log = LOGDIR / f"{name}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=ROOT, env=env).returncode
    return name, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--script", default="train_multitask.py",
                   help="used when no --preset is given")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--overrides", nargs="*", default=[],
                   help="extra hydra overrides applied to every run")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                   help="concurrent processes (default: cores - 1)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cmds = build_commands(args)
    print(f"{len(cmds)} runs, {args.jobs} at a time; logs -> {LOGDIR}/")
    for name, cmd in cmds:
        print(f"  {name}: {' '.join(cmd[1:])}")
    if args.dry_run:
        return
    LOGDIR.mkdir(parents=True, exist_ok=True)

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs)(
        delayed(run_one)(name, cmd) for name, cmd in cmds)

    failed = [(n, rc) for n, rc in results if rc != 0]
    for n, rc in results:
        print(f"  {'OK  ' if rc == 0 else 'FAIL'} {n}" + ("" if rc == 0 else
              f"  (exit {rc}; see {LOGDIR}/{n}.log)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
