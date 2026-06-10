"""studio_sweep — expand + validate + (optionally) launch a Studio SweepSpec.

The Studio's `sweep` module emits a SweepSpec — {base_spec, axes, seeds} — over
submit.sweep. This is the apparatus-side launcher for that same shape (and a hand
tool): it expands the cartesian product (mbrl.studio.sweep), runs the spectral
house-rule gate (mbrl.studio.spec_validator), prints the plan, optionally writes
per-arm Hydra experiment yamls, and optionally fans the arms out as training
subprocesses — REFUSING to launch arms that violate the house rules unless
--allow-warnings.

Complements scripts/parallel_runs.py (which fans named PRESETS / a single override
set over seeds). This one takes ARBITRARY axis cross-products from a SweepSpec.

  # plan only (default): print arms + any house-rule warnings
  python scripts/studio_sweep.py --spec sweep.json

  # inline axes instead of a file
  python scripts/studio_sweep.py \\
      --base '{"experiment":{"name":"champ"},"env":{"name":"Pendulum-v1"}}' \\
      --axis penalty.lambda=1e-4,1e-3,1e-2 --seeds 0 1 2

  # write reproducible per-arm experiment yamls, no launch
  python scripts/studio_sweep.py --spec sweep.json --write-yaml

  # actually launch locally (joblib); blocked if any arm warns
  python scripts/studio_sweep.py --spec sweep.json --launch --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mbrl.studio.spec_to_config import write_experiment_yaml  # noqa: E402
from mbrl.studio.sweep import plan_sweep  # noqa: E402

EXP_DIR = ROOT / "results" / "studio" / "experiments"
LOGDIR = ROOT / "results" / "logs" / "sweep"


def _coerce(v: str):
    v = v.strip()
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _parse_axis(s: str) -> dict:
    """'penalty.lambda=1e-4,1e-3' -> {'path':..., 'values':[...]} (typed)."""
    path, sep, vals = s.partition("=")
    if not sep or not path.strip() or not vals.strip():
        raise argparse.ArgumentTypeError(f"--axis must be path=v1,v2,...: {s!r}")
    return {"path": path.strip(), "values": [_coerce(v) for v in vals.split(",")]}


def _load(args):
    if args.spec:
        s = json.loads(Path(args.spec).read_text())
        return s.get("base_spec", {}), s.get("axes", []), s.get("seeds", args.seeds), s.get("group")
    base = json.loads(args.base) if args.base else {}
    return base, list(args.axis or []), args.seeds, args.group


def _write_yamls(plan) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for a in plan.arms:
        exp = a.spec.get("experiment")
        name = exp.get("name") if isinstance(exp, dict) and exp.get("name") else a.run_name
        if name in seen:
            continue
        seen.add(name)
        print(f"  wrote {write_experiment_yaml(a.spec, EXP_DIR, str(name))}")


def main() -> int:
    p = argparse.ArgumentParser(description="Expand/validate/launch a Studio SweepSpec.")
    p.add_argument("--spec", help="path to a SweepSpec JSON {base_spec,axes,seeds}")
    p.add_argument("--base", help="inline base_spec JSON (when not using --spec)")
    p.add_argument("--axis", type=_parse_axis, action="append", help="path=v1,v2,... (repeatable)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--group", default=None)
    p.add_argument("--write-yaml", action="store_true",
                   help="write a per-arm experiment yaml under results/studio/experiments")
    p.add_argument("--launch", action="store_true", help="fan arms out as train subprocesses (local)")
    p.add_argument("--allow-warnings", action="store_true",
                   help="launch even if arms violate the spectral house rules")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = p.parse_args()

    base, axes, seeds, group = _load(args)
    plan = plan_sweep(base, axes, seeds, group=group)
    print(f"sweep group '{plan.group}': {plan.n} arms"
          + (" — ALL CLEAN" if plan.ok else " — WARNINGS PRESENT"))
    for a in plan.arms:
        print(f"  [{'ok  ' if a.ok else 'WARN'}] {a.run_name}  ({a.label})")
        for w in a.warnings:
            print(f"         ! {w}")

    if args.write_yaml or args.launch:
        _write_yamls(plan)

    if not args.launch:
        return 0
    if not plan.ok and not args.allow_warnings:
        print("REFUSING to launch: spectral house-rule warnings present "
              "(pass --allow-warnings to override).", file=sys.stderr)
        return 2

    LOGDIR.mkdir(parents=True, exist_ok=True)
    searchpath = f"hydra.searchpath=[file://{EXP_DIR.as_posix()}]"
    cmds = [(a.run_name,
             [sys.executable, "scripts/train.py", searchpath, *a.overrides,
              f"hydra.run.dir=outputs/sweep/{a.run_name}"])
            for a in plan.arms]

    from joblib import Parallel, delayed

    def _run(name: str, argv: list):
        with open(LOGDIR / f"{name}.log", "w") as fh:
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        return name, rc

    results = Parallel(n_jobs=args.jobs)(delayed(_run)(n, c) for n, c in cmds)
    failed = [(n, rc) for n, rc in results if rc]
    for n, rc in results:
        print(f"  {'OK  ' if not rc else 'FAIL'} {n}" + ("" if not rc else f"  (see {LOGDIR}/{n}.log)"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
