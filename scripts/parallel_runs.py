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

# preset -> (script, [(arm_name, [overrides]), ...])
PRESETS = {
    # The multi-task science grid: regularized vs lam0=0 vs within-task-only
    # penalty. Distinct experiment.name per arm => distinct W&B group => the
    # dashboard (and make_figures) renders one band per arm.
    "multitask_ablation": ("train_multitask.py", [
        ("multitask-reg",    []),
        ("multitask-lam0",   ["penalty.schedule.lam0=0", "penalty.schedule.floor=0",
                              "penalty.auto_dose.enabled=false"]),
        ("multitask-notask", ["penalty.include_task=false"]),
    ]),
    # lambda-schedule ablation (validation item 8) on Pendulum
    "schedule_ablation": ("train.py", [
        ("sched-cuberoot", ["penalty.schedule.kind=cuberoot", "penalty.form=frobenius"]),
        ("sched-step",     ["penalty.schedule.kind=step", "penalty.form=frobenius",
                            "+penalty.schedule.total_steps=20000"]),
        ("sched-cosine",   ["penalty.schedule.kind=cosine", "penalty.form=frobenius",
                            "+penalty.schedule.total_steps=20000"]),
        ("sched-constant", ["penalty.schedule.kind=constant", "penalty.form=frobenius"]),
        ("sched-sin2chirp", ["penalty.schedule.kind=sin2chirp", "penalty.form=frobenius"]),
        # two-oscillator interference: beats + phase-cancellation nulls
        ("sched-sincos", ["penalty.schedule.kind=sincos", "penalty.form=frobenius"]),
        # resonance hypothesis: f2 = m*f1 (m = latent rank) => exactly periodic
        # nulls that reinforce, vs golden-ratio incommensurate control
        ("sched-sincos-comm", ["penalty.schedule.kind=sincos",
                               "+penalty.schedule.period2_mode=multiple",
                               "+penalty.schedule.period2_mult=4"]),
        ("sched-sincos-gold", ["penalty.schedule.kind=sincos",
                               "+penalty.schedule.period2_mode=golden"]),
        # floor hypothesis: lambda -> exactly 0 should degrade late training
        # if the user's MLP-collapse claim holds (vs sched-step, floor 1e-5)
        ("sched-step-zero", ["penalty.schedule.kind=step", "penalty.form=frobenius",
                             "+penalty.schedule.total_steps=20000",
                             "penalty.schedule.floor=0"]),
        # the user's narrowed-down recipe: clamped decaying TRACE penalty
        ("sched-trace-chirp", ["penalty.schedule.kind=sin2chirp",
                               "penalty.form=laplacian_trace"]),
    ]),
}

# Original-report doses (sec.7): lam=0.5, step-anneal released at half of training.
# 200K env steps x (200 model updates / 1000 env steps) = 40K schedule steps.
_RECIPE = ["penalty.schedule.kind=step", "penalty.schedule.lam0=0.5",
           "penalty.form=frobenius", "penalty.auto_dose.enabled=false",
           "+penalty.schedule.step_at=0.5", "+penalty.schedule.step_factor=0.0",
           "+penalty.schedule.total_steps=40000",
           "training.total_env_steps=200000"]

PRESETS |= {
    # GPU-lean Colab arms: fixed doses from docs/original_findings_report.md,
    # NO parameter sweeps. Pass env via --overrides env=walker2d etc.
    # Recommended: recipe 3 seeds; control 1 seed (new envs only — the
    # HalfCheetah control is known: -165 +- 41); HalfCheetah recipe run is the
    # apparatus regression test (must land near +98 +- 23).
    "colab_recipe":  ("train.py", [("recipe", _RECIPE)]),
    "colab_control": ("train.py", [("control", ["penalty.schedule.lam0=0",
                                                "penalty.auto_dose.enabled=false",
                                                "penalty.schedule.floor=0",
                                                "smoothing.enabled=false",
                                                "training.total_env_steps=200000"])]),
    # head-to-head for the report's sec.3 claim, same dose, only the estimator
    # differs (clamped trace vs Frobenius) — 2 arms, no sweep
    "colab_estimator": ("train.py", [
        ("est-frobenius", _RECIPE),
        ("est-trace",     _RECIPE + ["penalty.form=laplacian_trace"]),
    ]),
    # champion vs challenger: run ONLY after the local schedule_ablation ranks
    # profiles — promotes the local winner to one GPU head-to-head against the
    # original step-anneal at matched dose. 2 arms, not a sweep.
    "colab_schedule_final": ("train.py", [
        ("sched-step-champ", _RECIPE),
        ("sched-challenger", ["penalty.schedule.kind=sin2chirp",
                              "penalty.schedule.lam0=0.5",
                              "+penalty.schedule.period0=8000",
                              "+penalty.schedule.period_end=1000",
                              "+penalty.schedule.total_steps=40000",
                              "training.total_env_steps=200000"]),
    ]),
}

# Wide-latent (4x obs, auto-capped) + rank-locked sincos interference, scaled
# to 5x the local sample budget — the GPU version of the proper multitask grid.
# latent_dim=9999 deliberately overshoots: the Trainer caps it at 4*obs_dim for
# ANY env, so the same preset works on pendulum_target (k=12) and
# halfcheetah_vel (k=68; pass +penalty.schedule.period2_mult=68 to keep the
# rank lock matched). 100K model updates => schedule total_steps=100000.
_MTW = ["model.latent_dim=9999", "penalty.schedule.kind=sincos",
        "+penalty.schedule.period2_mode=multiple",
        "+penalty.schedule.period2_mult=12",
        "+penalty.schedule.total_steps=100000",
        "penalty.auto_dose.warmup_updates=2000",
        "training.total_env_steps=500000"]

PRESETS |= {
    "colab_multitask_wide": ("train_multitask.py", [
        ("mtw-reg",    _MTW),
        ("mtw-lam0",   _MTW + ["penalty.schedule.lam0=0", "penalty.schedule.floor=0",
                               "penalty.auto_dose.enabled=false"]),
        ("mtw-notask", _MTW + ["penalty.include_task=false"]),
    ]),
}

# The spectral RL validation ("the big one"): does the supervised +33.7%
# (claims_ledger bridge run 3) survive contact with the RL loop? 3 arms, no
# sweeps, HalfCheetah-by-default (pass env via --overrides):
#   spec-ladder  — sigma ladder x lambda polynomial (the run-3 recipe preset)
#   spec-single  — single-sigma spectral control (isolates the ladder effect)
#   mlp-recipe   — the original MLP+Hutchinson recipe (known anchor +98 +- 23
#                  on HalfCheetah; doubles as the apparatus regression test)
# Recommended: 3 seeds for the two spectral arms; the anchor can run 1 seed
# if GPU budget is tight (its HalfCheetah band is established).
_SPEC = ["model.latent_dim=17", "training.total_env_steps=200000"]
# spectral arms: smooth floored decay ONLY (closed-form refits have no inertia
# — schedules touching ~0 produce an instant unregularized interpolator), and
# latent capped at 1x obs_dim (wide latents overfit the closed-form fit).
_SPEC_GUARD = ["penalty.schedule.kind=cuberoot", "penalty.schedule.floor=1e-5",
               "+model.latent_cap_mult=1"]
PRESETS |= {
    "colab_spectral": ("train.py", [
        ("spec-ladder", _SPEC + ["+experiment=spectral_ladder"]),
        ("spec-single", _SPEC + _SPEC_GUARD + ["spectral.enabled=true"]),
        ("mlp-recipe",  _RECIPE),
        # bridge run 5: SNR-calibrated ladder (sigma* measured per env at the
        # first refit) — the supervised champion (+48.3%); spec-ladder is its
        # fixed-ladder control
        ("spec-auto",   _SPEC + ["+experiment=spectral_auto"]),
    ]),
}


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
