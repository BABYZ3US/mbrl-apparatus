"""Regenerate figures from EITHER the W&B cloud API or local files. CPU task.

Every training run writes a local JSONL mirror (results/runs/<name>/metrics.jsonl)
of the same keys it logs to W&B, so the full figure set is reproducible offline;
the W&B path pulls identical data from the cloud. Both feed one render path.

Usage:
  # local, no network (default; reads results/runs/ + results/*.json)
  python scripts/make_figures.py
  python scripts/make_figures.py --group multienv

  # cloud
  python scripts/make_figures.py --source wandb --project you/mbrl-curvature --group multienv

  # auto: try local first, fall back to W&B if --project is set
  python scripts/make_figures.py --source auto --project you/mbrl-curvature

  # synthetic-experiment JSONs (local_sweep.py outputs)
  python scripts/make_figures.py --sweep results/stone_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from mbrl.viz import curves

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures"

# keys every loader must produce per run
KEYS = ["env_steps", "eval/return", "penalty/value", "penalty/lambda",
        "imagine/return_var", "step",
        # multi-task zero-shot (train_multitask.py; logged on eval rows)
        "eval/zeroshot_interp", "eval/zeroshot_extrap", "eval/generalization_gap"]


# ---------------------------------------------------------------- loaders
def load_local(results_dir: Path, group: str | None) -> dict[str, dict]:
    """{run_name: {"meta":..., "series": {key: np.array}}} from JSONL mirrors."""
    from mbrl.utils.metrics_logger import load_runs
    out = {}
    for name, run in load_runs(results_dir, group=group).items():
        series = defaultdict(list)
        for row in run["history"]:
            # env_steps is the x-axis for eval/return, which is logged on a coarser
            # cadence (eval_every_iters); only record env_steps on eval rows so the
            # two series stay index-aligned. Otherwise the return curve is plotted
            # against the wrong (denser) env-step values.
            has_eval = "eval/return" in row
            for k in KEYS:
                if k not in row or (k == "env_steps" and not has_eval):
                    continue
                series[k].append(row[k])
        if series:
            out[name] = {"meta": run["meta"],
                         "series": {k: np.asarray(v) for k, v in series.items()}}
    return out


def load_wandb(project: str, group: str | None) -> dict[str, dict]:
    """Same structure as load_local, from the W&B API."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project, filters={"group": group} if group else None)
    out = {}
    for r in runs:
        h = r.history(keys=KEYS, pandas=True)
        if not len(h):
            continue
        series = {k: h[k].dropna().to_numpy() for k in KEYS if k in h}
        out[r.name] = {"meta": {"group": r.group, "seed": r.config.get("seed")},
                       "series": series}
    return out


# ---------------------------------------------------------------- renderers
def _stack_by_group(runs: dict, ykey: str, xkey: str = "env_steps"):
    """Align seeds within each group to common length -> {group: (x, Y[seeds, T])}."""
    by_group = defaultdict(list)
    for name, run in runs.items():
        s = run["series"]
        if ykey in s and xkey in s and len(s[ykey]):
            n = min(len(s[xkey]), len(s[ykey]))
            by_group[run["meta"].get("group", name)].append((s[xkey][:n], s[ykey][:n]))
    panel = {}
    for g, series in by_group.items():
        T = min(len(y) for _, y in series)
        panel[g] = (series[0][0][:T], np.stack([y[:T] for _, y in series]))
    return panel


def render_all(runs: dict, tag: str):
    made = []
    if (panel := _stack_by_group(runs, "eval/return")):
        ax = curves.return_vs_steps(panel, title=tag)
        out = FIGDIR / f"return_{tag}.png"
        ax.figure.savefig(out, bbox_inches="tight"); made.append(out)

    # multi-task: train vs zero-shot interp/extrap curves on one panel
    zs_panel = {}
    for label, key in [("train tasks", "eval/return"),
                       ("zero-shot interp", "eval/zeroshot_interp"),
                       ("zero-shot extrap", "eval/zeroshot_extrap")]:
        for g, xy in _stack_by_group(runs, key).items():
            zs_panel[f"{g}: {label}"] = xy
    if any("zero-shot" in k for k in zs_panel):
        ax = curves.return_vs_steps(zs_panel, title=f"{tag}: zero-shot generalization")
        out = FIGDIR / f"zeroshot_{tag}.png"
        ax.figure.savefig(out, bbox_inches="tight"); made.append(out)

    for name, run in runs.items():  # per-run penalty/lambda diagnostics
        s = run["series"]
        if {"penalty/value", "penalty/lambda", "step"} <= s.keys() and len(s["step"]):
            n = min(map(len, (s["step"], s["penalty/lambda"], s["penalty/value"])))
            ax = curves.lambda_and_penalty(s["step"][:n], s["penalty/lambda"][:n],
                                           s["penalty/value"][:n])
            out = FIGDIR / f"penalty_{name}.png"
            ax.figure.savefig(out, bbox_inches="tight"); made.append(out)
    return made


def render_sweep_json(path: Path):
    """Figures for local_sweep.py outputs (stone / smoothness)."""
    data = json.loads(path.read_text())
    made = []
    if path.stem.startswith("stone"):
        by_d = defaultdict(lambda: defaultdict(list))
        for row in data:
            by_d[row["d"]][row["n"]].append(row["test_mse"])
        for d, n_map in by_d.items():
            ns = sorted(n_map)
            errs = np.array([np.mean(n_map[n]) for n in ns])
            ax = curves.stone_rate(ns, errs, s=2.0, d=d)
            ax.set_title(f"Stone rate, d={d}")
            out = FIGDIR / f"stone_d{d}.png"
            ax.figure.savefig(out, bbox_inches="tight"); made.append(out)
    elif path.stem.startswith("smoothness"):
        import matplotlib.pyplot as plt
        by_s0 = defaultdict(lambda: defaultdict(list))
        for row in data:
            by_s0[row["s0"]][row["lam"]].append(row["test_mse"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s0, lam_map in sorted(by_s0.items()):
            lams = sorted(lam_map)
            errs = [np.mean(lam_map[l]) for l in lams]
            ax.plot([l if l > 0 else min(x for x in lams if x > 0) / 10 for l in lams],
                    errs, "o-", label=rf"$s_0={s0}$")
        ax.set_xscale("log"); ax.set_xlabel(r"$\lambda$"); ax.set_ylabel("test MSE")
        ax.set_title(r"U-curves vs target smoothness $s_0$ (R14 / item 6)")
        ax.legend()
        out = FIGDIR / "smoothness_ucurves.png"
        fig.savefig(out, bbox_inches="tight"); made.append(out)
    return made


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["local", "wandb", "auto"], default="local")
    p.add_argument("--project", default=None, help="entity/project (wandb source)")
    p.add_argument("--group", default=None, help="experiment name filter")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--sweep", action="append", default=[],
                   help="local_sweep result JSON(s); repeatable")
    args = p.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    runs = {}
    if args.source in ("local", "auto"):
        runs = load_local(Path(args.results_dir), args.group)
    if not runs and args.source in ("wandb", "auto"):
        if not args.project:
            p.error("--source wandb/auto needs --project entity/project")
        runs = load_wandb(args.project, args.group)

    made = render_all(runs, tag=args.group or "all") if runs else []

    # synthetic sweeps: explicit paths, or auto-discover in local/auto mode
    sweeps = [Path(s) for s in args.sweep]
    if not sweeps and args.source != "wandb":
        sweeps = list(Path(args.results_dir).glob("*_results.json"))
    for s in sweeps:
        made += render_sweep_json(s)

    if not made:
        print("no data found — run training or local_sweep.py first "
              "(or pass --source wandb --project ...)")
    for f in made:
        print("wrote", f)


if __name__ == "__main__":
    main()
