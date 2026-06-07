"""Compare regularized (lam0=1e-3) vs ablation (lam0=0) zero-shot gap.

Reads local JSONL mirrors written by train_multitask.py. For each arm/seed we
take the final logged eval and the mean over the last few evals (less noisy).
The headline number is the zero-shot generalization gap on INTERPOLATION
(smoothness only promises interpolation); extrapolation reported alongside.
"""
from __future__ import annotations
import json, glob, statistics as st
from pathlib import Path

RUNS = Path("results/runs")
EVAL_KEYS = ["eval/return", "eval/zeroshot_interp", "eval/zeroshot_extrap"]


def load(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    evals = [r for r in rows if "eval/zeroshot_interp" in r]
    return rows, evals


def tail_mean(evals, key, k=3):
    vals = [e[key] for e in evals[-k:] if key in e]
    return st.mean(vals) if vals else float("nan")


def summarize(prefix: str):
    out = {}
    for d in sorted(RUNS.glob(f"{prefix}-pendulum_target-s*")):
        seed = d.name.split("-s")[-1]
        rows, evals = load(d)
        if not evals:
            continue
        tr = tail_mean(evals, "eval/return")
        ip = tail_mean(evals, "eval/zeroshot_interp")
        ex = tail_mean(evals, "eval/zeroshot_extrap")
        out[seed] = {"train": tr, "interp": ip, "extrap": ex,
                     "gap_interp": tr - ip, "gap_extrap": tr - ex,
                     "n_eval": len(evals), "last_step": rows[-1].get("env_steps")}
    return out


def fmt(x): return f"{x:8.2f}"


def main():
    reg = summarize("multitask")
    lam0 = summarize("multitask_lam0")
    seeds = sorted(set(reg) & set(lam0))
    print(f"\n{'seed':>4} | {'arm':<10} | {'train':>8} {'interp':>8} {'extrap':>8} "
          f"| {'gap_int':>8} {'gap_ext':>8} | n_eval last_step")
    print("-" * 92)
    for s in seeds:
        for label, arm in (("reg(λ>0)", reg[s]), ("abl(λ=0)", lam0[s])):
            print(f"{s:>4} | {label:<10} | {fmt(arm['train'])} {fmt(arm['interp'])} "
                  f"{fmt(arm['extrap'])} | {fmt(arm['gap_interp'])} {fmt(arm['gap_extrap'])} "
                  f"| {arm['n_eval']:>6} {arm['last_step']}")
    if not seeds:
        print("(no completed seeds yet)")
        return
    print("-" * 92)

    def agg(d, k): return [d[s][k] for s in seeds]
    print("\n=== Across seeds (mean ± sd) ===")
    for label, src in (("reg(λ>0)", reg), ("abl(λ=0)", lam0)):
        gi = agg(src, "gap_interp"); ge = agg(src, "gap_extrap")
        zi = agg(src, "interp")
        sd = (lambda v: st.stdev(v) if len(v) > 1 else 0.0)
        print(f"  {label}: zeroshot_interp = {st.mean(zi):8.2f} ± {sd(zi):.2f} | "
              f"gap_interp = {st.mean(gi):8.2f} ± {sd(gi):.2f} | "
              f"gap_extrap = {st.mean(ge):8.2f} ± {sd(ge):.2f}")
    gi_reg = agg(reg, "gap_interp"); gi_abl = agg(lam0, "gap_interp")
    zi_reg = agg(reg, "interp"); zi_abl = agg(lam0, "interp")
    print(f"\n  Δ zeroshot_interp (reg - abl): {st.mean(zi_reg) - st.mean(zi_abl):+8.2f}  "
          f"(higher reward = better)")
    print(f"  Δ gap_interp     (reg - abl): {st.mean(gi_reg) - st.mean(gi_abl):+8.2f}  "
          f"(more negative = curvature penalty narrows the generalization gap)")


if __name__ == "__main__":
    main()
