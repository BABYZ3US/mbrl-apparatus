import json, math
from pathlib import Path

groups = {}
import sys
PATTERN = sys.argv[1] if len(sys.argv) > 1 else "ens-p*"
for d in sorted(Path("results/runs").glob(PATTERN)):
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    g = d.name.split("-HalfCheetah")[0]
    evals = [r["eval/return"] for r in rows if "eval/return" in r]
    rvar = [r["imagine/return_var"] for r in rows if "imagine/return_var" in r]
    pvar = [r["imagine/penalty_var"] for r in rows if "imagine/penalty_var" in r]
    pmean = [r["imagine/penalty_mean"] for r in rows if "imagine/penalty_mean" in r]
    tail = lambda xs: xs[int(len(xs) * 0.75):] if xs else []
    groups.setdefault(g, []).append({
        "final": sum(evals[-3:]) / max(1, len(evals[-3:])) if evals else None,
        "rvar": sum(tail(rvar)) / max(1, len(tail(rvar))) if rvar else None,
        "pvar": sum(tail(pvar)) / max(1, len(tail(pvar))) if pvar else None,
        "pmean": sum(tail(pmean)) / max(1, len(tail(pmean))) if pmean else None,
        "steps": max(r.get("env_steps", 0) for r in rows)})

def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0
    return "%.4g +/- %.3g" % (m, s)

print("group     steps   final_eval            late_rvar          late_pvar        late_pmean")
for g in sorted(groups):
    arms = groups[g]
    print("%-9s %-7d %-21s %-18s %-16s %s" % (
        g, min(a["steps"] for a in arms),
        ms([a["final"] for a in arms]), ms([a["rvar"] for a in arms]),
        ms([a["pvar"] for a in arms]), ms([a["pmean"] for a in arms])))
