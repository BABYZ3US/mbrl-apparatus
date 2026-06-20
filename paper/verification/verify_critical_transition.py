"""Experimental verification — derivations.md Section 9 (Conjecture 1, critical ratio r* = 1/5).

Detects the critical co-transition: the step where the realized return jumps most, and whether
the policy determinant proxy (policy entropy) and the imagined return move *together* at that step.

Usage:
  python verify_critical_transition.py results/runs/abl1-A12critical-s0-*/metrics.jsonl
  python verify_critical_transition.py            # synthetic demo + the recorded A10fifth-s0 example

The recorded A10fifth-s0 example (|lambda|^2 = 0.2 init) below is the one-seed empirical support.
"""
import sys
import json
import glob
import numpy as np

# Recorded example: A10fifth-s0 (0.2 init), eval window around the transition.
RECORDED_A10S0 = {
    "step": [195_000, 205_000, 215_000, 225_000],
    "eval/return_det": [-416, -227, -187, 540],
    "policy/entropy": [-8.44, -4.43, -2.55, -0.33],
    "imagine/return_mean": [-16.5, -12.1, -1.3, 5.2],
}


def report(step, ev, pe, ir, label):
    ev = np.asarray(ev, float)
    dev = np.diff(ev)
    i = int(np.argmax(dev))
    print(f"  [{label}] biggest eval jump: {ev[i]:.0f} -> {ev[i+1]:.0f} (+{dev[i]:.0f}) "
          f"at {step[i]//1000}k -> {step[i+1]//1000}k")
    if pe[i] is not None and pe[i + 1] is not None:
        print(f"      policy entropy (~log policy det): {pe[i]:.2f} -> {pe[i+1]:.2f} "
              f"(delta {pe[i+1]-pe[i]:+.2f}, rising toward 0)")
    if ir[i] is not None and ir[i + 1] is not None:
        crosses = ir[i] < 0 <= ir[i + 1]
        print(f"      imagined return: {ir[i]:.1f} -> {ir[i+1]:.1f}  (crosses zero: {crosses})")
    co = (pe[i] is not None and pe[i + 1] is not None and pe[i + 1] > pe[i]
          and ir[i] is not None and ir[i + 1] is not None and ir[i + 1] > ir[i])
    print(f"      => simultaneous vertical break (eval up & policy-det up & return up): {co}")
    return co


def from_jsonl(path):
    R = [json.loads(l) for l in open(path) if l.strip()]
    R = [r for r in R if r.get("eval/return_det") is not None]
    step = [r.get("env_steps", 0) for r in R]
    ev = [r["eval/return_det"] for r in R]
    pe = [r.get("policy/entropy") for r in R]
    ir = [r.get("imagine/return_mean") for r in R]
    return step, ev, pe, ir


if __name__ == "__main__":
    print("=== Critical co-transition detector (derivations Section 9) ===")
    args = [a for a in sys.argv[1:]]
    paths = []
    for a in args:
        paths += glob.glob(a)
    if paths:
        for p in paths:
            report(*from_jsonl(p), label=p.split("/")[-2] if "/" in p else p)
    else:
        print("  (no metrics path given — showing the recorded A10fifth-s0 example)")
        report(RECORDED_A10S0["step"], RECORDED_A10S0["eval/return_det"],
               RECORDED_A10S0["policy/entropy"], RECORDED_A10S0["imagine/return_mean"],
               label="A10fifth-s0 (0.2 init)")
    print("done")
