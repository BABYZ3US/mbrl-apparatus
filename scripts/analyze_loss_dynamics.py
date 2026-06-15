#!/usr/bin/env python
"""Loss-dynamics analyzer (PM 2026-06-15) — tools to read the heuristic loss as a system.

For each run's metrics.jsonl it computes:
  1) the COMPONENT correlation matrix + PCA over training — how the loss terms,
     spectral readouts, and dynamics signals co-evolve (the interaction structure;
     PC1 is usually a single performance<->regularization<->rank axis).
  2) the time x eigenvalue SPECTRUM (needs the latent/eig* keys added 2026-06-15) —
     the latent's PCA-over-training: which modes climb to the band ceiling, which die
     to the floor, how the effective rank emerges.
  3) an EQUILIBRIUM fit against the closed-form spectral bounds (see
     regularization/rank2_frame.py + the spectral_equilibrium widget):
       dead modes -> floor,  active modes -> ceiling + g/(2 w_band),
       cond(G) -> ceiling/floor,  rank = #{modes with utility > w_compress/(2 sqrt eps)}.

Numpy-only (matplotlib optional -> PNG heatmaps; otherwise JSON + printed report).

  python scripts/analyze_loss_dynamics.py --runs 'results/runs/cf1[67]-*' [--ceiling 1.0 --floor 0.1]
"""
from __future__ import annotations
import argparse, glob, json, math, os

import numpy as np

# interpretable loss / dynamics / spectral-summary signals (skip any that are absent)
COMPONENTS = [
    "loss/total", "loss/dyn", "loss/reward", "loss/policy", "loss/value",
    "frame/band", "frame/compress", "penalty/lambda", "penalty/return_gate",
    "dual/couple", "dual/p_consistency", "latent/gram_eff_rank", "latent/gram_cond",
    "latent/z_std", "imagine/return_mean", "imagine/align", "policy/entropy",
    "imagine/horizon", "eval",
]
SHORT = lambda k: (k.replace("latent/", "").replace("imagine/", "im/")
                   .replace("penalty/", "pen/").replace("policy/", "pol/"))


def load_rows(arm: str) -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(arm, "metrics.jsonl")) if l.strip()]


def series(rows: list[dict], key: str) -> np.ndarray:
    """Row-ordered, forward-filled series; eval pulled from eval/return|eval; cond -> log10."""
    out, last = [], math.nan
    for r in rows:
        v = r.get("eval/return", r.get("eval")) if key == "eval" else r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            last = v
        out.append(last)
    a = np.asarray(out, float)
    if key == "latent/gram_cond":
        a = np.log10(np.clip(a, 1.0, None))
    return a


def component_analysis(rows: list[dict], drop_warmup: int = 5) -> dict | None:
    cols, names = [], []
    for c in COMPONENTS:
        s = series(rows, c)
        if np.isfinite(s).sum() < 10 or np.nanstd(s) < 1e-9:
            continue
        cols.append(s); names.append(SHORT(c))
    if len(cols) < 3:
        return None
    M = np.asarray(cols).T
    M = M[np.isfinite(M).all(1)][drop_warmup:]
    if len(M) < 10:
        return None
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    C = Z.T @ Z / len(Z)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    evr = (w / w.sum()).tolist()
    loadings = {f"PC{p+1}": sorted(((names[i], round(float(V[i, p]), 3)) for i in range(len(names))),
                                   key=lambda t: -abs(t[1]))[:6] for p in range(min(3, len(names)))}
    return {"names": names, "corr": np.round(C, 3).tolist(),
            "explained_variance": [round(x, 3) for x in evr[:8]], "pc_loadings": loadings}


def spectrum(rows: list[dict]) -> np.ndarray | None:
    """[T, k] eigenvalue spectrum over training from latent/eig* keys (None if not logged)."""
    keys = sorted(k for k in rows[-1] if k.startswith("latent/eig"))
    if not keys:
        return None
    out = []
    for r in rows:
        if all(k in r for k in keys):
            out.append([r[k] for k in keys])
    return np.asarray(out, float) if out else None


def equilibrium_fit(spec: np.ndarray, ceiling: float, floor: float) -> dict:
    """Closed-form check on the FINAL spectrum: active modes ~ceiling, dead ~floor,
    cond -> ceiling/floor, effective rank via participation ratio."""
    ev = np.sort(spec[-1])[::-1]
    ev = np.clip(ev, 0, None)
    cond = float(ev[0] / max(ev[-1], 1e-12))
    pr = float((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12))   # participation ratio (eff rank)
    active = int((ev > 0.5 * ceiling).sum())
    dead_at_floor = int((np.abs(ev - floor) < 0.5 * floor).sum())
    return {"lambda_max": round(float(ev[0]), 3), "lambda_min": round(float(ev[-1]), 4),
            "cond": cond, "log10_cond": round(math.log10(max(cond, 1)), 2),
            "eff_rank_pr": round(pr, 2), "n_active_above_half_ceiling": active,
            "n_pinned_at_floor": dead_at_floor,
            "cond_bound_ceiling_over_floor": round(ceiling / floor, 1),
            "bound_satisfied": cond <= 1.5 * ceiling / floor}


def maybe_heatmap(spec: np.ndarray, steps, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(np.log10(np.clip(spec.T, 1e-6, None)), aspect="auto", origin="lower",
                   extent=[0, len(spec), 0, spec.shape[1]], cmap="viridis")
    ax.set_xlabel("eval index (training time)"); ax.set_ylabel("eigenvalue rank (0 = largest)")
    fig.colorbar(im, label="log10 eigenvalue"); fig.tight_layout(); fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs/*", help="glob of run dirs")
    ap.add_argument("--out", default="results/analysis", help="output dir for JSON/PNG")
    ap.add_argument("--ceiling", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = sorted(d for d in glob.glob(args.runs) if os.path.exists(os.path.join(d, "metrics.jsonl")))
    if not arms:
        print("no runs matched", args.runs); return
    report = {}
    print("=== loss-dynamics analysis (%d runs) ===" % len(arms))
    for arm in arms:
        rows = load_rows(arm)
        name = os.path.basename(arm)
        entry = {"n_rows": len(rows)}
        ca = component_analysis(rows)
        if ca:
            entry["component"] = ca
            print("\n%s  (%d rows)" % (name, len(rows)))
            print("  PCA explained var:", ca["explained_variance"][:5])
            print("  PC1:", ca["pc_loadings"]["PC1"])
        spec = spectrum(rows)
        if spec is not None and len(spec) > 1:
            eq = equilibrium_fit(spec, args.ceiling, args.floor)
            entry["equilibrium"] = eq
            png = os.path.join(args.out, name + "_spectrum.png")
            entry["heatmap"] = maybe_heatmap(spec, None, png)
            print("  spectrum: eff_rank_pr=%.1f log10cond=%.1f active=%d/%d  bound(c/f=%.0f) %s"
                  % (eq["eff_rank_pr"], eq["log10_cond"], eq["n_active_above_half_ceiling"],
                     spec.shape[1], eq["cond_bound_ceiling_over_floor"],
                     "OK" if eq["bound_satisfied"] else "VIOLATED (floor too soft)"))
        else:
            print("\n%s: no latent/eig* spectrum logged (pre-2026-06-15 run -> summaries only)" % name)
        report[name] = entry
    outpath = os.path.join(args.out, "loss_dynamics.json")
    json.dump(report, open(outpath, "w"), indent=2)
    print("\nwrote", outpath)


if __name__ == "__main__":
    main()
