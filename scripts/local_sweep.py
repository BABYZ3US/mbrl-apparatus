"""Local CPU grid runner for the synthetic experiments (validation items 6-7)
and Pendulum-class ablations — joblib across cores, one W&B run per cell.

These need no GPU and can run from day one, in parallel with Colab work.

Usage:
  python scripts/local_sweep.py --experiment stone --jobs 8
  python scripts/local_sweep.py --experiment smoothness --jobs 8
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


# ---------- synthetic targets of known Sobolev smoothness ----------
def synthetic_target(s0: float, d: int, rng) -> "callable":
    """Random function with spectral decay giving H^{s0} smoothness:
    coefficients c_j ~ N(0, j^{-(2 s0 + d)/d}) on a random Fourier basis."""
    n_modes = 256
    freqs = rng.normal(size=(n_modes, d)) * np.arange(1, n_modes + 1)[:, None] ** (1.0 / d)
    amps = rng.normal(size=n_modes) * np.arange(1, n_modes + 1) ** (-(2 * s0 + d) / (2 * d))
    phases = rng.uniform(0, 2 * np.pi, n_modes)

    def f(x):  # x: (N, d)
        return np.cos(x @ freqs.T + phases) @ amps
    return f


def fit_cell(s0: float, d: int, n: int, lam: float, seed: int, probes: int = 2) -> dict:
    """Train a small MLP with the H^2 penalty (Hutchinson, `probes` probes) on n
    samples of a smoothness-s0 target; return test MSE + wall-time. Items 6-7
    reduce to grids over this; item 3 (probe-count, R4) sweeps `probes`."""
    import time
    import torch
    from mbrl.regularization.hutchinson import hvp_penalty
    from mbrl.models.encoder import mlp

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f = synthetic_target(s0, d, rng)
    X = rng.uniform(-1, 1, (n, d)).astype(np.float32)
    Xte = rng.uniform(-1, 1, (2048, d)).astype(np.float32)
    y, yte = f(X).astype(np.float32), f(Xte).astype(np.float32)

    net = mlp([d, 128, 128, 1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(2000):
        pred = net(Xt).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, yt)
        if lam > 0:
            loss = loss + lam * hvp_penalty(lambda x: net(x).squeeze(-1), Xt,
                                            n_probes=probes, generator=gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    wall = time.perf_counter() - t0

    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(
            net(torch.from_numpy(Xte)).squeeze(-1), torch.from_numpy(yte)).item()
    return {"s0": s0, "d": d, "n": n, "lam": lam, "seed": seed, "probes": probes,
            "test_mse": mse, "wall_s": wall}


GRIDS = {
    # item 7: error vs n at fixed (s, d); predicted slope n^{-2s/(2s+d)}
    "stone": dict(s0=[2.0], d=[2, 4], n=[64, 128, 256, 512, 1024, 2048],
                  lam=[1e-3], seed=range(5), probes=[2]),
    # item 6: does optimal lambda track target smoothness? s* = s0 below 2, ->2 above
    "smoothness": dict(s0=[0.5, 1.0, 1.5, 2.0, 3.0], d=[4], n=[512],
                       lam=[0.0, 1e-4, 1e-3, 1e-2, 1e-1], seed=range(5), probes=[2]),
    # item 3 (R4): probe-count vs accuracy/compute. Does test MSE plateau by N=2
    # (the validated default) while wall-time grows ~linearly? — the enabling-lever
    # claim, swept rather than asserted. Penalized arm only (lam>0).
    "probe": dict(s0=[2.0], d=[4], n=[512], lam=[1e-3], seed=range(8),
                  probes=[1, 2, 4, 8, 16, 32]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=GRIDS, required=True)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    g = GRIDS[args.experiment]
    cells = list(itertools.product(g["s0"], g["d"], g["n"], g["lam"], g["seed"],
                                   g.get("probes", [2])))
    print(f"{args.experiment}: {len(cells)} cells")

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_cell)(*c) for c in cells)

    import json
    out = Path(args.out or f"results/{args.experiment}_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
