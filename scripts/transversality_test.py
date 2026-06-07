"""Experiment 10 — the gap-closing transversality test (docs/claims_ledger.md).

Supervised multi-kernel test redone under the two conditions the original lacked:
  (1) data from a COMPETENT policy (narrow state distribution), not random;
  (2) a genuinely CURVED reward (Gaussian-bump shaping: high curvature near the
      target, near-flat elsewhere — the regime where smoothness constraints bite).

Protocol per (n, seed, arm): collect transitions on Pendulum with a scripted
energy swing-up + hold controller (competent without needing RL to succeed;
--checkpoint uses a trained policy instead when available); fit reward + dynamics
models under arm ∈ {none, R, R+T}; measure held-out reward MSE and the
transversality angle alpha. Success criteria (both required, ledger):
  - R+T beats R by the predicted ~6-25% in sample efficiency, >=5 seeds;
  - the R+T benefit correlates with alpha across conditions.

Local CPU; joblib-parallel.   python scripts/transversality_test.py --jobs 8
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; heavy imports stay function-local
    import torch

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

# Fixed reference stats for (cos, sin, thdot, a), measured once from a 30k-step
# competent-policy rollout (seed 0). NEVER recompute per cell — the rich reward
# is defined in these coordinates and must be the same function everywhere.
REF_MU = np.array([-0.6111000180244446, -0.002300000051036477, 0.0142000000923872, 0.006800000090152025], dtype=np.float32)
REF_SD = np.array([0.6984999775886536, 0.37229999899864197, 1.3478000164031982, 1.8148000240325928], dtype=np.float32)


def collect_competent(n_steps: int, seed: int, noise: float = 0.1):
    """Energy-based swing-up + PD hold on Pendulum-v1: narrow, on-policy-like
    state distribution around the upright manifold."""
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    X, A, X2 = [], [], []
    for _ in range(n_steps):
        cos_t, sin_t, thdot = obs
        theta = np.arctan2(sin_t, cos_t)
        E = 0.5 * thdot ** 2 - 10.0 * cos_t          # ~ energy above bottom
        if cos_t < 0.95:                              # swing-up: pump energy
            a = 2.0 * np.sign(thdot * (10.0 - E))
        else:                                         # hold: PD around upright
            a = -8.0 * theta - 1.5 * thdot
        a = np.clip(a + noise * rng.normal(), -2, 2).reshape(1).astype(np.float32)
        obs2, _, term, trunc, _ = env.step(a)
        X.append(obs); A.append(a); X2.append(obs2)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(X, np.float32), np.array(A, np.float32), np.array(X2, np.float32)


def curved_reward(X: np.ndarray, A: np.ndarray, width: float = 0.15) -> np.ndarray:
    """v1 'bump' reward (kept for comparison): rank-1 curvature, all in theta.
    Run 01 finding: d_eff ~ 1 => transversality trivial (alpha ~ 88 deg), R+T
    redundant by construction, benefit ~ 0. Too simple — superseded by rich_reward."""
    theta = np.arctan2(X[:, 1], X[:, 0])
    return (np.exp(-theta ** 2 / width) - 0.05 * (A[:, 0] ** 2)).astype(np.float32)


def rich_reward(xa_std: "torch.Tensor"):
    """v2 reward, defined in STANDARDIZED (z-scored) input coords: per-direction
    modulated quadratics -0.5 A_i u_i^2 (1 + 0.15 sin(1.2 u_i + phi_i)) + a tanh
    cross term. Curvature is O(2) in EVERY direction at EVERY point, but the
    second derivative varies nonlinearly with position — multi-dimensional and
    not trivially learnable. Measured true d_eff ~ 3.5 on competent-policy data
    (vs ~1 for the bump), putting the test in the regime where kappa=1 predicts
    a ~15% benefit."""
    import torch
    amps, phases = [1.0, 0.95, 0.9, 0.85], [0.0, 1.1, 2.3, 3.6]
    r = 0.25 * torch.tanh(xa_std[..., 1]) * torch.tanh(xa_std[..., 3])
    for i, (A_, p) in enumerate(zip(amps, phases)):
        u = xa_std[..., i]
        r = r - 0.5 * A_ * u ** 2 * (1 + 0.15 * torch.sin(1.2 * u + p))
    return r


def fit_cell(n: int, seed: int, arm: str, lam: float = 1e-2,
             epochs: int = 3000, reward: str = "rich") -> dict:
    import torch
    from mbrl.models.encoder import mlp
    from mbrl.regularization.hutchinson import hvp_penalty
    from mbrl.regularization.transversality import transversality_angle, effective_dim

    torch.manual_seed(seed)
    X, A, X2 = collect_competent(n + 4096, seed)
    XA = np.concatenate([X, A], axis=1)
    # Standardize with FIXED reference stats (long competent-policy rollout),
    # so (a) the rich reward is the SAME target function in every cell — using
    # per-cell train stats would change the target with n and seed, and
    # (b) the coordinate system for the isotropic penalty / alpha / d_eff is
    # consistent. Raw coords are scale-dominated (td +-7 vs +-1 for cos/sin).
    XA = ((XA - REF_MU) / REF_SD).astype(np.float32)
    X2 = ((X2 - REF_MU[:3]) / REF_SD[:3]).astype(np.float32)

    if reward == "rich":
        r = rich_reward(torch.from_numpy(XA)).numpy().astype(np.float32)
    else:
        r = curved_reward(X, A)
    # SHUFFLED train/test split: the trajectory is phased (swing-up then hold),
    # so a temporal split trains on one regime and tests on another — that's a
    # distribution-shift experiment, not a sample-complexity one. (Found by the
    # sandbox verification run: 1e10 test MSE at n=128.)
    perm = np.random.default_rng(seed).permutation(len(XA))
    tr, te = perm[:n], perm[n:]
    xa_tr = torch.from_numpy(XA[tr]); r_tr = torch.from_numpy(r[tr])
    x2_tr = torch.from_numpy(X2[tr])
    xa_te = torch.from_numpy(XA[te]); r_te = torch.from_numpy(r[te])

    d = XA.shape[1]
    R = mlp([d, 128, 128, 1])
    T = mlp([d, 128, 128, X.shape[1]])
    opt = torch.optim.AdamW([*R.parameters(), *T.parameters()], lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    fn_r = lambda x: R(x).squeeze(-1)
    fn_t = lambda x: T(x).sum(-1)

    for _ in range(epochs):
        loss = torch.nn.functional.mse_loss(R(xa_tr).squeeze(-1), r_tr) \
             + torch.nn.functional.mse_loss(T(xa_tr), x2_tr)
        if arm in ("R", "RT"):
            loss = loss + lam * hvp_penalty(fn_r, xa_tr, 2, gen)
        if arm == "RT":
            loss = loss + lam * hvp_penalty(fn_t, xa_tr, 2, gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(R(xa_te).squeeze(-1), r_te).item()
    alpha = transversality_angle(fn_r, fn_t, xa_te[:512], n_probes=8,
                                 generator=gen)
    # d_eff is a PREDICTION, not a knob: the penalty's spectral filtering should
    # push it down on its own; the multi-kernel benefit should track
    # sqrt((d_eff - 1)/d_eff) computed from the MEASURED d_eff.
    d_eff = effective_dim(fn_r, xa_te[:512], n_probes=64, generator=gen)
    return {"n": n, "seed": seed, "arm": arm, "lam": lam, "reward": reward,
            "test_mse": mse, "alpha_deg": alpha, "d_eff": d_eff}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--seeds", type=int, default=5,
                   help=">=5 required for power at a ~6%% effect (ledger)")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--reward", choices=["rich", "bump"], default="rich",
                   help="rich: multi-dim curvature, d_eff~3.5 (the real test); "
                        "bump: v1 rank-1 reward (kept for comparison)")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default="results/transversality_results.json")
    args = p.parse_args()

    cells = list(itertools.product(args.ns, range(args.seeds), ("none", "R", "RT")))
    print(f"{len(cells)} cells ({len(args.ns)} n x {args.seeds} seeds x 3 arms)")
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_cell)(n, s, arm, args.lam, reward=args.reward)
        for n, s, arm in cells)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))

    # quick readout: R+T benefit over R per n, and its correlation with alpha
    import collections
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in results:
        by[row["n"]][row["arm"]].append(row)
    print(f"\n{'n':>6} {'none':>9} {'R':>9} {'R+T':>9} {'benefit%':>9} {'alpha':>7} "
          f"{'d_eff(none)':>11} {'d_eff(R)':>9} {'pred%':>7}")
    for n in sorted(by):
        m = {a: np.mean([r["test_mse"] for r in rows]) for a, rows in by[n].items()}
        # fold to [0, 90]: transversality is about alignment magnitude; >90 deg
        # just means a negative Frobenius inner product
        alpha = np.mean([min(r["alpha_deg"], 180 - r["alpha_deg"])
                         for r in by[n]["RT"]])
        de_none = np.mean([r["d_eff"] for r in by[n]["none"]])
        de_r = np.mean([r["d_eff"] for r in by[n]["R"]])
        ben = 100 * (m["R"] - m["RT"]) / m["R"]
        # theory: error ratio sqrt((d_eff-1)/d_eff) at the MEASURED (penalized) d_eff
        pred = 100 * (1 - np.sqrt(max(de_r - 1, 0.0) / max(de_r, 1e-9)))
        print(f"{n:>6} {m['none']:>9.4f} {m['R']:>9.4f} {m['RT']:>9.4f} "
              f"{ben:>8.1f}% {alpha:>6.1f}° {de_none:>11.2f} {de_r:>9.2f} "
              f"{pred:>6.1f}%")
    print("\n  d_eff(R) < d_eff(none) checks the penalty pushes d_eff down on its own;")
    print("  benefit% ~ pred% (from measured d_eff) closes the quantitative loop.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
