"""Build ONE self-contained interactive diagnostics page: results/dashboard.html.

Reads whatever training state exists right now (JSONL mirrors + checkpoints) and
embeds everything as JSON in a single HTML file (Plotly from CDN). Safe to run
while the multitask grid is training: torn JSONL lines are skipped, partially
written / missing checkpoints are tolerated, and every section degrades to an
inline error note instead of crashing the export.

  python scripts/make_dashboard.py            # refresh results/dashboard.html

Sections:
  1. lambda-interference explorer  — pure-JS re-implementation of the
     sin2chirp / sincos schedules (mbrl/regularization/schedule.py), live
     sliders, plus the actual logged penalty/lambda overlay.
  2. training pulse                — seed-averaged per-arm curves from
     results/runs/multitask-*/metrics.jsonl.
  3. transversality & d_eff        — recomputed per checkpoint of the newest
     multitask-reg lineage (R8 diagnostic).
  4. latent embedding density      — pairwise z histograms + true-reward scatter.
  5. reward-MLP 3D tensor          — activation / saliency tensors over
     reward / action / dynamics region bins.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

# section ids — tests/test_dashboard.py asserts all of these are present
SECTION_IDS = ["sec-lambda-explorer", "sec-training-pulse", "sec-transversality",
               "sec-latent-density", "sec-reward-tensor"]

# spec constants for the multitask Pendulum family (used as fallbacks; real
# shapes are inferred from each checkpoint's state_dict)
TASK_DIM = 1


# ---------------------------------------------------------------- utilities
def _py(o):
    """json.dumps default: numpy scalars/arrays -> plain python."""
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _flist(x) -> list[float]:
    return [float(v) for v in x]


# ---------------------------------------------------------------- run mirrors
def load_multitask_runs(root: Path) -> dict[str, dict]:
    """{run_name: {rows, meta, mtime}} from results/runs/multitask-*/metrics.jsonl.
    Torn (partially written) lines are skipped — the grid is live."""
    runs = {}
    for d in sorted((root / "results" / "runs").glob("multitask-*")):
        f = d / "metrics.jsonl"
        if not f.exists():
            continue
        rows = []
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue  # torn line mid-write
            if isinstance(row, dict):
                rows.append(row)
        meta = {}
        mf = d / "meta.json"
        if mf.exists():
            try:
                meta = json.loads(mf.read_text(errors="ignore"))
            except (json.JSONDecodeError, ValueError, OSError):
                pass
        if rows:
            runs[d.name] = {"rows": rows, "meta": meta, "mtime": f.stat().st_mtime}
    return runs


def arm_of(name: str, meta: dict) -> str:
    """multitask-reg-pendulum_target-s0 -> 'reg' (prefers meta['group'])."""
    group = meta.get("group") or name
    arm = re.sub(r"^multitask-?", "", str(group))
    arm = re.sub(r"-?pendulum_target.*$|-?halfcheetah_vel.*$|-s\d+$", "", arm)
    return arm or "base"


def seed_averaged(runs: dict, key: str, xkey: str) -> dict[str, dict]:
    """Per arm: x-aligned mean over seeds (x values are deterministic cadences,
    so they align; seeds at different progress just contribute where they have
    data). Returns {arm: {x, mean, n_seeds}}."""
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, run in runs.items():
        arm = arm_of(name, run["meta"])
        for row in run["rows"]:
            if key in row and xkey in row:
                try:
                    x, y = float(row[xkey]), float(row[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    acc[arm][x].append(y)
    out = {}
    for arm, d in acc.items():
        xs = sorted(d)
        out[arm] = {"x": xs,
                    "mean": [float(np.mean(d[x])) for x in xs],
                    "n": [len(d[x]) for x in xs]}
    return out


# ---------------------------------------------------------------- section 1
def build_lambda_section(runs: dict) -> dict:
    """Defaults for the JS explorer + the actual logged penalty/lambda overlay
    from the newest multitask-reg mirror."""
    data = {"defaults": {"kind": "sin2chirp", "lam0": 1e-3, "period0": 20000.0,
                         "period2": 10000.0, "period_end": 2000.0,
                         "total_steps": 100000, "floor": 1e-5, "t0": 10000.0},
            "overlay": None}
    reg = [(r["mtime"], n) for n, r in runs.items()
           if arm_of(n, r["meta"]) == "reg"]
    if reg:
        reg.sort()
        name = reg[-1][1]
        xs, ys = [], []
        for row in runs[name]["rows"]:
            if "penalty/lambda" in row and "step" in row:
                xs.append(float(row["step"]))
                ys.append(float(row["penalty/lambda"]))
        if xs:
            data["overlay"] = {"run": name, "step": xs, "lam": ys}
    return data


# ---------------------------------------------------------------- section 2
PULSE_PANELS = [
    {"title": "zero-shot returns vs env steps", "xkey": "env_steps",
     "keys": ["eval/return", "eval/zeroshot_interp", "eval/zeroshot_extrap"],
     "dashes": ["solid", "dash", "dot"], "logy": False,
     "xlabel": "env steps", "ylabel": "return"},
    {"title": "penalty value (log y) & lambda", "xkey": "step",
     "keys": ["penalty/value", "penalty/lambda"],
     "dashes": ["solid", "dot"], "logy": True, "y2": "penalty/lambda",
     "xlabel": "grad step", "ylabel": "penalty/value"},
    {"title": "imagined horizon", "xkey": "step",
     "keys": ["imagine/horizon"], "dashes": ["solid"], "logy": False,
     "xlabel": "grad step", "ylabel": "horizon"},
    {"title": "reward-head disagreement", "xkey": "step",
     "keys": ["model/reward_disagreement"], "dashes": ["solid"], "logy": False,
     "xlabel": "grad step", "ylabel": "disagreement"},
]


def build_pulse_section(runs: dict) -> dict:
    if not runs:
        return {"error": "no multitask run mirrors found under results/runs/"}
    panels = []
    for spec in PULSE_PANELS:
        series = {}
        for key in spec["keys"]:
            series[key] = seed_averaged(runs, key, spec["xkey"])
        panels.append({"spec": {k: v for k, v in spec.items()}, "series": series})
    arms = sorted({arm_of(n, r["meta"]) for n, r in runs.items()})
    return {"panels": panels, "arms": arms,
            "runs": sorted(runs), "n_runs": len(runs)}


# ---------------------------------------------------------------- checkpoints
def list_lineage(root: Path) -> tuple[str, list[Path]]:
    """Newest multitask-reg-* checkpoint lineage (by latest ckpt mtime) and its
    ckpt_step*.pt files sorted by step."""
    cands = []
    for hashdir in root.glob("checkpoints/multitask-reg-*/*"):
        if not hashdir.is_dir():
            continue
        pts = []
        for p in hashdir.glob("ckpt_step*.pt"):
            m = re.search(r"step(\d+)", p.name)
            if m:
                pts.append((int(m.group(1)), p))
        if pts:
            pts.sort()
            cands.append((max(p.stat().st_mtime for _, p in pts), hashdir, pts))
    if not cands:
        return "", []
    cands.sort()
    _, hashdir, pts = cands[-1]
    label = f"{hashdir.parent.name}/{hashdir.name}"
    return label, [p for _, p in pts]


def safe_load(path: Path):
    """torch.load tolerant of a checkpoint mid-write (training is live)."""
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except Exception:
        return None


def build_models(payload: dict):
    """Reconstruct encoder/dynamics/reward/policy from payload['trainer'];
    shapes inferred from the state dicts (multitask Pendulum: 3/1/1, h256 d2 k3)."""
    from mbrl.models import Encoder, AffineDynamics, RewardModel, Policy
    tr = payload["trainer"]
    enc_sd, dyn_sd, rew_sd, pol_sd = (tr[k] for k in
                                      ("encoder", "dynamics", "reward", "policy"))
    obs_dim = enc_sd["net.0.weight"].shape[1]
    k = dyn_sd["f.0.weight"].shape[1]                      # latent dim
    hidden = rew_sd["net.0.weight"].shape[0]
    depth = sum(1 for kk in enc_sd
                if re.fullmatch(r"net\.\d+\.weight", kk)
                and enc_sd[kk].dim() == 2) - 1
    n_heads = sum(1 for kk in rew_sd if re.fullmatch(r"heads\.\d+\.weight", kk))
    act_dim = rew_sd["net.0.weight"].shape[1] - k - TASK_DIM

    enc = Encoder(obs_dim, k, hidden, depth)
    enc.load_state_dict(enc_sd)
    dyn = AffineDynamics(k, act_dim, hidden, depth)
    dyn.load_state_dict(dyn_sd)
    rew = RewardModel(k, act_dim, hidden, depth, task_dim=TASK_DIM, n_heads=n_heads)
    rew.load_state_dict(rew_sd)
    pol = Policy(k, act_dim, hidden, depth, task_dim=TASK_DIM)
    pol.load_state_dict(pol_sd)
    for m in (enc, dyn, rew, pol):
        m.eval()
    return enc, dyn, rew, pol, k, act_dim


def collect_states(enc, pol, n_steps: int = 600, tau: float = 0.0,
                   seed: int = 0, n_envs: int = 8):
    """Roll PendulumTarget with the checkpoint policy (deterministic tanh(mu));
    returns (obs, actions, true rewards) as float32 arrays. Batched across
    n_envs parallel envs — batch-1 torch forwards dominate wall time otherwise
    (~8 ms each on CPU); batching keeps each checkpoint eval well under 5 s."""
    from mbrl.envs.tasks import make_task_env
    envs = [make_task_env("pendulum_target", tau) for _ in range(n_envs)]
    obs = np.stack([e.reset(seed=seed + 1000 * i)[0] for i, e in enumerate(envs)])
    tau_t = torch.full((n_envs, TASK_DIM), float(tau))
    steps = (n_steps + n_envs - 1) // n_envs
    obs_l, act_l, rew_l = [], [], []
    with torch.no_grad():
        for t in range(steps):
            z = enc(torch.as_tensor(obs, dtype=torch.float32))
            mu, _ = pol(z, tau_t)
            a = (torch.tanh(mu) * pol.action_scale).numpy()
            for i, e in enumerate(envs):
                obs2, r, term, trunc, _ = e.step(a[i])
                obs_l.append(np.asarray(obs[i], np.float32))
                act_l.append(np.asarray(a[i], np.float32))
                rew_l.append(float(r))
                obs[i] = obs2 if not (term or trunc) else e.reset(
                    seed=seed + 1000 * i + t + 1)[0]
    for e in envs:
        e.close()
    return (np.stack(obs_l[:n_steps]), np.stack(act_l[:n_steps]),
            np.asarray(rew_l[:n_steps], np.float32))


# ---------------------------------------------------------------- section 3
def build_transversality_section(root: Path, n_states: int, batch: int,
                                 probes_angle: int, probes_deff: int,
                                 max_ckpts: int) -> tuple[dict, dict | None]:
    """Per-checkpoint alpha(step) and d_eff(step) for the newest reg lineage.
    Also returns the NEWEST checkpoint's loaded apparatus for sections 4/5."""
    from mbrl.regularization.transversality import (transversality_angle,
                                                    effective_dim)
    label, pts = list_lineage(root)
    if not pts:
        return ({"error": "no checkpoints yet — rerun after training drops "
                          "checkpoints/multitask-reg-*/<hash>/ckpt_step*.pt"},
                None)
    if len(pts) > max_ckpts:  # keep the export fast under a long lineage
        idx = np.unique(np.linspace(0, len(pts) - 1, max_ckpts).astype(int))
        pts = [pts[i] for i in idx]
    steps, env_steps, angles, deffs = [], [], [], []
    newest = None
    for p in pts:
        payload = safe_load(p)
        if payload is None or "trainer" not in payload:
            continue  # partially written / unreadable — skip
        try:
            enc, dyn, rew, pol, k, m = build_models(payload)
            obs, act, r_true = collect_states(enc, pol, n_steps=n_states)
            with torch.no_grad():
                z = enc(torch.as_tensor(obs))
            rng = np.random.default_rng(0)
            sel = rng.choice(len(obs), size=min(batch, len(obs)), replace=False)
            x = torch.cat([z[sel], torch.as_tensor(act[sel]),
                           torch.zeros(len(sel), TASK_DIM)], dim=-1)
            fn_r = rew.on_concat                      # head mean on cat(z,a,tau)
            def fn_t(xx, _k=k, _m=m, _dyn=dyn):       # ignores the tau slice
                return _dyn(xx[..., :_k], xx[..., _k:_k + _m]).sum(-1)
            gen = torch.Generator().manual_seed(0)
            ang = transversality_angle(fn_r, fn_t, x, n_probes=probes_angle,
                                       generator=gen)
            deff = effective_dim(fn_r, x, n_probes=probes_deff, generator=gen)
        except Exception as e:  # never let one bad ckpt kill the page
            print(f"  [transversality] skipping {p.name}: {e}")
            continue
        step = int(payload["trainer"].get("step", 0))
        steps.append(step)
        env_steps.append(int(payload.get("env_steps", step)))
        angles.append(float(ang))
        deffs.append(float(deff))
        newest = {"path": str(p), "models": (enc, dyn, rew, pol, k, m),
                  "rollout": (obs, act, r_true, z)}
    if not steps:
        return ({"error": "checkpoints exist but none were readable (training "
                          "may be mid-write) — rerun in a minute"}, None)
    return ({"lineage": label, "step": steps, "env_steps": env_steps,
             "alpha_deg": angles, "d_eff": deffs,
             "n_ckpts": len(steps)}, newest)


# ---------------------------------------------------------------- section 4
def build_latent_section(newest: dict | None, n_hist_bins: int = 30) -> dict:
    if newest is None:
        return {"error": "no readable checkpoint — latent density needs the "
                         "newest encoder; rerun after training"}
    obs, act, r_true, z = newest["rollout"]
    zn = z.numpy()
    k = zn.shape[1]
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            H, xe, ye = np.histogram2d(zn[:, i], zn[:, j], bins=n_hist_bins)
            pairs.append({"i": i, "j": j, "hist": H.T,  # plotly z[row=y][col=x]
                          "xedges": xe, "yedges": ye})
    marginals = []
    for i in range(k):
        h, e = np.histogram(zn[:, i], bins=n_hist_bins)
        marginals.append({"i": i, "counts": h, "edges": e})
    return {"k": k, "pairs": pairs, "marginals": marginals,
            "z": zn, "reward": r_true, "ckpt": Path(newest["path"]).name,
            "n_states": len(zn)}


# ---------------------------------------------------------------- section 5
def build_tensor_section(newest: dict | None, n_bins: int = 8,
                         top: int = 96) -> dict:
    """A[layer,neuron,region] (mean SiLU activation) and S[layer,neuron,region]
    (mean |dR_mean/dh|) over region bins along three axes."""
    if newest is None:
        return {"error": "no readable checkpoint — the reward-MLP tensor needs "
                         "the newest reward model; rerun after training"}
    enc, dyn, rew, pol, k, m = newest["models"]
    obs, act, r_true, z = newest["rollout"]
    B = len(obs)
    a_t = torch.as_tensor(act)
    x = torch.cat([z, a_t, torch.zeros(B, TASK_DIM)], dim=-1)

    # --- capture trunk SiLU activations via forward hooks; grads via autograd
    captured: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        out.retain_grad()
        captured.append(out)

    handles = [mod.register_forward_hook(hook)
               for mod in rew.net if isinstance(mod, torch.nn.SiLU)]
    rew.zero_grad(set_to_none=True)
    out = rew.on_concat(x)            # head-mean R-hat (symlog space), (B,)
    out.sum().backward()
    for h in handles:
        h.remove()
    acts = [h.detach().numpy() for h in captured]            # (B, hidden) x L
    sals = [h.grad.abs().numpy() for h in captured]          # |dR/dh|
    n_layers = len(acts)

    # --- region bin index per sample, per axis
    def qbins(v):
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)
        return idx, edges

    with torch.no_grad():
        z2 = dyn(z, a_t)
        dz = torch.linalg.vector_norm(z2 - z, dim=-1).numpy()
    a1 = act[:, 0]
    axes = {}
    ridx, redges = qbins(r_true)
    axes["reward"] = {"idx": ridx, "edges": redges,
                      "label": "true reward (8 quantile bins)"}
    a_edges = np.linspace(a1.min(), a1.max() + 1e-9, n_bins + 1)
    aidx = np.clip(np.digitize(a1, a_edges) - 1, 0, n_bins - 1)
    axes["action"] = {"idx": aidx, "edges": a_edges,
                      "label": "action a (8 uniform bins)"}
    didx, dedges = qbins(dz)
    axes["dynamics"] = {"idx": didx, "edges": dedges,
                        "label": "|z' - z| dynamics step (8 quantile bins)"}

    # --- top neurons per layer by activation variance
    top = min(top, acts[0].shape[1])
    neuron_idx = []
    for l in range(n_layers):
        order = np.argsort(acts[l].var(axis=0))[::-1][:top]
        neuron_idx.append(order.astype(int))

    combos = {}
    for tname, mats in (("act", acts), ("sal", sals)):
        for aname, ax in axes.items():
            T = np.zeros((n_layers, top, n_bins), np.float32)
            for b in range(n_bins):
                mask = ax["idx"] == b
                if not mask.any():
                    continue
                for l in range(n_layers):
                    T[l, :, b] = mats[l][mask][:, neuron_idx[l]].mean(axis=0)
            combos[f"{tname}_{aname}"] = T
    return {"layers": n_layers, "top": top, "n_bins": n_bins,
            "neuron_idx": [ix.tolist() for ix in neuron_idx],
            "axis_labels": {n: ax["label"] for n, ax in axes.items()},
            "axis_edges": {n: _flist(ax["edges"]) for n, ax in axes.items()},
            "combos": {n: T for n, T in combos.items()},
            "ckpt": Path(newest["path"]).name, "n_states": B}


# ---------------------------------------------------------------- HTML
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>mbrl-curvature — multitask diagnostics</title>
<script src="__PLOTLY_CDN__"></script>
<style>
  :root { --bg:#10141a; --panel:#181f29; --ink:#dbe4ee; --dim:#8395aa;
          --accent:#5dade2; --edge:#26303d; }
  body { background:var(--bg); color:var(--ink); margin:0;
         font:14px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
  header { padding:22px 28px 6px; }
  h1 { font-size:21px; margin:0 0 2px; }
  h2 { font-size:16px; margin:0 0 8px; color:var(--accent); }
  .sub { color:var(--dim); font-size:12px; }
  section { background:var(--panel); border:1px solid var(--edge);
            border-radius:10px; margin:14px 22px; padding:16px 18px; }
  .caption { color:var(--dim); font-size:12px; margin-top:6px; max-width:980px; }
  .errnote { color:#e6a23c; background:#2a2113; border:1px solid #5c4716;
             border-radius:6px; padding:10px 12px; font-size:13px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
  .ctrl { display:grid; grid-template-columns:repeat(4,minmax(170px,1fr));
          gap:8px 18px; margin-bottom:10px; }
  .ctrl label { font-size:12px; color:var(--dim); display:block; }
  .ctrl input[type=range] { width:100%; }
  .ctrl select { width:100%; background:#0d1117; color:var(--ink);
                 border:1px solid var(--edge); border-radius:4px; padding:3px; }
  .val { color:var(--ink); font-variant-numeric:tabular-nums; }
  .plot { width:100%; }
  select.big { background:#0d1117; color:var(--ink); border:1px solid var(--edge);
               border-radius:4px; padding:4px 8px; margin-bottom:8px; }
</style>
</head>
<body>
<header>
  <h1>mbrl-curvature &mdash; multitask diagnostics dashboard</h1>
  <div class="sub">generated __GENERATED__ &middot; rerun
  <code>python scripts/make_dashboard.py</code> to refresh from the live grid</div>
</header>

<section id="sec-lambda-explorer">
  <h2>1 &middot; &lambda;-interference explorer</h2>
  <div class="ctrl">
    <div><label>kind</label>
      <select id="le-kind"><option value="sin2chirp">sin2chirp</option>
      <option value="sincos">sincos</option></select></div>
    <div><label>lam0 (log) <span class="val" id="le-lam0-v"></span></label>
      <input type="range" id="le-lam0" min="-4" max="0" step="0.01"></div>
    <div><label>period0 <span class="val" id="le-period0-v"></span></label>
      <input type="range" id="le-period0" min="1000" max="60000" step="500"></div>
    <div><label>period2 (sincos) <span class="val" id="le-period2-v"></span></label>
      <input type="range" id="le-period2" min="1000" max="60000" step="500"></div>
    <div><label>period_end (chirp) <span class="val" id="le-period_end-v"></span></label>
      <input type="range" id="le-period_end" min="200" max="20000" step="100"></div>
    <div><label>total_steps <span class="val" id="le-total-v"></span></label>
      <input type="range" id="le-total" min="10000" max="300000" step="1000"></div>
    <div><label>floor (log) <span class="val" id="le-floor-v"></span></label>
      <input type="range" id="le-floor" min="-6" max="-1" step="0.05"></div>
  </div>
  <div id="plot-lambda" class="plot" style="height:380px"></div>
  <div class="caption">Exact JS re-implementation of <code>LambdaSchedule</code>
  (schedule.py): <b>sin2chirp</b> = lam0&middot;(1&minus;t/T)&middot;sin&sup2;(chirped
  phase); <b>sincos</b> = lam0&middot;env&middot;&frac14;(sin 2&pi;t/p0 + cos
  2&pi;t/p2)&sup2; &mdash; beat period 1/|1/p0&minus;1/p2|. Both clamped at the
  floor. The grey static line is the actual logged <code>penalty/lambda</code>
  of the newest multitask-reg run (auto-dosed lam0 may sit far above the slider
  range &mdash; that is the dose, not a bug).</div>
</section>

<section id="sec-training-pulse">
  <h2>2 &middot; training pulse (seed-averaged per arm)</h2>
  <div id="pulse-body"></div>
  <div class="caption">Solid/dash/dot in the returns panel = train return /
  zero-shot interpolation / zero-shot extrapolation (report separately:
  smoothness only promises interpolation). Lines are means over the seeds that
  have reached each x; hover shows the seed count.</div>
</section>

<section id="sec-transversality">
  <h2>3 &middot; transversality angle &alpha; &amp; effective dimension d_eff</h2>
  <div id="trans-body"></div>
  <div class="caption">Recomputed per checkpoint of the newest multitask-reg
  lineage: &alpha; = angle between &nabla;&sup2;R and &nabla;&sup2;T in
  Frobenius space (R8; empirical range 60&ndash;71&deg;), d_eff = participation
  ratio of the reward Hessian spectrum (theory: the H&sup2; penalty pushes it
  down). States from on-policy PendulumTarget rollouts per checkpoint.</div>
</section>

<section id="sec-latent-density">
  <h2>4 &middot; latent embedding density (newest checkpoint)</h2>
  <div id="latent-body"></div>
  <div class="caption">Pairwise 2D histograms of the encoder latents z (k=3)
  over the on-policy rollout, with a scatter overlay colored by the TRUE
  PendulumTarget reward of each state; bottom row shows 1D marginals.</div>
</section>

<section id="sec-reward-tensor">
  <h2>5 &middot; reward-MLP 3D tensor (activations &amp; saliency)</h2>
  <div id="tensor-body"></div>
  <div class="caption" id="tensor-caption">How to read: each dot is one
  (layer, neuron, region) cell of the reward trunk. <b>x</b> = trunk layer,
  <b>y</b> = neuron rank (top-96 by activation variance; rank 0 = most
  variable), <b>z</b> = region bin along the chosen axis (reward quantiles,
  action bins, or dynamics-step |z'&minus;z| quantiles). Color = cell value
  (diverging), size &prop; |value|. <b>Activation</b> = mean SiLU output of
  that neuron over states in the region (what the neuron reports);
  <b>Saliency</b> = mean |&part;R&#770;_mean/&part;h| (how much the head-mean
  reward responds to that neuron there). Columns that stay hot across all
  region bins are task-general features; columns hot in only a few bins are
  region-specialized. If the curvature penalty is doing its job, saliency
  should concentrate on few neurons and vary smoothly along the region
  axis.</div>
</section>

<script>
const DATA = __DATA_JSON__;
const ARMCOLORS = {reg:"#5dade2", lam0:"#e67e22", notask:"#2ecc71", base:"#9b59b6"};
const FONT = {color:"#dbe4ee", size:11};
const LAYOUT = {paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"#10141a",
  font:FONT, margin:{l:55,r:55,t:34,b:42}};
function lay(extra){ return Object.assign(JSON.parse(JSON.stringify(LAYOUT)), extra); }
function armColor(a){ return ARMCOLORS[a] || "#95a5a6"; }
function note(el, msg){ document.getElementById(el).innerHTML =
  '<div class="errnote">'+msg+'</div>'; }

/* ---------------- section 1: lambda explorer ---------------- */
(function(){
  const D = DATA.lambda, P = D.defaults;
  const $ = id => document.getElementById(id);
  $("le-kind").value = P.kind;
  $("le-lam0").value = Math.log10(P.lam0);
  $("le-period0").value = P.period0;
  $("le-period2").value = P.period2;
  $("le-period_end").value = P.period_end;
  $("le-total").value = P.total_steps;
  $("le-floor").value = Math.log10(P.floor);

  // EXACT port of LambdaSchedule.__call__ for sin2chirp / sincos
  function lambdaAt(t, p){
    let lam;
    if (p.kind === "sin2chirp"){
      const f0 = 1.0 / p.period0;
      let env, phase;
      if (p.total_steps){
        const frac = Math.min(t / p.total_steps, 1.0);
        env = 1.0 - frac;
        const f1 = 1.0 / p.period_end;
        phase = 2 * Math.PI * t * (f0 + 0.5 * (f1 - f0) * frac);
      } else {
        env = p.t0 / (p.t0 + t);
        phase = 2 * Math.PI * f0 * t;
      }
      lam = p.lam0 * env * Math.pow(Math.sin(phase), 2);
    } else {  // sincos
      let env;
      if (p.total_steps){
        env = Math.max(1.0 - Math.min(t / p.total_steps, 1.0), 0.0);
      } else {
        env = p.t0 / (p.t0 + t);
      }
      const s = Math.sin(2 * Math.PI * t / p.period0);
      const c = Math.cos(2 * Math.PI * t / p.period2);
      lam = p.lam0 * env * 0.25 * Math.pow(s + c, 2);
    }
    return Math.max(lam, p.floor);
  }

  function params(){
    return {kind: $("le-kind").value,
            lam0: Math.pow(10, parseFloat($("le-lam0").value)),
            period0: parseFloat($("le-period0").value),
            period2: parseFloat($("le-period2").value),
            period_end: parseFloat($("le-period_end").value),
            total_steps: parseFloat($("le-total").value),
            floor: Math.pow(10, parseFloat($("le-floor").value)),
            t0: P.t0};
  }
  function fmt(v){ return v >= 100 ? v.toFixed(0) : v.toPrecision(3); }
  function redraw(){
    const p = params();
    $("le-lam0-v").textContent = p.lam0.toExponential(2);
    $("le-period0-v").textContent = fmt(p.period0);
    $("le-period2-v").textContent = fmt(p.period2);
    $("le-period_end-v").textContent = fmt(p.period_end);
    $("le-total-v").textContent = fmt(p.total_steps);
    $("le-floor-v").textContent = p.floor.toExponential(2);
    const N = 1200, xs = new Array(N), ys = new Array(N);
    for (let i = 0; i < N; i++){
      const t = i * p.total_steps / (N - 1);
      xs[i] = t; ys[i] = lambdaAt(t, p);
    }
    const traces = [{x: xs, y: ys, mode: "lines", name: "λ(t) explorer",
                     line: {color: "#5dade2", width: 1.6}}];
    if (D.overlay){
      traces.push({x: D.overlay.step, y: D.overlay.lam, mode: "lines",
        name: "logged penalty/lambda ("+D.overlay.run+")",
        line: {color: "#8395aa", width: 1.2, dash: "dot"}});
    }
    Plotly.react("plot-lambda", traces, lay({
      xaxis: {title: "grad step t", gridcolor: "#26303d"},
      yaxis: {title: "λ", type: "log", gridcolor: "#26303d"},
      legend: {orientation: "h", y: 1.12}}), {displayModeBar: false});
  }
  ["le-kind","le-lam0","le-period0","le-period2","le-period_end","le-total",
   "le-floor"].forEach(id => $(id).addEventListener("input", redraw));
  redraw();
})();

/* ---------------- section 2: training pulse ---------------- */
(function(){
  const D = DATA.pulse;
  if (D.error){ note("pulse-body", D.error); return; }
  const body = document.getElementById("pulse-body");
  body.innerHTML = '<div class="grid2">' + D.panels.map((p, i) =>
    '<div id="pulse-'+i+'" style="height:300px"></div>').join("") + '</div>' +
    '<div class="sub">runs: '+D.runs.join(", ")+'</div>';
  const DASH = {};
  D.panels.forEach((panel, i) => {
    const spec = panel.spec, traces = [];
    spec.keys.forEach((key, ki) => {
      const byArm = panel.series[key] || {};
      Object.keys(byArm).sort().forEach(arm => {
        const s = byArm[arm];
        if (!s.x.length) return;
        traces.push({x: s.x, y: s.mean, mode: "lines",
          name: arm + " · " + key.split("/").pop(),
          yaxis: (spec.y2 === key) ? "y2" : "y",
          customdata: s.n,
          hovertemplate: "%{y:.4g} (n=%{customdata})<extra>"+arm+" "+key+"</extra>",
          line: {color: armColor(arm), width: 1.5,
                 dash: spec.dashes[ki] || "solid"}});
      });
    });
    const layout = lay({title: {text: spec.title, font: {size: 13}},
      xaxis: {title: spec.xlabel, gridcolor: "#26303d"},
      yaxis: {title: spec.ylabel, gridcolor: "#26303d",
              type: spec.logy ? "log" : "linear"},
      showlegend: true, legend: {font: {size: 9}}});
    if (spec.y2) layout.yaxis2 = {title: "λ", overlaying: "y",
                                  side: "right", showgrid: false};
    Plotly.newPlot("pulse-"+i, traces, layout, {displayModeBar: false});
  });
})();

/* ---------------- section 3: transversality ---------------- */
(function(){
  const D = DATA.trans;
  if (D.error){ note("trans-body", D.error); return; }
  document.getElementById("trans-body").innerHTML =
    '<div id="plot-trans" style="height:340px"></div>' +
    '<div class="sub">lineage: '+D.lineage+' &middot; '+D.n_ckpts+
    ' checkpoint(s)</div>';
  Plotly.newPlot("plot-trans", [
    {x: D.env_steps, y: D.alpha_deg, mode: "lines+markers",
     name: "α (deg)", line: {color: "#5dade2"}},
    {x: D.env_steps, y: D.d_eff, mode: "lines+markers", name: "d_eff",
     yaxis: "y2", line: {color: "#e67e22"}},
  ], lay({xaxis: {title: "env steps", gridcolor: "#26303d"},
    yaxis: {title: "transversality α (degrees)", gridcolor: "#26303d",
            range: [0, 95]},
    yaxis2: {title: "d_eff", overlaying: "y", side: "right", showgrid: false},
    legend: {orientation: "h", y: 1.12},
    shapes: [{type: "rect", xref: "paper", x0: 0, x1: 1, y0: 60, y1: 71,
              fillcolor: "rgba(93,173,226,0.07)", line: {width: 0}}]}),
    {displayModeBar: false});
})();

/* ---------------- section 4: latent density ---------------- */
(function(){
  const D = DATA.latent;
  if (D.error){ note("latent-body", D.error); return; }
  const body = document.getElementById("latent-body");
  let html = '<div class="grid3">';
  D.pairs.forEach((p, i) => { html += '<div id="lat-'+i+'" style="height:300px"></div>'; });
  html += '</div><div class="grid3">';
  D.marginals.forEach((m, i) => { html += '<div id="latm-'+i+'" style="height:170px"></div>'; });
  html += '</div><div class="sub">checkpoint: '+D.ckpt+' &middot; '+
          D.n_states+' on-policy states</div>';
  body.innerHTML = html;
  function mid(e){ const o = []; for (let i = 0; i < e.length - 1; i++)
    o.push(0.5 * (e[i] + e[i + 1])); return o; }
  D.pairs.forEach((p, i) => {
    Plotly.newPlot("lat-"+i, [
      {type: "heatmap", z: p.hist, x: mid(p.xedges), y: mid(p.yedges),
       colorscale: "Viridis", showscale: false},
      {type: "scatter", mode: "markers",
       x: D.z.map(r => r[p.i]), y: D.z.map(r => r[p.j]),
       marker: {size: 3, color: D.reward, colorscale: "Portland",
                opacity: 0.55, colorbar: (i === D.pairs.length - 1) ?
                {title: {text: "true r", font: FONT}, thickness: 10} : undefined},
       hovertemplate: "z"+p.i+"=%{x:.2f} z"+p.j+"=%{y:.2f} r=%{marker.color:.2f}<extra></extra>",
       name: "states"}],
      lay({title: {text: "z"+p.i+" vs z"+p.j, font: {size: 12}},
        xaxis: {title: "z"+p.i, gridcolor: "#26303d"},
        yaxis: {title: "z"+p.j, gridcolor: "#26303d"}, showlegend: false}),
      {displayModeBar: false});
  });
  D.marginals.forEach((m, i) => {
    Plotly.newPlot("latm-"+i, [{type: "bar", x: mid(m.edges), y: m.counts,
      marker: {color: "#5dade2"}}],
      lay({title: {text: "z"+m.i+" marginal", font: {size: 11}},
        margin: {l: 40, r: 10, t: 26, b: 28},
        xaxis: {gridcolor: "#26303d"}, yaxis: {gridcolor: "#26303d"},
        bargap: 0}), {displayModeBar: false});
  });
})();

/* ---------------- section 5: reward-MLP tensor ---------------- */
(function(){
  const D = DATA.tensor;
  if (D.error){ note("tensor-body", D.error); return; }
  const body = document.getElementById("tensor-body");
  body.innerHTML = '<select class="big" id="tensor-combo">' +
    [["act_reward","Activation × reward bins"],
     ["act_action","Activation × action bins"],
     ["act_dynamics","Activation × dynamics bins"],
     ["sal_reward","Saliency × reward bins"],
     ["sal_action","Saliency × action bins"],
     ["sal_dynamics","Saliency × dynamics bins"]]
    .map(o => '<option value="'+o[0]+'">'+o[1]+'</option>').join("") +
    '</select><div id="plot-tensor" style="height:560px"></div>' +
    '<div class="sub">checkpoint: '+D.ckpt+' &middot; top '+D.top+
    ' neurons/layer by activation variance &middot; '+D.n_states+' states</div>';
  function draw(){
    const name = document.getElementById("tensor-combo").value;
    const T = D.combos[name], isSal = name.startsWith("sal");
    const axis = name.split("_")[1];
    const xs = [], ys = [], zs = [], cs = [], txt = [];
    let vmax = 1e-12;
    for (let l = 0; l < D.layers; l++)
      for (let n = 0; n < D.top; n++)
        for (let b = 0; b < D.n_bins; b++){
          const v = T[l][n][b];
          xs.push(l); ys.push(n); zs.push(b); cs.push(v);
          txt.push("layer "+l+" · neuron #"+D.neuron_idx[l][n]+
                   " (rank "+n+") · bin "+b+" · "+v.toPrecision(3));
          vmax = Math.max(vmax, Math.abs(v));
        }
    const sizes = cs.map(v => 1.5 + 7 * Math.abs(v) / vmax);
    Plotly.react("plot-tensor", [{type: "scatter3d", mode: "markers",
      x: xs, y: ys, z: zs, text: txt, hoverinfo: "text",
      marker: {size: sizes, color: cs, colorscale: "RdBu", reversescale: true,
        cmin: isSal ? 0 : -vmax, cmax: vmax,
        colorbar: {title: {text: isSal ? "|∂R̂/∂h|" : "mean act",
                   font: FONT}, thickness: 12}, opacity: 0.85}}],
      lay({scene: {
        xaxis: {title: "trunk layer", tickvals: [0, 1], gridcolor: "#26303d",
                backgroundcolor: "rgba(0,0,0,0)"},
        yaxis: {title: "neuron rank (by act var)", gridcolor: "#26303d",
                backgroundcolor: "rgba(0,0,0,0)"},
        zaxis: {title: D.axis_labels[axis], gridcolor: "#26303d",
                backgroundcolor: "rgba(0,0,0,0)"},
        camera: {eye: {x: 1.7, y: 1.4, z: 0.9}}},
        margin: {l: 0, r: 0, t: 10, b: 0}}), {displayModeBar: false});
  }
  document.getElementById("tensor-combo").addEventListener("change", draw);
  draw();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results" / "dashboard.html"))
    ap.add_argument("--n-states", type=int, default=600,
                    help="rollout states per checkpoint")
    ap.add_argument("--batch", type=int, default=256,
                    help="Hessian-probe batch size")
    ap.add_argument("--probes-angle", type=int, default=8)
    ap.add_argument("--probes-deff", type=int, default=32)
    ap.add_argument("--max-ckpts", type=int, default=12,
                    help="evenly subsample the lineage beyond this many ckpts")
    args = ap.parse_args(argv)
    t0 = time.time()
    torch.manual_seed(0)
    # tiny batch-1/256 CPU ops thrash under torch's default thread pool — the
    # whole export is ~3x faster single-threaded (measured)
    torch.set_num_threads(1)

    data: dict = {}
    runs = {}
    try:
        runs = load_multitask_runs(ROOT)
    except Exception as e:
        print(f"[runs] failed: {e}")

    try:
        data["lambda"] = build_lambda_section(runs)
    except Exception as e:
        data["lambda"] = {"defaults": {"kind": "sin2chirp", "lam0": 1e-3,
                                       "period0": 20000.0, "period2": 10000.0,
                                       "period_end": 2000.0,
                                       "total_steps": 100000, "floor": 1e-5,
                                       "t0": 10000.0},
                          "overlay": None}
        print(f"[lambda] overlay failed: {e}")

    try:
        data["pulse"] = build_pulse_section(runs)
    except Exception as e:
        data["pulse"] = {"error": f"training-pulse build failed: {e}"}
    print(f"[pulse] {len(runs)} run mirror(s)  t={time.time()-t0:.1f}s")

    newest = None
    try:
        data["trans"], newest = build_transversality_section(
            ROOT, args.n_states, args.batch, args.probes_angle,
            args.probes_deff, args.max_ckpts)
    except Exception as e:
        data["trans"] = {"error": f"transversality build failed: {e}"}
    print(f"[trans] {data['trans'].get('n_ckpts', 0)} ckpt(s)  "
          f"t={time.time()-t0:.1f}s")

    try:
        data["latent"] = build_latent_section(newest)
    except Exception as e:
        data["latent"] = {"error": f"latent-density build failed: {e}"}

    try:
        data["tensor"] = build_tensor_section(newest)
    except Exception as e:
        data["tensor"] = {"error": f"reward-tensor build failed: {e}"}
    print(f"[latent+tensor] done  t={time.time()-t0:.1f}s")

    html = (HTML_TEMPLATE
            .replace("__PLOTLY_CDN__", PLOTLY_CDN)
            .replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M:%S"))
            .replace("__DATA_JSON__", json.dumps(data, default=_py)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(html)
    tmp.replace(out)  # atomic-ish: never leave a torn dashboard
    print(f"[done] wrote {out} ({out.stat().st_size/1024:.0f} KB) "
          f"in {time.time()-t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
