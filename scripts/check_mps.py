"""Apple-Silicon (MPS) capability check + benchmark for this project.

Run on the Mac:  python scripts/check_mps.py

Checks, in order of importance:
  1. The Hutchinson penalty's DOUBLE BACKWARD works on MPS (the deal-breaker —
     MPS op coverage for second derivatives is the usual gap).
  2. Penalty values agree CPU vs MPS (fp32) within tolerance.
  3. Full Trainer.model_update + behaviour_update run on MPS.
  4. Benchmark: model_update CPU vs MPS at realistic batch sizes -> verdict on
     when the M2 GPU actually pays (kernel-launch overhead dominates small nets).

If (1) fails on your torch version: train on CPU locally (still fine for
Pendulum-class work) or retry after upgrading torch — do NOT set
PYTORCH_ENABLE_MPS_FALLBACK=1 for real runs; silent CPU fallback inside the
penalty destroys the timing wins and can change numerics mid-graph.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mbrl.regularization.hutchinson import hvp_penalty
from mbrl.utils.seeding import make_generator

OK, FAIL = "\033[92mOK\033[0m", "\033[91mFAIL\033[0m"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def make_net(width=256, depth=2, d_in=10, device="cpu"):
    layers, sizes = [], [d_in] + [width] * depth + [1]
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers).to(device)


def check_double_backward(device: str) -> bool:
    torch.manual_seed(0)
    net = make_net(device=device)
    x = torch.randn(64, 10, device=device)
    gen = make_generator(device, 0)
    try:
        pen = hvp_penalty(lambda x: net(x).squeeze(-1), x, n_probes=2, generator=gen)
        pen.backward()  # third differentiation level: d(penalty)/d(params)
        sync(device)
        grads_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        return bool(torch.isfinite(pen).item()) and grads_ok
    except Exception as e:
        print(f"   ({type(e).__name__}: {str(e)[:120]})")
        return False


def check_agreement() -> bool:
    torch.manual_seed(0)
    net_cpu = make_net(device="cpu")
    net_mps = make_net(device="mps")
    net_mps.load_state_dict(net_cpu.state_dict())
    x = torch.randn(256, 10)
    gen = make_generator("cpu", 0)  # same CPU probe stream for both
    p_cpu = hvp_penalty(lambda x: net_cpu(x).squeeze(-1), x, n_probes=2,
                        generator=gen, create_graph=False)
    gen = make_generator("cpu", 0)
    p_mps = hvp_penalty(lambda x: net_mps(x).squeeze(-1), x.to("mps"), n_probes=2,
                        generator=gen, create_graph=False)
    rel = abs(p_cpu.item() - p_mps.item()) / max(abs(p_cpu.item()), 1e-9)
    print(f"   penalty cpu={p_cpu.item():.6g} mps={p_mps.item():.6g} rel_err={rel:.2e}")
    return rel < 1e-3


def check_trainer(device: str) -> bool:
    from omegaconf import OmegaConf
    from mbrl.training import Trainer
    cfg = OmegaConf.create({
        "seed": 0,
        "model": {"latent_dim": 4, "hidden": 256, "depth": 2, "ema_decay": 0.99},
        "penalty": {"n_probes": 2, "penalize_dynamics": False,
                    "schedule": {"kind": "cuberoot", "lam0": 1e-3, "t0": 100,
                                 "floor": 1e-6}},
        "smoothing": {"enabled": True, "sigma": 1.5},
        "imagination": {"horizon": 15, "gamma": 0.99},
        "optim": {"model_lr": 3e-4, "policy_lr": 1e-4, "value_lr": 3e-4},
    })
    try:
        t = Trainer(cfg, obs_dim=17, action_dim=6, device=device)
        batch = (torch.randn(256, 17), torch.randn(256, 6),
                 torch.randn(256), torch.randn(256, 17))
        m = t.model_update(batch)
        z0 = torch.randn(256, 4, device=device)
        b = t.behaviour_update(z0)
        sync(device)
        import math
        return all(math.isfinite(v) for v in (*m.values(), *b.values()))
    except Exception as e:
        print(f"   ({type(e).__name__}: {str(e)[:120]})")
        return False


def bench(device: str, batch: int, iters: int = 30) -> float:
    torch.manual_seed(0)
    net = make_net(device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    x = torch.randn(batch, 10, device=device)
    y = torch.randn(batch, device=device)
    gen = make_generator(device, 0)
    for _ in range(5):  # warmup (MPS compiles kernels on first use)
        loss = torch.nn.functional.mse_loss(net(x).squeeze(-1), y) \
            + 1e-3 * hvp_penalty(lambda x: net(x).squeeze(-1), x, 2, gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = torch.nn.functional.mse_loss(net(x).squeeze(-1), y) \
            + 1e-3 * hvp_penalty(lambda x: net(x).squeeze(-1), x, 2, gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000  # ms/update


def main():
    print(f"torch {torch.__version__}")
    if not torch.backends.mps.is_available():
        print(f"{FAIL} MPS not available "
              "(needs macOS 12.3+, arm64 python, torch>=1.12). CPU it is.")
        return
    print(f"{OK} MPS available\n")

    print("1) double backward (penalty -> param grads) on MPS:")
    db = check_double_backward("mps")
    print(f"   {OK if db else FAIL}")
    if not db:
        print("   -> verdict: keep device=cpu locally; Colab for GPU. (See docstring.)")
        return

    print("2) CPU/MPS penalty agreement:")
    print(f"   {OK if check_agreement() else FAIL}")

    print("3) full Trainer step on MPS:")
    print(f"   {OK if check_trainer('mps') else FAIL}")

    print("\n4) benchmark, ms per model_update-style step (penalty included):")
    print(f"   {'batch':>8} {'cpu':>10} {'mps':>10} {'speedup':>9}")
    verdict_batch = None
    for batch in (64, 256, 1024, 4096):
        c, m = bench("cpu", batch), bench("mps", batch)
        sp = c / m
        if sp > 1.2 and verdict_batch is None:
            verdict_batch = batch
        print(f"   {batch:>8} {c:>9.2f} {m:>9.2f} {sp:>8.2f}x")

    print("\nVerdict:")
    if verdict_batch:
        print(f" - MPS pays off from batch ~{verdict_batch}; "
              f"set device=mps and optim.batch_size>={max(verdict_batch, 256)}.")
    else:
        print(" - CPU wins at these sizes (launch overhead); keep device=cpu locally.")
    print(" - Env stepping & joblib sweeps stay on CPU cores either way;")
    print("   MPS only accelerates model/behaviour learning.")
    print(" - Keep optim.amp=false on MPS; penalty is fp32 regardless.")


if __name__ == "__main__":
    main()
