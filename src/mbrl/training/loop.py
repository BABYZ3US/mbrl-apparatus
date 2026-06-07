"""The training loop from framework 2.6, instrumented for W&B.

Model learning (GPU-bound): fit dynamics + reward with the isotropic 2-probe
Hutchinson curvature penalty in latent coords, lambda annealed per R12.
Behaviour learning: Dreamer-style — imagine with the policy through the learned
model, score with lambda-returns against an EMA target value net, backprop the
policy gradient directly through dynamics + reward (cheap: no critic ensembles;
the curvature penalty keeps the imagined surface smooth enough to differentiate
through, R15). Optional task conditioning (task_dim > 0) for multi-task runs.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..models import Encoder, EMAEncoder, AffineDynamics, RewardModel, Policy, ValueFn
from ..regularization.hutchinson import hvp_penalty, laplacian_trace_penalty
from ..regularization.schedule import LambdaSchedule
from ..training.returns import lambda_returns
from ..training.smoothing import smooth_rewards
from ..utils.seeding import make_generator


class Trainer:
    def __init__(self, cfg, obs_dim: int, action_dim: int, device: str = "cpu",
                 task_dim: int = 0):
        self.cfg, self.device, self.task_dim = cfg, torch.device(device), task_dim
        k, h, d = cfg.model.latent_dim, cfg.model.hidden, cfg.model.depth
        self.encoder = Encoder(obs_dim, k, h, d).to(device)
        self.ema = EMAEncoder(self.encoder, cfg.model.ema_decay)
        self.dynamics = AffineDynamics(k, action_dim, h, d).to(device)
        self.reward = RewardModel(k, action_dim, h, d, task_dim=task_dim).to(device)
        self.policy = Policy(k, action_dim, h, d, task_dim=task_dim).to(device)
        self.value = ValueFn(k, h, d, task_dim=task_dim).to(device)
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)

        self.model_opt = torch.optim.AdamW(
            [*self.encoder.parameters(), *self.dynamics.parameters(), *self.reward.parameters()],
            lr=cfg.optim.model_lr)
        self.policy_opt = torch.optim.AdamW(self.policy.parameters(), lr=cfg.optim.policy_lr)
        self.value_opt = torch.optim.AdamW(self.value.parameters(), lr=cfg.optim.value_lr)

        self.lam = LambdaSchedule(**cfg.penalty.schedule)
        self.step = 0
        self.gen = make_generator(self.device, cfg.seed)
        # Dreamer-V3-style return scale: EMA of the lambda-returns' 5-95%
        # range; policy gradient uses returns / max(1, scale). Fixes the
        # iteration-6 imagined-return variance explosion (analysis_multitask_01
        # F1) and was complementary to the curvature penalty in the original
        # experiments (user-reported; not in the founding doc).
        self.ret_scale = 1.0

    # ---------------- model learning ----------------
    def model_update(self, batch) -> dict:
        if self.task_dim:
            obs, a, r, obs_next, tau = (x.to(self.device) for x in batch)
        else:
            obs, a, r, obs_next = (x.to(self.device) for x in batch)
            tau = None
        z = self.encoder(obs)
        with torch.no_grad():
            z_next_tgt = self.ema(obs_next)

        dyn_loss = F.mse_loss(self.dynamics(z, a), z_next_tgt)
        rew_loss = F.mse_loss(self.reward(z, a, tau), r)

        # Isotropic curvature penalty in joint latent(-task) coords (R4, R16);
        # detached coords: penalize R's surface geometry, not the encoder through it.
        # Including tau in the Hessian coords enforces smooth interpolation
        # BETWEEN tasks — the multi-task generalization lever.
        parts = [z.detach(), a]
        if tau is not None and self.cfg.penalty.get("include_task", True):
            parts.append(tau)
        x_pen = torch.cat(parts, dim=-1)
        if tau is not None and not self.cfg.penalty.get("include_task", True):
            fn = lambda x: self.reward.net(
                torch.cat([x, tau.detach()], dim=-1)).squeeze(-1)
        else:
            fn = self.reward.on_concat
        # form dispatch (R5: same Euler-Lagrange; both unbiased at >=2 probes).
        # 'laplacian_trace' + a clamped decaying schedule (sin2chirp) is the
        # user's narrowed-down active ingredient from the original experiments.
        if self.cfg.penalty.get("form", "frobenius") == "laplacian_trace":
            import functools
            penalty_fn = functools.partial(
                laplacian_trace_penalty,
                clamp=self.cfg.penalty.get("clamp_trace", True))
        else:
            penalty_fn = hvp_penalty
        pen = penalty_fn(fn, x_pen, n_probes=self.cfg.penalty.n_probes,
                         generator=self.gen)
        if self.cfg.penalty.penalize_dynamics:  # optional transversal term (R8/R9)
            k = z.shape[-1]
            fn_t = lambda x: self.dynamics(x[..., :k], x[..., k:k + a.shape[-1]]).sum(-1)
            za = torch.cat([z.detach(), a], dim=-1)
            pen = pen + penalty_fn(fn_t, za, n_probes=self.cfg.penalty.n_probes,
                                   generator=self.gen)

        lam_t = self.lam(self.step)
        loss = dyn_loss + rew_loss + lam_t * pen

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 100.0)
        self.model_opt.step()
        self.ema.update(self.encoder)
        self.step += 1
        return {"loss/dyn": dyn_loss.item(), "loss/reward": rew_loss.item(),
                "penalty/value": pen.item(), "penalty/lambda": lam_t,
                "loss/total": loss.item(), "step": self.step}

    # ---------------- behaviour learning (Dreamer lambda-returns) ----------------
    def behaviour_update(self, z0: torch.Tensor, tau0: torch.Tensor | None = None) -> dict:
        cfg_i = self.cfg.imagination
        H, gamma, lam_ret = cfg_i.horizon, cfg_i.gamma, cfg_i.get("lambda_", 0.95)
        ent_coef = cfg_i.get("entropy_coef", 3e-4)

        # --- differentiable imagination (gradients flow through T and R) ---
        zs, rs, logps = [z0], [], []
        z = z0
        for _ in range(H):
            a, logp = self.policy.sample(z, tau0)
            z = self.dynamics(z, a)
            zs.append(z)
            rs.append(self.reward(zs[-2], a, tau0))
            logps.append(logp)
        zs = torch.stack(zs)                      # (H+1, B, k)
        rs = smooth_rewards(torch.stack(rs), self.cfg.smoothing)  # (H, B)
        logps = torch.stack(logps)                # (H, B)

        with torch.no_grad():
            flat = zs.reshape(-1, zs.shape[-1])
            tgt_tau = tau0.repeat(H + 1, 1) if tau0 is not None else None
            v_tgt = self.value_target(flat, tgt_tau).reshape(H + 1, -1)
        returns = lambda_returns(rs, v_tgt, gamma, lam_ret)       # (H, B)

        # --- return normalization (Dreamer-V3): scale-invariant policy gradient
        with torch.no_grad():
            lo = torch.quantile(returns.detach().float(), 0.05)
            hi = torch.quantile(returns.detach().float(), 0.95)
            decay = cfg_i.get("ret_scale_decay", 0.99)
            self.ret_scale = decay * self.ret_scale + (1 - decay) * float(hi - lo)
        norm = max(1.0, self.ret_scale)

        # --- policy: maximize normalized lambda-returns + entropy
        #     (never curvature-penalized, R10)
        entropy = -logps.mean()
        pi_loss = -(returns / norm).mean() - ent_coef * entropy
        self.policy_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 100.0)
        self.policy_opt.step()

        # --- value: regress to lambda-returns on detached latents ---
        flat = zs[:-1].detach().reshape(-1, zs.shape[-1])
        v_tau = tau0.repeat(H, 1) if tau0 is not None else None
        v = self.value(flat, v_tau).reshape(H, -1)
        v_loss = F.mse_loss(v, returns.detach())
        self.value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.value_opt.step()

        # --- EMA target value ---
        decay = cfg_i.get("value_target_decay", 0.98)
        with torch.no_grad():
            for pt, p in zip(self.value_target.parameters(), self.value.parameters()):
                pt.lerp_(p, 1.0 - decay)

        return {"loss/value": v_loss.item(), "loss/policy": pi_loss.item(),
                "policy/entropy": entropy.item(),
                "policy/ret_scale": self.ret_scale,
                "imagine/return_mean": returns.mean().item(),
                "imagine/return_var": returns.var().item()}  # R15 diagnostic

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, horizon: int, tau0: torch.Tensor | None = None):
        zs, as_, rs = [z0], [], []
        z = z0
        for _ in range(horizon):
            a, _ = self.policy.sample(z, tau0)
            z = self.dynamics(z, a)
            zs.append(z); as_.append(a); rs.append(self.reward(zs[-2], a, tau0))
        return torch.stack(zs), torch.stack(as_), torch.stack(rs)

    # ---------------- checkpoint protocol ----------------
    def state_dict(self):
        return {"encoder": self.encoder.state_dict(), "ema": self.ema.state_dict(),
                "dynamics": self.dynamics.state_dict(), "reward": self.reward.state_dict(),
                "policy": self.policy.state_dict(), "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "model_opt": self.model_opt.state_dict(),
                "policy_opt": self.policy_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(), "step": self.step,
                "ret_scale": self.ret_scale,
                "hutchinson_gen": self.gen.get_state()}

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"]); self.ema.load_state_dict(sd["ema"])
        self.dynamics.load_state_dict(sd["dynamics"]); self.reward.load_state_dict(sd["reward"])
        self.policy.load_state_dict(sd["policy"]); self.value.load_state_dict(sd["value"])
        if "value_target" in sd:
            self.value_target.load_state_dict(sd["value_target"])
        self.model_opt.load_state_dict(sd["model_opt"])
        self.policy_opt.load_state_dict(sd["policy_opt"])
        self.value_opt.load_state_dict(sd["value_opt"])
        self.step = sd["step"]
        self.ret_scale = sd.get("ret_scale", 1.0)
        if "hutchinson_gen" in sd:  # probe RNG must resume too (bitwise resume)
            self.gen.set_state(sd["hutchinson_gen"])
