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

import numpy as np
import torch
import torch.nn.functional as F

from ..models import Encoder, EMAEncoder, AffineDynamics, RewardModel, Policy, ValueFn
from ..models.reward import symlog, symexp
from ..regularization.hutchinson import hvp_penalty, laplacian_trace_penalty
from ..regularization.schedule import LambdaSchedule
from ..training.returns import lambda_returns
from ..training.smoothing import smooth_rewards
from ..utils.seeding import make_generator


class Trainer:
    def __init__(self, cfg, obs_dim: int, action_dim: int, device: str = "cpu",
                 task_dim: int = 0):
        self.cfg, self.device, self.task_dim = cfg, torch.device(device), task_dim
        # latent can be at most the input dimension (user rule); keep it small
        k = min(cfg.model.latent_dim, obs_dim)
        h, d = cfg.model.hidden, cfg.model.depth
        self.encoder = Encoder(obs_dim, k, h, d).to(device)
        self.ema = EMAEncoder(self.encoder, cfg.model.ema_decay)
        self.dynamics = AffineDynamics(k, action_dim, h, d).to(device)
        self.symlog = bool(cfg.model.get("symlog_reward", False))
        self.reward = RewardModel(k, action_dim, h, d, task_dim=task_dim,
                                  n_heads=int(cfg.model.get("reward_heads", 1))).to(device)
        self.pessimism = float(cfg.imagination.get("pessimism", 0.0))
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

        # Auto-dosed lambda (penalty.auto_dose): during the first
        # warmup_updates model updates run with lam=0 while accumulating mean
        # fit loss and mean penalty; then dose lam0 = target_ratio * fit / pen.
        ad = cfg.penalty.get("auto_dose", None)
        self.ad_enabled = bool(ad and ad.get("enabled", False))
        self.ad_target_ratio = float(ad.get("target_ratio", 0.1)) if ad else 0.1
        self.ad_warmup = int(ad.get("warmup_updates", 500)) if ad else 500
        self.ad_lam_max = float(ad.get("lam0_max", 10.0)) if ad else 10.0
        # Dose against the TAIL of warmup (last 20%), not the whole window: with
        # small-Gaussian init the model starts near-linear (curvature ~ 1e-7),
        # and averaging from update 0 explodes the ratio (smoke run measured
        # lam0_auto = 8.5e5 before this fix). Curvature grows as the model fits;
        # the tail is the honest scale. lam0_max caps the residual risk.
        self.ad_tail_start = max(1, int(self.ad_warmup * 0.8))
        self.ad_count, self.ad_fit_sum, self.ad_pen_sum = 0, 0.0, 0.0
        self.lam0_auto = None  # set once at end of warmup (checkpointed)

        # Curvature-certified adaptive horizon (imagination.adaptive_horizon):
        # imagine further only when the penalty EMA has fallen off its peak.
        ah = cfg.imagination.get("adaptive_horizon", None)
        self.ah_enabled = bool(ah and ah.get("enabled", False))
        self.ah_h_min = int(ah.get("h_min", 5)) if ah else 5
        self.ah_h_max = int(ah.get("h_max", 25)) if ah else 25
        self.ah_decay = float(ah.get("decay", 0.99)) if ah else 0.99
        self.pen_ema, self.pen_peak = None, 0.0  # checkpointed

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
        # reward model predicts symlog(r) when model.symlog_reward is on;
        # imagination applies symexp to whatever it consumes (behaviour_update)
        r_target = symlog(r) if self.symlog else r
        rew_loss = F.mse_loss(self.reward(z, a, tau), r_target)

        # Isotropic curvature penalty in joint latent(-task) coords (R4, R16);
        # detached coords: penalize R's surface geometry, not the encoder through it.
        # Including tau in the Hessian coords enforces smooth interpolation
        # BETWEEN tasks — the multi-task generalization lever.
        parts = [z.detach(), a]
        if tau is not None and self.cfg.penalty.get("include_task", True):
            parts.append(tau)
        x_pen = torch.cat(parts, dim=-1)
        # Penalty target = head mean on the model's raw (symlog-space) output —
        # smoothness is enforced in prediction space (see RewardModel.on_concat).
        if tau is not None and not self.cfg.penalty.get("include_task", True):
            fn = lambda x: self.reward.on_concat(
                torch.cat([x, tau.detach()], dim=-1))
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

        # ---- auto-dosed lambda: lam=0 during warmup, then dose lam0 once ----
        if self.ad_enabled and self.ad_count < self.ad_warmup:
            lam_t = 0.0
            self.ad_count += 1
            if self.ad_count > self.ad_tail_start:  # tail window only
                self.ad_fit_sum += dyn_loss.item() + rew_loss.item()
                self.ad_pen_sum += pen.item()
            if self.ad_count == self.ad_warmup:
                n_tail = self.ad_warmup - self.ad_tail_start
                mean_fit = self.ad_fit_sum / n_tail
                mean_pen = self.ad_pen_sum / n_tail
                self.lam0_auto = min(
                    self.ad_target_ratio * mean_fit / max(mean_pen, 1e-12),
                    self.ad_lam_max)
                self.lam.lam0 = self.lam0_auto
        else:
            lam_t = self.lam(self.step)
        loss = dyn_loss + rew_loss + lam_t * pen

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 100.0)
        self.model_opt.step()
        self.ema.update(self.encoder)
        self.step += 1

        # penalty EMA + running peak — the adaptive-horizon certificate signal.
        # NaN hygiene: one non-finite pen.item() must not poison the EMA forever
        # (a poisoned EMA crashed the horizon controller in the shiny run).
        import math as _math
        pen_val = pen.item()
        if _math.isfinite(pen_val):
            self.pen_ema = (pen_val if self.pen_ema is None
                            else self.ah_decay * self.pen_ema + (1 - self.ah_decay) * pen_val)
            self.pen_peak = max(self.pen_peak, self.pen_ema)

        out = {"loss/dyn": dyn_loss.item(), "loss/reward": rew_loss.item(),
               "penalty/value": pen_val, "penalty/lambda": lam_t,
               "loss/total": loss.item(), "step": self.step}
        if self.lam0_auto is not None:
            out["penalty/lam0_auto"] = self.lam0_auto
        return out

    def _imagined_reward(self, z, a, tau=None):
        """Reward as consumed by imagination: per-head symexp (if the model is
        trained in symlog space), then ensemble mean - pessimism * std.
        Returns (reward (B,), mean head disagreement scalar)."""
        heads = self.reward.all_heads(z, a, tau)          # (n_heads, B)
        if self.symlog:
            # Clamp BEFORE symexp: imagined rollouts extrapolate, and
            # expm1(|x| > ~89) overflows float32 -> inf rewards -> NaN policy
            # (the shiny-run crash). +-20 in symlog space is +-4.8e8 raw —
            # far beyond any real reward, harmless to fitting.
            heads = symexp(heads.clamp(-20.0, 20.0))
        if heads.shape[0] == 1:
            return heads[0], heads.new_zeros(())
        std = heads.std(0)
        return heads.mean(0) - self.pessimism * std, std.mean().detach()

    def _imagination_horizon(self) -> int:
        """Curvature-certified horizon: imagine further only as the penalty EMA
        falls off its running peak (low curvature => trust longer rollouts)."""
        if not self.ah_enabled:
            return int(self.cfg.imagination.horizon)
        if self.pen_ema is None or self.pen_peak <= 0:
            return self.ah_h_min  # no curvature evidence yet -> conservative
        frac = min(max(1.0 - self.pen_ema / max(self.pen_peak, 1e-12), 0.0), 1.0)
        if not np.isfinite(frac):  # last-ditch guard: never crash the loop
            return self.ah_h_min
        return int(round(self.ah_h_min + (self.ah_h_max - self.ah_h_min) * frac))

    # ---------------- behaviour learning (Dreamer lambda-returns) ----------------
    def behaviour_update(self, z0: torch.Tensor, tau0: torch.Tensor | None = None) -> dict:
        cfg_i = self.cfg.imagination
        gamma, lam_ret = cfg_i.gamma, cfg_i.get("lambda_", 0.95)
        H = self._imagination_horizon()
        ent_coef = cfg_i.get("entropy_coef", 3e-4)

        # --- differentiable imagination (gradients flow through T and R) ---
        zs, rs, logps, dis = [z0], [], [], []
        z = z0
        for _ in range(H):
            a, logp = self.policy.sample(z, tau0)
            z = self.dynamics(z, a)
            zs.append(z)
            r_im, d = self._imagined_reward(zs[-2], a, tau0)
            rs.append(r_im)
            dis.append(d)
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
            span = float(hi - lo)
            if np.isfinite(span):  # NaN hygiene: don't poison the scale EMA
                self.ret_scale = decay * self.ret_scale + (1 - decay) * span
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
                "model/reward_disagreement": torch.stack(dis).mean().item(),
                "imagine/horizon": H,
                "imagine/return_mean": returns.mean().item(),
                "imagine/return_var": returns.var().item()}  # R15 diagnostic

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, horizon: int, tau0: torch.Tensor | None = None):
        zs, as_, rs = [z0], [], []
        z = z0
        for _ in range(horizon):
            a, _ = self.policy.sample(z, tau0)
            z = self.dynamics(z, a)
            zs.append(z); as_.append(a)
            rs.append(self._imagined_reward(zs[-2], a, tau0)[0])
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
                # auto-dose: computed lam0 + warmup accumulators (bitwise resume)
                "lam0_sched": self.lam.lam0, "lam0_auto": self.lam0_auto,
                "ad_count": self.ad_count, "ad_fit_sum": self.ad_fit_sum,
                "ad_pen_sum": self.ad_pen_sum,
                # adaptive-horizon certificate state
                "pen_ema": self.pen_ema, "pen_peak": self.pen_peak,
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
        if "lam0_sched" in sd:  # auto-dose may have rewritten the schedule's lam0
            self.lam.lam0 = sd["lam0_sched"]
        self.lam0_auto = sd.get("lam0_auto", None)
        self.ad_count = sd.get("ad_count", 0)
        self.ad_fit_sum = sd.get("ad_fit_sum", 0.0)
        self.ad_pen_sum = sd.get("ad_pen_sum", 0.0)
        self.pen_ema = sd.get("pen_ema", None)
        self.pen_peak = sd.get("pen_peak", 0.0)
        if "hutchinson_gen" in sd:  # probe RNG must resume too (bitwise resume)
            self.gen.set_state(sd["hutchinson_gen"])


def collect_vectorized(trainer, env, buffer, obs, autoreset, n_steps: int,
                       device: str = "cpu"):
    """Collect >= n_steps real env steps from a gymnasium vector env.

    Handles gymnasium 1.x NEXT_STEP vector autoreset: when sub-env i reports
    terminated|truncated at step t, the step t+1 "transition" for that sub-env
    is (final_obs, ignored_action, reset_obs) — NOT a real (s, a, s') pair —
    so it is masked out of the buffer. `autoreset` is the per-env boolean mask
    carried across calls. Returns (obs, autoreset, steps_taken).
    """
    num_envs = env.num_envs
    taken = 0
    while taken < n_steps:
        with torch.no_grad():
            z = trainer.encoder(torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                                                device=device))
            a, _ = trainer.policy.sample(z)
        a_np = a.cpu().numpy()
        obs_next, r, term, trunc, _ = env.step(a_np)
        for i in range(num_envs):
            if not autoreset[i]:  # skip the fake post-done boundary transition
                buffer.add(obs[i], a_np[i], float(r[i]), obs_next[i])
        autoreset = np.logical_or(term, trunc)
        obs = obs_next
        taken += num_envs
    return obs, autoreset, taken
