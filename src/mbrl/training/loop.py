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
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..models import (Encoder, EMAEncoder, VAEEncoder, CustomEncoder, AffineDynamics,
                      GaussianAffineDynamics, FullMLPDynamics, OperatorDynamics,
                      RewardModel, Policy, ValueFn, DualLatent)
from ..models.ensemble import EnsembleAffineDynamics
from ..models.planner import SequencePlanner
from ..models.reward import symlog, symexp
from ..models.spectral import SpectralReward, poly_weights, snr_band_weights
from ..regularization.hutchinson import hvp_penalty, laplacian_trace_penalty
from ..regularization.schedule import LambdaSchedule
from ..training.returns import lambda_returns, gae_advantages
from ..training.smoothing import smooth_rewards
from ..utils.seeding import make_generator


class Trainer:
    def __init__(self, cfg, obs_dim: int, action_dim: int, device: str = "cpu",
                 task_dim: int = 0):
        self.cfg, self.device, self.task_dim = cfg, torch.device(device), task_dim
        # latent cap: up to latent_cap_mult x the input dimension (default 4,
        # user rule v2). SPECTRAL runs cap at 1x (user, 2026-06-07): the
        # closed-form reward fit over-resolves wide latents — first RL attempt
        # overfit reward with k > obs_dim; spectral_ladder sets cap_mult 1.
        cap_mult = int(cfg.model.get("latent_cap_mult", 4))
        k = min(cfg.model.latent_dim, cap_mult * obs_dim)
        h, d = cfg.model.hidden, cfg.model.depth
        # encoder: "mlp" (deterministic, default) or "vae" (run 10 —
        # recon + KL grounding; latent near-stationary under the KL pull)
        _enc_kind = str(cfg.model.get("encoder", "mlp"))
        self.enc_vae = _enc_kind == "vae"
        if _enc_kind == "custom":
            # W7 WIRED (2026-06-10): the Studio's NN-brick chain becomes the
            # encoder trunk (net_builder); a projection head pins z to k-dim.
            _net = [dict(l) for l in (cfg.model.get("encoder_net", []) or [])]
            if not _net:
                raise ValueError("model.encoder=custom requires a non-empty "
                                 "model.encoder_net (wire an NN-layer chain)")
            self.encoder = CustomEncoder(obs_dim, k, _net).to(device)
        else:
            enc_cls = VAEEncoder if self.enc_vae else Encoder
            self.encoder = enc_cls(obs_dim, k, h, d).to(device)
        self.vae_beta = float(cfg.model.get("vae", {}).get("beta", 1e-3))
        self.vae_recon_w = float(cfg.model.get("vae", {}).get("recon_weight", 1.0))
        self.ema = EMAEncoder(self.encoder, cfg.model.ema_decay)
        # dynamics: "affine" (deterministic) or "gaussian" (state probability
        # transitions, user 2026-06-08 — mean stays affine in action so the
        # R15 zero-action-curvature property is preserved; variance is
        # state-only). forward() of the gaussian model is an rsample, so
        # imagination becomes a stochastic rollout with gradient flow.
        _dyn_kind = str(cfg.model.get("dynamics", "affine"))
        self.dyn_stochastic = _dyn_kind == "gaussian"
        self.dyn_operator = _dyn_kind == "operator"
        self.op_w = {}   # operator structural-prior weights (set below if operator)
        self.rad_anneal_tau = 0.0   # svband-ceiling anneal off unless set in operator cfg
        self.struct_every = 1       # phased SVD: structural priors every Nth update (1 = every)
        self._op_metrics_cache = {} # last computed operator diagnostics (logged on skipped steps)
        dyn_cls = {"affine": AffineDynamics, "gaussian": GaussianAffineDynamics,
                   "mlp": FullMLPDynamics, "operator": OperatorDynamics}[_dyn_kind]
        if _dyn_kind == "mlp":
            import warnings as _w2
            _w2.warn("[dynamics] mlp is the run-9 R15-ablation arm: action "
                     "curvature is deliberately unconstrained. Never a default.")
        # algo.dynamics_ensemble >= 2: an R15-SAFE PETS-style ensemble of affine
        # members replaces the single dynamics (WIRED arm, 2026-06-10). Members
        # are deterministic-affine, so it composes only with dynamics=affine.
        _n_ens = int((cfg.get("algo", {}) or {}).get("dynamics_ensemble", 0) or 0)
        self.dyn_ensemble = _n_ens >= 2
        if self.dyn_ensemble and _dyn_kind != "affine":
            raise ValueError(
                "algo.dynamics_ensemble requires model.dynamics=affine (members are "
                f"deterministic affine maps, R15-safe); got dynamics='{_dyn_kind}'")
        if self.dyn_ensemble:
            self.dynamics = EnsembleAffineDynamics(k, action_dim, _n_ens, h, d).to(device)
        elif self.dyn_operator:
            # operator-field dynamics z'=A(z)z+B(z)a (R15-safe; A=I+Â near-I init).
            # Structural priors default OFF (w_*=0) -> pure operator map; turn them
            # on to keep A a coherent bundle (normal/smooth/spread/radius).
            _op = dict(cfg.model.get("operator", {}) or {})
            self.dynamics = OperatorDynamics(
                k, action_dim, h, d,
                structure=str(_op.get("structure", "none")),
                rank=int(_op.get("rank", 0)),
                radius_min=float(_op.get("radius_min", 0.0)),
                radius_max=float(_op.get("radius_max", 1.0)),
                init_shift=float(_op.get("init_shift", 1.0))).to(device)
            self.op_w = {kk: float(_op.get(f"w_{kk}", 0.0))
                         for kk in ("normal", "smooth", "spread", "radius", "svband")}
            # annealed svband ceiling: radius_max decays radius_anneal_start → radius_max
            # (the floor) over self.step (op_d only; applied each model_update).
            self.rad_anneal_tau = float(_op.get("radius_anneal_tau", 0.0) or 0.0)
            self.rad_anneal_start = float(_op.get("radius_anneal_start", 1.0))
            self.rad_anneal_floor = float(_op.get("radius_max", 1.0))
            self.struct_every = max(1, int(_op.get("struct_every", 1)))
            if self.op_w["svband"] > 0.0 and float(_op.get("radius_max", 1.0)) >= 1.0:
                import warnings as _w3
                _w3.warn("[operator] w_svband>0 with radius_max>=1: the free band is "
                         "not entirely below σ=1, so the anti-freeze gap is not one-way "
                         "(modes can sit at/above marginal). Set operator.radius_max<1.")
        else:
            self.dynamics = dyn_cls(k, action_dim, h, d).to(device)
        # epistemic discount on imagined reward: r -= coef * ensemble disagreement
        self.ens_pessimism = float((cfg.get("algo", {}) or {}).get("ensemble_pessimism", 0.0) or 0.0)
        self.symlog = bool(cfg.model.get("symlog_reward", False))
        # DUAL-LATENT controlled-operator model (model.dual_latent.enabled): shared
        # encoder z splits into a dynamics latent d=D(z) and a policy latent p=P(z);
        # reward/policy/value read p. A SEPARATE, gated path (_model_update_dual /
        # _behaviour_update_dual) — when off, the validated z-based loop is byte-
        # for-byte unchanged. Requires the operator dynamics (it owns its own
        # operators); incompatible with spectral/ensemble (a clean fresh arm).
        _dl = dict(cfg.model.get("dual_latent", {}) or {})
        self.dual_latent = bool(_dl.get("enabled", False))
        self.dual = None
        self.energy = None          # cf5 lyapunov energy head (built below if enabled)
        self.frame_enabled = False  # cf5 rank-2 reward⊥energy frame (default off)
        self.frame_balance = False  # cf7 equilibrium coupling of alignment vs energy
        self._bal_ema_align = None  # running |couple| (checkpointed)
        self._bal_ema_energy = None # running |dissip| (checkpointed)
        if self.dual_latent:
            if not self.dyn_operator:
                raise ValueError("model.dual_latent requires model.dynamics=operator")
            if bool((cfg.get("spectral", {}) or {}).get("enabled", False)) or self.dyn_ensemble:
                raise ValueError("model.dual_latent is incompatible with spectral/ensemble")
            self.dual = DualLatent(
                k, action_dim, h, d, mode=str(_dl.get("mode", "shared")),
                d_dim=int(_dl.get("d_dim", 0)), p_dim=int(_dl.get("p_dim", 0)),
                op_structure=str(cfg.model.get("operator", {}).get("structure", "none")),
                op_rank=int(cfg.model.get("operator", {}).get("rank", 0)),
                op_radius_min=float(cfg.model.get("operator", {}).get("radius_min", 0.0)),
                op_radius_max=float(cfg.model.get("operator", {}).get("radius_max", 1.0)),
                op_d_init_shift=float(cfg.model.get("operator", {}).get("init_shift", 1.0)),
                couple_dim=int(_dl.get("couple_dim", 0))).to(device)
            self.couple_w = float(_dl.get("couple_weight", 0.0))
            self.pconsist_w = float(_dl.get("p_consistency_weight", 1.0))
            # Lyapunov/Stein consistency on op_d (twin): force the empirical d
            # second moment to be op_d's stationary covariance, G = A G Aᵀ + Q̂
            # (term (c), docs/unified_spectral_loss.md). 0 = off (default).
            self.lyap_w = float(_dl.get("lyap_weight", 0.0))
            # det(op_p) > 0 (twin): require the POLICY operator to be invertible AND
            # orientation-preserving — a soft barrier relu(floor − det A_p)². Keeps op_p
            # in GL⁺ (no policy mode collapses, no orientation flip) so its entropy
            # exponent log det A_p stays finite. The conservative-op_p counterpart to
            # svband's dissipative-op_d. 0 = off (default).
            self.detpos_w = float(_dl.get("detpos_weight", 0.0))
            self.detpos_floor = float(_dl.get("detpos_floor", 0.05))
            # reward-curvature penalty on p: optional (twin wants p ROUGH — see
            # _model_update_dual). Per-operator structural weights let the dynamics
            # operator op_d be regularized smooth while the policy operator op_p is
            # left rough: operator_p overrides fall back to model.operator.w_* (op_d).
            self.dual_penalize_reward = bool(_dl.get("penalize_reward", True))
            # smooth_p: regularize op_p like op_d (conjoined) or leave p rough (separate)
            self.op_w_p = (dict(self.op_w) if bool(_dl.get("smooth_p", True))
                           else {kk: 0.0 for kk in self.op_w})
            # cf4: even with p left ROUGH (smooth_p=false ⇒ no normal/smooth/spread
            # priors), optionally BOUND op_p's spectral radius so imagined p-rollouts
            # can't diverge (the NaN root cause). radius_p>0 reinstates ONLY the radius
            # prior on op_p (relu(σ_max−1)²): roughness (sharp reward/value structure) is
            # preserved, expansiveness (σ_max>1) is not. 0 = off (the cf3 behaviour).
            _radius_p = float(_dl.get("radius_p", 0.0) or 0.0)
            if _radius_p > 0.0:
                self.op_w_p = {**self.op_w_p, "radius": _radius_p}
            # cf5 rank-2 reward⊥energy frame (model.dual_latent.rank2_frame, PM
            # 2026-06-14). Default off ⇒ no-op. The (lyapunov) energy head is built
            # HERE so model_opt trains it (params appended to _model_params below).
            # See regularization/rank2_frame.py for the term math.
            _rf = dict(_dl.get("rank2_frame", {}) or {})
            self.frame_enabled = bool(_rf.get("enabled", False))
            self.frame_energy_mode = str(_rf.get("energy_mode", "lyapunov"))
            self.frame_energy_anchor = float(_rf.get("energy_anchor", 0.0) or 0.0)   # anti-collapse
            self.frame_w_ortho = float(_rf.get("w_ortho", 0.0) or 0.0)
            self.frame_w_rank2 = float(_rf.get("w_rank2", 0.0) or 0.0)
            self.frame_w_lyap = float(_rf.get("w_lyap", 0.0) or 0.0)
            self.frame_w_dissip = float(_rf.get("w_dissip", 0.0) or 0.0)   # cf6 dissipativity
            self.frame_supply = str(_rf.get("supply", "reward"))
            # cf7 (PM 2026-06-14): equilibrium coupling of the alignment (couple) and
            # energy (dissipativity) penalties. ON ⇒ each is normalized by its running
            # magnitude before summing, so neither outweighs the other (the fixed
            # couple_weight / w_dissip are then ignored). The equilibrium point is
            # balanced gradient influence. Needs the dissipativity (lyapunov + w_dissip).
            self.frame_balance = bool(_rf.get("balance", False))
            self.frame_balance_w = float(_rf.get("balance_weight", 0.1) or 0.1)
            self.frame_balance_decay = float(_rf.get("balance_decay", 0.99))
            self.frame_subsample = int(_rf.get("subsample", 64))
            self.frame_target_rank = int(_rf.get("target_rank", 2))
            self.frame_w_shell = float(_rf.get("w_shell", 0.0) or 0.0)   # cf10 two-sided shell
            self.frame_shell_target = float(_rf.get("shell_target", 1.0))
            self.frame_shell_floor = float(_rf.get("shell_floor", 0.0))  # tail floor (cond bound)
            self.frame_w_logdet = float(_rf.get("w_logdet", 0.0) or 0.0)  # cf12 KL/log-det cond barrier
            self.frame_logdet_eps = float(_rf.get("logdet_eps", 1e-2))
            self.frame_w_band = float(_rf.get("w_band", 0.0) or 0.0)       # cf14 two-sided band
            self.frame_band_ceiling = float(_rf.get("band_ceiling", 1.0))
            self.frame_band_floor = float(_rf.get("band_floor", 0.1))
            self.frame_band_floor_shape = str(_rf.get("band_floor_shape", "relu2"))  # cf18 floor wall
            self.frame_band_floor_beta = float(_rf.get("band_floor_beta", 20.0))
            self.frame_w_compress = float(_rf.get("w_compress", 0.0) or 0.0)  # cf15 nuclear-norm compression
            if self.frame_enabled and self.frame_energy_mode == "contractive" \
                    and self.dual.mode != "twin":
                raise ValueError("rank2_frame.energy_mode=contractive needs "
                                 "dual_latent.mode=twin (it reads op_d's spectrum)")
            if self.frame_enabled and self.frame_w_dissip > 0.0 \
                    and self.frame_supply != "reward":
                import warnings as _wf
                _wf.warn(f"rank2_frame.supply={self.frame_supply} not yet wired (only "
                         "'reward' is — advantage/return need energy in the imagined "
                         "rollout's p-space); using per-step reward as the supply.")
            if self.frame_enabled and self.frame_energy_mode == "lyapunov":
                from ..regularization.rank2_frame import EnergyHead
                self.energy = EnergyHead(self.dual.d_dim, cfg.model.hidden,
                                         cfg.model.depth,
                                         anchor=self.frame_energy_anchor).to(device)
        # heads read the POLICY latent p in dual mode (dim p_dim), else the backbone z
        rk = self.dual.p_dim if self.dual_latent else k
        self.reward = RewardModel(rk, action_dim, h, d, task_dim=task_dim,
                                  n_heads=int(cfg.model.get("reward_heads", 1))).to(device)
        self.pessimism = float(cfg.imagination.get("pessimism", 0.0))
        # Data-driven symexp clamp (replaces the fixed +-20): running max of
        # |symlog(r_target)| over real batches; imagination clamps head outputs
        # to +-(symexp_margin * symlog_bound) before symexp. The fixed +-20
        # allowed symexp up to 4.8e8 — imagined-return variance reached 1e19
        # concentrated in low-lambda windows (38/100 iterations on the last
        # grid). Init 1.0 = conservative floor before any real data is seen.
        self.symlog_bound = 1.0  # checkpointed (bitwise resume)
        self.symexp_margin = float(cfg.imagination.get("symexp_margin", 1.5))
        self.policy = Policy(rk, action_dim, h, d, task_dim=task_dim,
                             init_scale=float(cfg.model.get("policy_init_scale", 1.0))).to(device)
        self.value = ValueFn(rk, h, d, task_dim=task_dim).to(device)
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)
        # A4 clipped double-value (PM 2026-06-15; TD3 twin-value min on the imagined λ-return
        # bootstrap; off by default). Second value net + EMA target; v_tgt = min(V1,V2) kills
        # value over-estimation. In-framework: still value-on-imagined-rollouts, no model-free Q.
        self.double_value = bool(cfg.optim.get("clipped_double_value", False))
        if self.double_value:
            self.value2 = ValueFn(rk, h, d, task_dim=task_dim).to(device)
            self.value2_target = copy.deepcopy(self.value2).requires_grad_(False)
            self.value2_opt = torch.optim.AdamW(self.value2.parameters(), lr=cfg.optim.value_lr)
        # A3 auto-tuned entropy temperature (PM 2026-06-15; SAC dual; off by default).
        # α=exp(log_alpha) tuned so policy entropy tracks target_entropy — replaces the
        # static/reward-annealed entropy coef. In-framework: just adapts the existing bonus.
        _aa = dict(cfg.optim.get("auto_alpha", {}) or {})
        self.auto_alpha = bool(_aa.get("enabled", False))
        if self.auto_alpha:
            self.alpha_target_H = float(_aa.get("target_entropy", 1.0))
            self.log_alpha = torch.tensor(float(np.log(float(_aa.get("init", 3e-4)))),
                                          device=device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(_aa.get("lr", 1e-3)))

        # Transformer action-sequence planner (planner.enabled): REPLACES the
        # per-step MLP policy as the actor. It emits an H-step plan from z0; the
        # affine T still predicts the latents (R15); the same imagination +
        # λ-return objective trains it (R10: never curvature-penalized). The
        # horizon is FIXED to imagination.horizon (the plan length) — adaptive
        # horizon is incompatible and overridden when the planner is on.
        pl = cfg.get("planner", None)
        self.use_planner = bool(pl and pl.get("enabled", False))
        self.planner = None
        if self.use_planner:
            self.planner = SequencePlanner(
                k, action_dim, horizon=int(cfg.imagination.horizon),
                d_model=int(pl.get("d_model", 128)), nhead=int(pl.get("nhead", 4)),
                layers=int(pl.get("layers", 2)), task_dim=task_dim).to(device)

        # dual-latent owns its own operator(s)+projections (model_opt trains them);
        # the unused self.dynamics gets no gradient and is simply never stepped.
        _model_params = [*self.encoder.parameters(), *self.reward.parameters()]
        _model_params += (list(self.dual.parameters()) if self.dual_latent
                          else list(self.dynamics.parameters()))
        if self.energy is not None:          # cf5 lyapunov energy head trains with the model
            _model_params += list(self.energy.parameters())
        self.model_opt = torch.optim.AdamW(_model_params, lr=cfg.optim.model_lr)
        actor_params = (self.planner if self.use_planner else self.policy).parameters()
        self.policy_opt = torch.optim.AdamW(actor_params, lr=cfg.optim.policy_lr)
        self.value_opt = torch.optim.AdamW(self.value.parameters(), lr=cfg.optim.value_lr)
        # LR tied to lambda (PM 2026-06-15): decay the MODEL learning rate with the SAME
        # exponent as the curvature lambda(t) (cuberoot/R12), so regularization strength and
        # model step size anneal on a matched timescale. Same t0 as the penalty schedule ⇒
        # lr(t) ∝ lambda(t). Default kind=constant ⇒ legacy fixed lr (byte-identical).
        _lrs = dict(cfg.optim.get("lr_schedule", {}) or {})
        self._model_lr0 = float(cfg.optim.model_lr)
        self._model_lr_now = self._model_lr0
        self.lr_sched = None
        if str(_lrs.get("kind", "constant")) != "constant":
            self.lr_sched = LambdaSchedule(
                kind=str(_lrs.get("kind", "cuberoot")), lam0=self._model_lr0,
                t0=float(_lrs.get("t0", cfg.penalty.schedule.get("t0", 10000.0))),
                floor=float(_lrs.get("floor", 0.0)))
        # imagination-latent alignment stabilizer (2507.16450) + configurable
        # actor grad-clip — the transformer-stabilization study's levers.
        self.align_weight = float(cfg.imagination.get("align_weight", 0.0))
        self.actor_clip = float(cfg.optim.get("actor_clip", 100.0))
        # reward-adaptive policy regularization (PM 2026-06-15): one return-fraction
        # rf=clip((ret_ema-mid)/scale,0,1) [0=low return→explore, 1=high→exploit] drives
        # three composable knobs — anneal the entropy bonus, floor the entropy by reward,
        # and tighten the actor grad-clip as return rises (the policy analog of the λ/horizon
        # ratchets). Pairs with policy_init_scale (consistent near-zero start). All default off.
        ra = dict(cfg.get("reward_adapt", {}) or {})
        self.ra_mid = float(ra.get("mid", 0.0)); self.ra_scale = float(ra.get("scale", 500.0))
        self.ra_anneal = bool(ra.get("entropy_anneal", False))
        _ef = dict(ra.get("entropy_floor", {}) or {})
        self.ra_floor_on = bool(_ef.get("enabled", False))
        self.ra_floor_h = float(_ef.get("h_high", 1.0)); self.ra_floor_coef = float(_ef.get("coef", 0.01))
        # floor-shape (PM 2026-06-15): relu = constant lift (reverses a deep collapse but
        # weak), sigmoid = lift peaks AT the target then vanishes below (a sharp catch-it-
        # as-it-crosses barrier — prevents the collapse from starting; pair with sharp beta).
        self.ra_floor_shape = str(_ef.get("shape", "relu")).lower()
        self.ra_floor_beta = float(_ef.get("beta", 4.0))
        _ac = dict(ra.get("actor_clip_adapt", {}) or {})
        self.ra_clip_on = bool(_ac.get("enabled", False)); self.ra_clip_min = float(_ac.get("min_frac", 0.1))
        # reward-adaptive HARD log_std floor (cf21, PM 2026-06-15): drive the policy's
        # variance bound by rf — floor HIGH (explore) at low return, relaxing toward `lo`
        # (commit) as return climbs. The hard structural fix for the seed-spread collapse;
        # replaces the soft entropy floor. eval samples stochastically, so the relaxation
        # is what recovers near-deterministic peak return.
        _lf = dict(ra.get("logstd_floor", {}) or {})
        self.ra_lsf_on = bool(_lf.get("enabled", False))
        self.ra_lsf_hi = float(_lf.get("hi", -1.0))   # log_std floor at rf=0 (σ ≥ e^hi, explore)
        self.ra_lsf_lo = float(_lf.get("lo", -5.0))   # log_std floor at rf=1 (commit; -5 = legacy)
        self._reward_adapt_on = self.ra_anneal or self.ra_floor_on or self.ra_clip_on or self.ra_lsf_on
        # NaN-stabilization levers (cf4, PM 2026-06-14). Diagnosed on cf3-i0-s2 @100k:
        # the twin's UNREGULARIZED policy operator op_p (radius_p≈1.06>1) makes imagined
        # p-rollouts grow geometrically; right as the policy crosses return≈0 a rollout
        # diverges → imagined returns → inf → actor grad → 46k → NaN. loss/total stayed
        # ~1e-3 the whole time, so normalizing IT is the wrong target — the blowup is the
        # actor gradient on diverging returns. Defence in depth (all default-off ⇒ legacy
        # byte-exact): reward_clip/return_clip cap the imagined reward + λ-returns BEFORE
        # they reach the policy/value loss; value_clip grad-clips the value optimizer (was
        # unclipped); skip_nonfinite skips the opt step when the grad norm is non-finite so
        # one bad rollout can't poison the weights (the run recovers). The root cause is
        # bounded separately by model.dual_latent.radius_p (op_p's spectral radius).
        self.reward_clip = float(cfg.imagination.get("reward_clip", 0.0) or 0.0)
        self.return_clip = float(cfg.imagination.get("return_clip", 0.0) or 0.0)
        self.value_clip = float(cfg.optim.get("value_clip", 0.0) or 0.0)
        self.skip_nonfinite = bool(cfg.optim.get("skip_nonfinite", False))
        self._nonfinite_skips = 0   # diagnostic counter (checkpointed for bitwise resume)
        # pure-diagnostic scalars (no behaviour/RNG impact): closed-form spectral
        # fit quality (set in _spectral_refit) and per-rollout reward-clip
        # accumulators (set in _imagined_reward, reset per behaviour_update).
        self._last_fit_mse = 0.0
        self._last_fit_r2 = 0.0
        self._reward_clip_over = 0.0   # # imagined rewards over reward_clip this rollout
        self._reward_clip_tot = 0.0    # # imagined rewards seen this rollout
        self._last_return_clip_frac = 0.0   # frac λ-returns over return_clip (last step)
        self._last_value_grad_norm = 0.0    # pre-clip value-opt grad norm (last step)
        # Policy INERTIA (PM 2026-06-13): give the policy extra inertia relative to
        # the (faster) operator — a two-timescale stabilizer against the collapse
        # (the policy lunging at transient model errors). A slow EMA of the policy
        # weights (mirrors value_target), used two ways: act/collect with the EMA
        # (policy_ema_act = behaviour inertia) and/or anchor the live policy to it
        # (policy_inertia*||θ-θ_ema||² = weight inertia / soft trust region). The
        # policy is already 3x slower than the operator (1e-4 vs 3e-4); this widens
        # it. Only for the MLP-policy actor (not the planner). Default OFF.
        self.policy_ema_decay = float(cfg.optim.get("policy_ema_decay", 0.0))
        self.policy_ema_act = bool(cfg.optim.get("policy_ema_act", False))
        self.policy_inertia = float(cfg.optim.get("policy_inertia", 0.0))
        self.policy_ema = None
        if self.policy_ema_decay > 0.0 and not self.use_planner:
            self.policy_ema = copy.deepcopy(self.policy).requires_grad_(False)

        self.lam = LambdaSchedule(**cfg.penalty.schedule)
        # HARD floor on the EFFECTIVE penalty lambda (PM 2026-06-15): after the schedule
        # and BOTH gates multiply, clamp lam_t >= lambda_min so the curvature regularizer
        # never fully releases (a gate -> ~0 lets the model sharpen without bound). 0 = off.
        # Logged as penalty/lambda (the clamped value), so it stays correlation-trackable.
        self.lambda_min = float(cfg.penalty.get("lambda_min", 0.0) or 0.0)
        self.step = 0
        self.gen = make_generator(self.device, cfg.seed)
        # ---- Studio viz tensor/salience snapshots (DEFAULT-OFF, byte-exact) ----
        # Two named-tensor surfaces for the Studio, gated EXACTLY like the reward-
        # surface snapshot (viz.surface_every in scripts/train.py): write nothing on
        # default runs. viz.tensor_every>0  => every Nth model update, snapshot the
        # latent Gram G (and, in twin mode, op_d/op_p) to tensors/gram_<step>.json.
        # viz.salience_every>0 => snapshot |∂reward/∂obs| to
        # tensors/reward_input_salience_<step>.json from a SELF-CONTAINED autograd
        # pass (its own detached obs leaf; no training tensors/opt/RNG touched, NO
        # self.gen draws). Both keyed on self.step (the model-update index). 0 = OFF.
        _viz = dict(cfg.get("viz", {}) or {})
        self.tensor_every = int(_viz.get("tensor_every", 0) or 0)
        self.salience_every = int(_viz.get("salience_every", 0) or 0)
        self._viz_tensors_on = self.tensor_every > 0 or self.salience_every > 0
        # run name + results ROOT for the artifact path — built the SAME way the
        # surface snapshot does (scripts/train.py): <logging.dir>/runs/<run>/tensors/,
        # matching the SurfaceIndex/TensorIndex run-path convention. Only consulted
        # when a cadence is on (default runs never touch the filesystem here).
        if self._viz_tensors_on:
            self._viz_run = f"{cfg.experiment.name}-{cfg.env.name}-s{cfg.seed}"
            self._viz_root = str(cfg.logging.dir)
        # Gated stochastic excitation (model.operator.excite_*): a discrete Bernoulli(excite_p)
        # gate per behaviour update, OPEN only when the latent sits at its attractor (ema z_std
        # within excite_zstd_band of excite_zstd_anchor). When open, inject process noise
        # ε~N(0,(excite_scale·innov)²) at EVERY imagined-rollout step — a parametric drive of the
        # marginal oscillator pinned at |λ|≈1, to phase-kick it toward the firing basin. innov =
        # EMA of the Stein-innovation RMS √(tr Q̂/k); z_std_ema = EMA of latent z_std. Inert unless
        # excite_enabled (every existing arm + checkpoint byte-identical). All draws use self.gen
        # (checkpointed) for bitwise resume. PM 2026-06-16 (discrete-gate Q-drive of the edge cycle).
        _opx = dict(cfg.model.get("operator", {}) or {})
        self.excite_enabled = bool(_opx.get("excite_enabled", False))
        self.excite_p = float(_opx.get("excite_p", 0.2))
        self.excite_zstd_anchor = float(_opx.get("excite_zstd_anchor", 0.8))
        self.excite_zstd_band = float(_opx.get("excite_zstd_band", 0.1))
        self.excite_scale = float(_opx.get("excite_scale", 1.0))
        self.excite_decay = float(_opx.get("excite_ema_decay", 0.99))
        self.z_std_ema = None      # checkpointed; EMA of latent z_std (the excite gate anchor)
        self.innov_ema = None      # checkpointed; EMA of Stein innovation RMS (the excite scale)
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
        self.ah_ratchet = bool(ah.get("ratchet", False)) if ah else False  # cf17 monotonic floor
        self.ah_ratchet_base = int(ah.get("ratchet_base", 15)) if ah else 15
        self.pen_ema, self.pen_peak = None, 0.0  # checkpointed
        self._ah_ratchet_floor = 0      # running-max H once engaged (checkpointed)
        self._ah_ratchet_on = False     # has H reached ratchet_base yet (checkpointed)

        # Disagreement-gated lambda (penalty.disagreement_gate): lam(t) scaled
        # by the reward ensemble's convergence — full while heads disagree
        # (uncertain, early), released toward `floor` as they agree (R12 anneal,
        # model-paced not clock-paced). Gate in [floor, 1]; reuses the EMA/peak
        # pattern; dis_ema/peak are checkpointed (no new RNG -> bitwise resume).
        # cfg-only reads here so it is order-independent of the spectral block.
        dg = cfg.penalty.get("disagreement_gate", None)
        self.dg_enabled = bool(dg and dg.get("enabled", False))
        self.dg_floor = float(dg.get("floor", 0.1)) if dg else 0.1
        self.dg_decay = float(dg.get("decay", 0.99)) if dg else 0.99
        self.dis_ema, self.dis_peak = None, 0.0  # checkpointed
        self.dg_gate_now = 1.0   # current gate factor (1.0 = disabled / full lam)
        self._spec_head_dis = None  # spectral ensemble head-spread (checkpointed)
        if self.dg_enabled:
            _nh = int(cfg.model.get("reward_heads", 1))
            _sp = cfg.get("spectral", None)
            _aux_on = bool(_sp.get("encoder_aux", True)) if _sp else True
            _spec_on = bool(_sp and _sp.get("enabled", False))
            if _nh < 2 or (_spec_on and not _aux_on):
                import warnings as _wdg
                _wdg.warn("[penalty] disagreement_gate needs reward_heads>=2 and "
                          "a trained reward model (non-spectral or spectral+"
                          "encoder_aux); disabling — lambda follows the schedule.")
                self.dg_enabled = False

        # Return-gated lambda (penalty.return_gate, PM 2026-06-13): a WEAK,
        # never-zero multiplier on lam(t) keyed on ACTUAL eval return. ABSOLUTE,
        # SIGN-AWARE, SMOOTH (rev. 2026-06-13): gate = floor + (1-floor)*(1-σ((R̄
        # - mid)/scale)). Genuinely-positive return (R̄ ≫ mid) -> gate -> floor
        # (relax lam); negative return (R̄ ≪ mid) -> gate -> 1 (hold lam high);
        # R̄ ≈ mid -> mid of [floor,1] (NO spike near zero). `mid` (default 0 = the
        # do-nothing boundary on HalfCheetah) makes it sign-aware; `slew` caps the
        # per-eval change so a collapse can't spike lam. No running min/max (so no
        # degenerate span, no outlier ratchet). Fed by observe_return() from eval;
        # ret_ema + rg_gate_now checkpointed (no new RNG -> bitwise resume).
        rg = cfg.penalty.get("return_gate", None)
        self.rg_enabled = bool(rg and rg.get("enabled", False))
        self.rg_floor = float(rg.get("floor", 0.1)) if rg else 0.1   # 0.5 was too high (PM)
        self.rg_decay = float(rg.get("decay", 0.95)) if rg else 0.95
        self.rg_mid = float(rg.get("mid", 0.0)) if rg else 0.0       # return midpoint (sign anchor)
        self.rg_scale = float(rg.get("scale", 100.0)) if rg else 100.0  # transition width
        self.rg_slew = float(rg.get("slew", 0.1)) if rg else 0.1     # max gate change per eval
        self.rg_shape = str(rg.get("shape", "quadratic")) if rg else "quadratic"  # quad|cuberoot|sigmoid|bump|leaky_relu
        self.rg_leak = float(rg.get("leak", 0.1)) if rg else 0.1     # leaky_relu gate: pre-knee slope
        self.rg_ratchet = bool(rg.get("ratchet", False)) if rg else False  # cf19 monotonic gate
        self.ret_ema = None      # checkpointed
        self._apply_logstd_floor()   # cf21: set the initial (rf=0, explore) variance bound
        self.rg_gate_now = 1.0   # current gate factor (1.0 = disabled / full lam)
        self._rg_ratchet_on = False    # engaged once return crosses mid (checkpointed)
        self._rg_ratchet_min = 1.0     # running-min gate once engaged (checkpointed)

        # Spectral reward path (spectral.enabled): the reward is an ensemble of
        # closed-form RFF ridge heads over the SAME coords as the penalty
        # (z.detach(), a[, tau]); refit every refit_every model updates from a
        # rolling (x, symlog-target) cache with POLYNOMIAL per-band penalty
        # weights — theta_d(t) = coefs[d] * lam(t + shifts[d]) per degree, so
        # frequency bands clamp/release at different phases of the schedule.
        # The H^2 penalty is EXACT in this basis: no Hutchinson on the reward,
        # no reward fit loss for the MLP (the MLP reward model is bypassed).
        sp = cfg.get("spectral", None)
        self.spec_enabled = bool(sp and sp.get("enabled", False))
        if self.spec_enabled:
            self.spec_refit_every = int(sp.get("refit_every", 200))
            self.spec_cache_size = int(sp.get("cache_size", 4096))
            poly = sp.get("poly", None)
            self.spec_degrees = [int(d) for d in poly.get("degrees", [2])] if poly else [2]
            self.spec_coefs = [float(c) for c in poly.get("coefs", [1.0])] if poly else [1.0]
            shifts = (poly.get("shifts", None) if poly else None) or [0] * len(self.spec_degrees)
            self.spec_shifts = [int(s) for s in shifts]
            if not (len(self.spec_degrees) == len(self.spec_coefs) == len(self.spec_shifts)):
                raise ValueError("spectral.poly degrees/coefs/shifts must have equal length")
            spec_in = k + action_dim + task_dim
            self.spec_nf = int(sp.get("n_features", 512))
            self.spec_nheads = int(sp.get("heads", 3))
            # 2-adic head: solve each head's ill-conditioned (M,M) ridge by exact p-adic Dixon
            # lifting (utils.exact_solve) instead of torch.linalg.solve. Off by default. Slow
            # (pure-Python bignum) at M=n_features ⇒ a certification path, not for fast runs.
            self.spec_exact_solve = bool(sp.get("exact_solve", False))
            sw = sp.get("sigma_w", 1.0)
            if isinstance(sw, str) and sw == "learned":
                # LEARNED scales (user, 2026-06-08): no manual clamp — init at
                # the ladder, then per-block log-scales train by gradient on
                # the reward fit error through the cos features ("gradients
                # flow through the scaled pipes"). c re-anchors each refit.
                init = [float(s) for s in sp.get("init_ladder",
                                                 [0.25, 0.5, 1.0, 2.0])]
                self.spec_sigma = "learned"
                self.spec_sigma_star = None
                self.spec_heads = self._build_spec_heads(init, learn=True)
                self.spec_sigma_opt = torch.optim.Adam(
                    [h.log_s for h in self.spec_heads],
                    lr=float(sp.get("sigma_lr", 1e-3)))
            elif isinstance(sw, str) and sw == "auto":
                # SNR-CALIBRATED ladder (bridge run 5): heads are built lazily
                # at the first refit — calibrate_sigma_ladder measures the
                # SNR=1 crossing sigma* on the cache and places rungs at
                # sigma* x cal_mults. cal_low placement won the supervised
                # head-to-head (+48.3% vs +33.7% for the hand ladder).
                self.spec_sigma = "auto"
                self.spec_cal_mults = tuple(
                    float(m) for m in sp.get("cal_mults", [0.5, 1.0, 2.0, 4.0]))
                self.spec_sigma_star = None   # logged after calibration
                self.spec_heads = []
            else:
                try:                       # scalar, or a list = sigma LADDER
                    sw = float(sw)         # (multi-scale frame; bridge run 3)
                except (TypeError, ValueError):
                    sw = [float(s) for s in sw]
                self.spec_sigma = sw
                self.spec_sigma_star = None
                self.spec_heads = self._build_spec_heads(sw)
            self.spec_cache_x = torch.zeros(0, spec_in)   # rolling FIFO (CPU)
            self.spec_cache_y = torch.zeros(0)
            self.spec_since_refit = 0   # model updates since last refit
            self.spec_refits = 0        # 0 => heads still predict zeros (logged)
            # encoder-grounding auxiliary (see model_update): ON by default —
            # without it the encoder collapses to constant z on MuJoCo
            self.spec_aux = bool(sp.get("encoder_aux", True))
            self.spec_aux_weight = float(sp.get("encoder_aux_weight", 1.0))
            # weights_mode: "poly" (schedule-driven lambda polynomial) or "snr"
            # (explicit Wiener weights from split-half band SNR — Tier-1
            # Wiener identity made load-bearing; cutoff at SNR=1, no hand shape,
            # schedule NOT applied). EMA across refits stabilizes the estimate.
            self.spec_weights_mode = str(sp.get("weights_mode", "poly"))
            self.spec_snr_bands = int(sp.get("snr_bands", 8))
            self.spec_snr_ema_decay = float(sp.get("snr_ema", 0.9))
            self.spec_snr_ema = [None] * self.spec_nheads  # per-head theta
            self.spec_snr_info = None   # last head-0 band diagnostics
            self.spec_snr_gen = torch.Generator().manual_seed(int(cfg.seed) + 777)
            # recalibration-on-drift (improvement plan #2): sigma* is measured
            # on early-policy data; re-probe every recal_every refits and
            # rebuild the ladder if it moved more than recal_drift x.
            self.spec_auto = (self.spec_sigma == "auto")
            self.spec_recal_every = int(sp.get("recal_every", 0))   # 0 = off
            self.spec_recal_drift = float(sp.get("recal_drift", 2.0))
            # learned-sigma elastic anchor (improvement plan #3): L2 on log_s
            # toward init bounds the drift toward overfit-friendly bandwidths
            self.spec_sigma_wd = float(sp.get("sigma_wd", 1e-3))
            # config validator (improvement plan #11): warn loudly on known-bad
            # combinations; never error — ablations must stay possible
            import warnings as _w
            _kind = str(cfg.penalty.schedule.get("kind", ""))
            _floor = float(cfg.penalty.schedule.get("floor", 0.0) or 0.0)
            if _kind in ("step", "sincos", "sin2chirp") or _floor <= 0.0:
                _w.warn(f"[spectral] schedule kind={_kind} floor={_floor}: "
                        "zero-touching schedules make the next closed-form "
                        "refit an unregularized interpolator (ledger, "
                        "2026-06-07). Use cuberoot with floor > 0 unless this "
                        "is a deliberate ablation arm.")
            if int(cfg.model.get("latent_cap_mult", 4)) > 1:
                _w.warn("[spectral] latent_cap_mult > 1 with the spectral "
                        "path: wide latents overfit the closed-form reward "
                        "fit (ledger, 2026-06-07). Set model.latent_cap_mult=1 "
                        "unless deliberately ablating.")

    # ---------------- spectral reward (closed-form RFF ridge ensemble) ----------------
    def _build_spec_heads(self, sigma_w, learn: bool = False):
        spec_in = self.spec_cache_x.shape[1] if hasattr(self, "spec_cache_x") \
            and self.spec_cache_x.ndim == 2 else None
        # at __init__ time the cache doesn't exist yet; derive from encoder
        if spec_in is None:
            # dynamics.m = action_dim, uniform across ALL dynamics classes —
            # EnsembleAffineDynamics gained k/m for exactly this consumer
            # (caught by test_experiment_configs composing spectral + ensemble)
            spec_in = self.encoder.latent_dim + self.dynamics.m + self.task_dim
        return [SpectralReward(spec_in, n_features=self.spec_nf, sigma_w=sigma_w,
                               seed=int(self.cfg.seed) * 1000 + i,
                               device=str(self.device), learn_scales=learn,
                               exact_solve=self.spec_exact_solve)
                for i in range(self.spec_nheads)]
    def _spectral_band_weights(self, head, t: int) -> torch.Tensor:
        """Per-feature ridge weights at model-update time t:
        sum_d coefs[d] * lam(t + shifts[d]) * |w_j|^(2*degrees[d]).
        Per-degree time SHIFTS phase-shift the lambda schedule so different
        frequency bands clamp/release at different points of training."""
        # self.dg_gate_now (disagreement gate) scales the schedule uniformly so
        # the spectral penalty releases with the reward ensemble's convergence
        # too; it is 1.0 when the gate is disabled (no behaviour change).
        theta = [c * self.dg_gate_now * self.rg_gate_now * self.lam(t + s)
                 for c, s in zip(self.spec_coefs, self.spec_shifts)]
        return poly_weights(head.w2.sqrt(), self.spec_degrees, theta)

    def _spectral_refit(self):
        """Refit ALL heads on the full rolling cache — one (M, M) solve per
        head, ~0.04s wall. weights_mode=poly: schedule-driven lambda
        polynomial. weights_mode=snr: explicit Wiener weights from measured
        band SNR (cutoff at SNR=1), EMA-smoothed across refits; the lambda
        schedule is NOT applied (the Wiener weights are absolute)."""
        X = self.spec_cache_x.to(self.device)
        y = self.spec_cache_y.to(self.device)
        if self.spec_sigma == "auto" and not self.spec_heads:
            # first refit: calibrate the ladder from the cache (bridge run 5)
            from ..models.spectral import calibrate_sigma_ladder
            ladder, cinfo = calibrate_sigma_ladder(
                X.cpu(), y.cpu(), mults=self.spec_cal_mults,
                n_features=self.spec_nf, seed=int(self.cfg.seed))
            self.spec_sigma_star = cinfo["sigma_star"]
            self.spec_sigma = ladder      # frozen for the run (checkpointed)
            self.spec_heads = self._build_spec_heads(ladder)
        elif (self.spec_auto and self.spec_recal_every > 0 and self.spec_heads
              and self.spec_refits > 0
              and self.spec_refits % self.spec_recal_every == 0):
            # recalibration-on-drift: re-probe sigma*; rebuild the basis only
            # if it moved more than recal_drift x (c re-anchors immediately
            # below, so a rebuild at a refit boundary is safe)
            from ..models.spectral import calibrate_sigma_ladder
            ladder, cinfo = calibrate_sigma_ladder(
                X.cpu(), y.cpu(), mults=self.spec_cal_mults,
                n_features=self.spec_nf, seed=int(self.cfg.seed))
            new_star, old_star = cinfo["sigma_star"], self.spec_sigma_star
            self.spec_sigma_star = new_star   # always log the fresh value
            ratio = new_star / max(old_star, 1e-12)
            if ratio > self.spec_recal_drift or ratio < 1.0 / self.spec_recal_drift:
                self.spec_sigma = ladder
                self.spec_heads = self._build_spec_heads(ladder)
                self.spec_snr_ema = [None] * self.spec_nheads
                self.spec_recal_rebuilds = getattr(self, "spec_recal_rebuilds", 0) + 1
        for i, head in enumerate(self.spec_heads):
            if self.spec_weights_mode == "snr":
                theta, info = snr_band_weights(
                    head.features(X), y, head.w2.sqrt(),
                    n_bands=self.spec_snr_bands, generator=self.spec_snr_gen)
                theta = theta.to(self.device)
                if self.spec_snr_ema[i] is not None:
                    d = self.spec_snr_ema_decay
                    theta = d * self.spec_snr_ema[i] + (1 - d) * theta
                self.spec_snr_ema[i] = theta
                if i == 0:
                    self.spec_snr_info = info   # logged in model_update
                head.fit(X, y, weights=theta)
            else:
                head.fit(X, y, weights=self._spectral_band_weights(head, self.step))
        # band-SNR diagnostics on EVERY refit regardless of mode (improvement
        # plan #4): drift detection + overfit early-warning, ~free at M=512
        if self.spec_weights_mode != "snr" and self.spec_heads:
            h0 = self.spec_heads[0]
            _, self.spec_snr_info = snr_band_weights(
                h0.features(X), y, h0.w2.sqrt(),
                n_bands=self.spec_snr_bands, generator=self.spec_snr_gen)
        # closed-form fit quality (diagnostic, in-sample on the SAME cache the
        # heads were just fit on): ensemble-mean prediction MSE + R^2. Pure
        # observation under no_grad — no grad-bearing ops, no self.gen draws, no
        # change to the fit above. Stashed on self; emitted with the other
        # spectral/* keys in model_update. (No held-out slice is maintained for
        # the rolling cache, so this is honestly in-sample; named fit_* not val_*.)
        if self.spec_heads:
            with torch.no_grad():
                pred = torch.stack([h.predict(X) for h in self.spec_heads]).mean(0)
                ss_res = (y - pred).pow(2).sum()
                ss_tot = (y - y.mean()).pow(2).sum()
                self._last_fit_mse = float((y - pred).pow(2).mean())
                self._last_fit_r2 = (float(1.0 - ss_res / ss_tot)
                                     if float(ss_tot) > 1e-12 else 0.0)
        self.spec_refits += 1
        self.spec_since_refit = 0

    def _spectral_penalty_value(self) -> float:
        """EXACT mean-over-heads E_x ||grad^2 R||_F^2 — replaces the Hutchinson
        estimate in penalty/value so auto-dose, the adaptive horizon, and the
        dashboards keep working unchanged."""
        if not self.spec_heads:   # sigma_w=auto before the first refit
            return 0.0
        return sum(h.hessian_frobenius_sq() for h in self.spec_heads) / len(self.spec_heads)

    # ---------------- reward-surface + Hessian export (Studio viz-reward, M4) ----
    def reward_concat_fn(self):
        """The reward as f(x) over concatenated (z, a[, tau]) — the spectral
        ensemble mean if spectral, else the MLP head mean (RewardModel.on_concat).
        This is the SAME function the curvature penalty / imagination consume, so a
        surface slice or Hessian spectrum reflects the reward ACTUALLY in use.
        Before the first spectral refit (no heads) it returns zeros (no crash)."""
        if self.spec_enabled:
            heads = self.spec_heads
            if not heads:
                return lambda x: x.new_zeros(x.shape[:-1])
            return lambda x: torch.stack([h.predict(x) for h in heads]).mean(0)
        return self.reward.on_concat

    def reward_input_dim(self) -> int:
        """Dim of the reward's (z, a[, tau]) input — the surface / Hessian space."""
        return self.reward.k + self.reward.m + self.reward.task_dim

    def reward_hessian_eigs(self, center=None):
        """Reward-Hessian eigenvalues (descending) at `center` (default the origin
        of (z, a[, tau]) space) — backs the viz_ablation Hessian-spectrum panel."""
        from ..viz.surface_export import hessian_spectrum
        if center is None:
            center = torch.zeros(self.reward_input_dim(), device=self.device)
        return hessian_spectrum(self.reward_concat_fn(), center)

    def reward_surface_payload(self, plane=(0, 1), n: int = 81, extent: float = 2.0,
                               center=None, path=None, step=None, run=None) -> dict:
        """The pull.surface payload (R̂ slice + curvature + budget) for the live
        viz-reward panel — see mbrl.viz.surface_export.export_surface."""
        from ..viz.surface_export import export_surface
        if center is None:
            center = torch.zeros(self.reward_input_dim(), device=self.device)
        return export_surface(self.reward_concat_fn(), center, plane=plane, n=n,
                              extent=extent, path=path,
                              step=self.step if step is None else step, run=run)

    # ---------------- order-parameter / distance-to-collapse readout ----------------
    @torch.no_grad()
    def _representation_readouts(self, z: torch.Tensor) -> dict:
        """Superconductor-analogy diagnostics on the latent representation's Gram
        (PM 2026-06-14). The batch covariance G = Zc^T Zc / B is the order-parameter
        amplitude: as the coherent state depins (singular-Gram defects proliferate),
        eigendirections of G vanish → cond(G)=λ_max/λ_min → ∞ and eff_rank → 1. These
        are a LIVE distance-to-collapse readout the operator-spectrum metrics don't
        capture (that is the OPERATOR A; this is the representation z). The cf3 NaN
        had no representation-level early warning logged — this fills that gap.
        Intuition/diagnostic only (finite net ⇒ a driven bifurcation, NOT a sharp
        thermodynamic transition — no universal exponents implied). Cheap k×k eigh,
        no_grad — never differentiated, so it cannot perturb training."""
        zc = z - z.mean(0, keepdim=True)
        G = (zc.transpose(-1, -2) @ zc) / max(zc.shape[0], 1)      # (k,k) PSD covariance
        ev = torch.linalg.eigvalsh(G.float()).clamp_min(0.0)       # ascending eigenvalues
        tot = ev.sum().clamp_min(1e-12)
        p = (ev / tot).clamp_min(1e-12)
        spectral_entropy = float(-(p * p.log()).sum())
        out = {"latent/gram_cond": float(ev[-1] / ev[0].clamp_min(1e-12)),  # σmax/σmin
               "latent/gram_eff_rank": float(torch.exp(torch.as_tensor(spectral_entropy))),
               "latent/gram_spectral_entropy": spectral_entropy}
        # full Gram spectrum (descending; latent/eig00 = largest) — the per-eigenvalue
        # series that turns the 3 summaries above into a time×eigenvalue heatmap / latent
        # PCA-over-training. eig already computed ⇒ ~free. Lets analyze_loss_dynamics.py
        # fit the closed-form spectral equilibria (active modes→ceiling, dead→floor).
        ev_desc = ev.flip(0)
        for i in range(ev_desc.shape[0]):
            out["latent/eig%02d" % i] = float(ev_desc[i])
        # GATED viz snapshot (default-OFF): persist the SAME Gram G + its descending
        # spectrum (already computed above, ~free) to tensors/gram_<step>.json for
        # the Studio cov-gram surface. In twin mode also snapshot op_d/op_p if the
        # operator matrices are cheaply on hand. Wrapped so viz can never kill a
        # run; self.step is already advanced (_model_step runs before this readout).
        if self.tensor_every > 0 and (self.step % self.tensor_every == 0):
            self._snapshot_gram(z, G, ev_desc)
        return out

    # ---------------- gated Studio named-tensor snapshots (default-OFF) ----------
    def _write_tensor_json(self, name: str, payload: dict) -> None:
        """Write-then-rename a named tensor to <root>/runs/<run>/tensors/<name>_<step>.json.

        Mirrors mbrl.viz.surface_export.write_surface_json's destination convention
        (one level over: tensors/ beside surfaces/), with an atomic tmp+rename so a
        torn write can never be half-read by the stdlib TensorIndex. Creates the
        tensors/ dir. Never raises into training — any viz IO error only warns."""
        try:
            out_dir = Path(self._viz_root) / "runs" / self._viz_run / "tensors"
            out_dir.mkdir(parents=True, exist_ok=True)
            final = out_dir / f"{name}_{int(self.step)}.json"
            tmp = out_dir / f".{name}_{int(self.step)}.json.tmp"
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, final)   # atomic on POSIX: readers see whole-or-nothing
        except Exception as _e:  # noqa: BLE001 — viz must never kill a training run
            print(f"[warn] tensor snapshot {name!r} failed ({_e!r}); training continues")

    @torch.no_grad()
    def _snapshot_gram(self, z: torch.Tensor, G: torch.Tensor,
                       ev_desc: torch.Tensor) -> None:
        """Snapshot the latent Gram G = E[zzᵀ] (the cov-gram surface) — the EXACT G
        and descending eigenvalues from _representation_readouts. In twin mode also
        snapshot the (batch-mean) policy/dynamics operator matrices op_d/op_p, which
        are cheaply available from the operators. no_grad/detached; default-OFF."""
        self._write_tensor_json("gram", {
            "run": self._viz_run, "name": "gram", "step": int(self.step),
            "matrix": G.detach().cpu().tolist(),
            "eig": ev_desc.detach().cpu().tolist(),   # descending eigenvalues
        })
        # twin operators: A_d(d), A_p(p) are per-sample matrix FIELDS (B,k,k); the
        # batch mean is one representative (k,k) operator per branch — cheap, no_grad.
        if self.dual_latent and self.dual is not None and self.dual.mode == "twin":
            try:
                d = self.dual.d_of(z)
                p = self.dual.p_of(z)
                Ad, _ = self.dual.op_d.operators(d)
                Ap, _ = self.dual.op_p.operators(p)
                self._write_tensor_json("op_d", {
                    "run": self._viz_run, "name": "op_d", "step": int(self.step),
                    "matrix": Ad.mean(0).detach().cpu().tolist()})
                self._write_tensor_json("op_p", {
                    "run": self._viz_run, "name": "op_p", "step": int(self.step),
                    "matrix": Ap.mean(0).detach().cpu().tolist()})
            except Exception as _e:  # noqa: BLE001 — optional; never block the gram snapshot
                print(f"[warn] twin-operator snapshot failed ({_e!r}); training continues")

    def _snapshot_input_salience(self, obs: torch.Tensor, a: torch.Tensor,
                                 tau: torch.Tensor | None) -> None:
        """INPUT/FEATURE salience |∂reward/∂obs| over the batch (default-OFF).

        A SELF-CONTAINED autograd pass that touches NO training tensor, optimizer or
        RNG (no self.gen draws): clone+detach the obs into a fresh leaf, forward
        obs→encoder→z→reward head (reading the policy latent p in dual mode, the same
        wiring training uses), backward the scalar mean predicted reward, and read
        sal = obs.grad.abs().mean(0). Snapshots to reward_input_salience_<step>.json.
        Differentiating all the way to obs through the encoder works directly here, so
        this is true ∂/∂obs (no fallback to ∂/∂z needed)."""
        try:
            obs_l = obs.detach().clone().requires_grad_(True)   # fresh leaf, own graph
            a_d = a.detach()
            tau_d = tau.detach() if tau is not None else None
            with torch.enable_grad():
                # RNG-FREE encode: the VAE encoder's forward() samples (global-RNG
                # randn) when training — use its deterministic MEAN here so this pass
                # consumes NO RNG and stays byte-exact. Deterministic encoders are
                # already sample-free, so forward() is fine for them.
                z = (self.encoder.moments(obs_l)[0] if self.enc_vae
                     else self.encoder(obs_l))
                rz = self.dual.p_of(z) if self.dual_latent else z   # reward reads p in dual mode
                r_pred = self.reward(rz, a_d, tau_d)                 # head mean (symlog space)
                scalar = r_pred.mean()
                grad = torch.autograd.grad(scalar, obs_l)[0]         # ∂reward/∂obs
            sal = grad.detach().abs().mean(dim=0)                    # (obs_dim,)
            self._write_tensor_json("reward_input_salience", {
                "run": self._viz_run, "name": "reward_input_salience",
                "step": int(self.step), "salience": sal.cpu().tolist(),
                "dims": int(obs_l.shape[-1]), "wrt": "obs"})
        except Exception as _e:  # noqa: BLE001 — viz must never kill a training run
            print(f"[warn] salience snapshot failed ({_e!r}); training continues")

    # ---------------- rank-2 reward⊥energy frame (cf5) ----------------
    def _rank2_frame(self, z, d, p, a, tau):
        """The cf5 frame loss term + diagnostics (see regularization/rank2_frame.py):
        press z into a rank-2 subspace whose two orthogonal axes are reward-ascent
        (∇_z R) and energy-descent (−∇_z E, lyapunov | op_d's contractive mode). z must
        carry grad (it is the live encoder output). Returns (loss_term, metrics)."""
        from ..regularization.rank2_frame import (axis_cos2, rank2_tail_penalty,
                                                   lyapunov_grounding, contractive_axis_in_d,
                                                   spectral_shell_penalty, log_det_barrier,
                                                   spectral_band_penalty, spectral_compress_penalty)
        dl = self.dual
        metrics = {}
        term = z.new_zeros(())
        # axis orthogonality: cos²(∇_z R, ∇_z E) → 0, double-backward on a subsample
        if self.frame_w_ortho > 0.0:
            n = min(self.frame_subsample, z.shape[0])
            zs = z[:n]
            a_s = a[:n]
            tau_s = tau[:n] if tau is not None else None
            r_hat = self.reward(dl.p_of(zs), a_s, tau_s)            # reward in z via P
            g_r = torch.autograd.grad(r_hat.sum(), zs, create_graph=True)[0]
            if self.frame_energy_mode == "contractive":
                v_min = contractive_axis_in_d(dl.op_d, dl.d_of(zs))  # detached d-direction
                proj = (dl.d_of(zs) * v_min).sum(-1)
                g_e = torch.autograd.grad(proj.sum(), zs, create_graph=True)[0]
            else:                                                  # lyapunov
                e = self.energy(dl.d_of(zs))
                g_e = torch.autograd.grad(e.sum(), zs, create_graph=True)[0]
            ortho = axis_cos2(g_r, g_e)
            term = term + self.frame_w_ortho * ortho
            metrics["frame/ortho_cos"] = float(ortho.detach().clamp_min(0.0).sqrt())
        # rank-2 pressure: press z's variance into its top-`target_rank` eigendirections
        if self.frame_w_rank2 > 0.0:
            tail = rank2_tail_penalty(z, self.frame_target_rank)
            term = term + self.frame_w_rank2 * tail
            metrics["frame/rank2_tail"] = float(tail.detach())
        # cf10 two-sided rank-k energy shell (Ginzburg-Landau well): hold the top-k Gram
        # eigenvalues at the setpoint and push the rest to 0 — anti-collapse in EVERY mode
        if self.frame_w_shell > 0.0:
            shell = spectral_shell_penalty(z, self.frame_target_rank, self.frame_shell_target,
                                           self.frame_shell_floor)
            term = term + self.frame_w_shell * shell
            metrics["frame/shell"] = float(shell.detach())
        # cf12 log-det / KL volume barrier: push eigenvalues off zero -> bound cond(G)
        if self.frame_w_logdet > 0.0:
            ld = log_det_barrier(z, self.frame_logdet_eps)
            term = term + self.frame_w_logdet * ld
            metrics["frame/logdet_barrier"] = float(ld.detach())
        # cf14 two-sided spectral band: bound EVERY Gram eigenvalue between a hard floor
        # and a hard ceiling, free in between -> cond bounded, rank EMERGES (no rank demand)
        if self.frame_w_band > 0.0:
            band = spectral_band_penalty(z, self.frame_band_ceiling, self.frame_band_floor,
                                         self.frame_band_floor_shape, self.frame_band_floor_beta)
            term = term + self.frame_w_band * band
            metrics["frame/band"] = float(band.detach())
        # cf15 nuclear-norm compression Σ√(relu(λ-floor)): the inward pressure the band's
        # free interior lacks — concentrate variance into modes that earn it (rank emerges
        # lower/decisively). Compresses only ABOVE the floor (inert below ⇒ no early-latent
        # shock, never fights the floor wall).
        if self.frame_w_compress > 0.0:
            comp = spectral_compress_penalty(z, self.frame_band_floor)
            term = term + self.frame_w_compress * comp
            metrics["frame/compress"] = float(comp.detach())
        # lyapunov grounding: the autonomous drift must descend E
        if (self.frame_energy_mode == "lyapunov" and self.energy is not None
                and self.frame_w_lyap > 0.0):
            ground = lyapunov_grounding(self.energy, dl.op_d, d, dl.m)
            term = term + self.frame_w_lyap * ground
            metrics["frame/lyap_resid"] = float(ground.detach())
        return term, metrics

    def _balance_align_energy(self, align, energy):
        """cf7 equilibrium coupling: hold the alignment (couple) and energy
        (dissipativity) penalties at EQUAL influence so neither outweighs the other.
        Each is normalized by its running magnitude (EMA) then summed — the fixed
        couple_weight/w_dissip are bypassed. Equilibrium point = balanced normalized
        gradients. EMA state checkpointed (bitwise resume)."""
        dec = self.frame_balance_decay
        a_mag = float(align.detach().abs()); e_mag = float(energy.detach().abs())
        self._bal_ema_align = (a_mag if self._bal_ema_align is None
                               else dec * self._bal_ema_align + (1 - dec) * a_mag)
        self._bal_ema_energy = (e_mag if self._bal_ema_energy is None
                                else dec * self._bal_ema_energy + (1 - dec) * e_mag)
        # cap the adaptive weights: when a side's running magnitude is ~0 (already
        # satisfied — e.g. couple≈0 when the sectors are aligned), its 1/ema weight
        # explodes (observed bal_w_align≈1e6) and would amplify that side's gradient a
        # million-fold. Cap at 10× so a near-zero penalty can't blow up; the side that
        # genuinely needs balancing (e.g. the energy) sits well under the cap.
        wmax = 10.0
        wa = min(self.frame_balance_w / (self._bal_ema_align + 1e-8), wmax)
        we = min(self.frame_balance_w / (self._bal_ema_energy + 1e-8), wmax)
        term = wa * align + we * energy
        return term, {"frame/bal_w_align": wa, "frame/bal_w_energy": we,
                      "frame/bal_align": a_mag, "frame/bal_energy": e_mag}

    def _anneal_operator_radius(self):
        """Anneal the svband ceiling radius_max from rad_anneal_start → rad_anneal_floor
        on exp(−step/τ) — asymptotic, never reaching the floor. The op_d energy ratio
        |λ|² rides the natural ~1 early (where the dynamics fit wants it) and is pulled
        toward floor² over training, so a persistent descending target can't be undone
        the way a one-shot init was. op_d only; op_p stays conservative. No-op when τ≤0;
        deterministic in self.step ⇒ no checkpoint state and bitwise-exact on resume."""
        if self.rad_anneal_tau <= 0.0:
            return
        rm = self.rad_anneal_floor + (self.rad_anneal_start - self.rad_anneal_floor) * \
            float(np.exp(-self.step / self.rad_anneal_tau))
        self._radius_ceil = rm
        if self.dual_latent and self.dual is not None and getattr(self.dual, "op_d", None) is not None:
            self.dual.op_d.radius_max = rm
        elif self.dyn_operator:
            self.dynamics.radius_max = rm

    # ---------------- model learning ----------------
    def _encode_batch(self, batch):
        """Shared model-update prologue: batch -> device, optional VAE encode, EMA
        target. Returns (obs, a, r, obs_next, tau, z, z_next_tgt, vae_terms, vae_metrics).
        Identical for the primary and dual-latent paths (no self.gen draws)."""
        if self.task_dim:
            obs, a, r, obs_next, tau = (x.to(self.device) for x in batch)
        else:
            obs, a, r, obs_next = (x.to(self.device) for x in batch)
            tau = None
        vae_terms = vae_metrics = None
        if self.enc_vae:   # one forward: recon + KL + the z sample
            recon, kl, z = self.encoder.losses(obs)
            vae_terms = self.vae_recon_w * recon + self.vae_beta * kl
            vae_metrics = {"vae/recon": recon.item(), "vae/kl": kl.item()}
        else:
            z = self.encoder(obs)
        with torch.no_grad():
            z_next_tgt = self.ema(obs_next)
        return obs, a, r, obs_next, tau, z, z_next_tgt, vae_terms, vae_metrics

    def _reward_target(self, r):
        """Reward fit target: symlog(r) when model.symlog_reward is on, else r.
        Also tracks the running max |symlog(r)| over real batches (symlog_bound,
        checkpointed) that bounds the imagination symexp clamp. Byte-identical in
        the primary and dual-latent paths; no self.gen draws."""
        r_target = symlog(r) if self.symlog else r
        if self.symlog:  # track the real-data symlog range for the imagination clamp
            batch_max = r_target.abs().max().item()
            if np.isfinite(batch_max):  # NaN hygiene: never poison the bound
                self.symlog_bound = max(self.symlog_bound, batch_max)
        return r_target

    def _model_step(self, loss, pen_val) -> bool:
        """Shared model-optimizer epilogue for both model_update paths: zero_grad,
        backward, clip the ENCODER grad to 100, optionally skip the step when the
        full model grad-norm is non-finite (skip_nonfinite, default False ⇒ the
        unconditional step below is byte-identical to the legacy primary), step,
        EMA-update the encoder, advance self.step, then update the adaptive-horizon
        penalty EMA/peak. Returns whether the optimizer stepped. No self.gen draws
        (the non-finite guard is a measure-only inf-norm). Keep every ratchet/EMA
        scalar update at the same control-flow point as before."""
        import math as _math
        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 100.0)
        _stepped = True
        if self.skip_nonfinite:   # cf4: a NaN model loss must not poison the weights
            _mnorm = torch.nn.utils.clip_grad_norm_(
                (pp for g in self.model_opt.param_groups for pp in g["params"]),
                float("inf"))     # measure-only (inf ⇒ no extra clip beyond the encoder)
            _stepped = bool(torch.isfinite(_mnorm))
            if not _stepped:
                self._nonfinite_skips += 1
        if _stepped:
            self.model_opt.step()
            self.ema.update(self.encoder)
        self.step += 1
        # penalty EMA + running peak — the adaptive-horizon certificate signal.
        # NaN hygiene: one non-finite penalty value must not poison the EMA
        # forever (a poisoned EMA crashed the horizon controller in the shiny run).
        if _math.isfinite(pen_val):
            self.pen_ema = (pen_val if self.pen_ema is None
                            else self.ah_decay * self.pen_ema + (1 - self.ah_decay) * pen_val)
            self.pen_peak = max(self.pen_peak, self.pen_ema)
        return _stepped

    def model_update(self, batch) -> dict:
        self._apply_lr_schedule()      # tie model LR to lambda's exponent (no-op when off)
        self._anneal_operator_radius()  # decay the svband ceiling toward radius_max (no-op when off)
        if self.dual_latent:
            return self._model_update_dual(batch)
        obs, a, r, obs_next, tau, z, z_next_tgt, vae_terms, vae_metrics = self._encode_batch(batch)

        dyn_calib = {}
        if self.dyn_stochastic:   # Gaussian NLL on the transition distribution
            dyn_loss = self.dynamics.nll(z, a, z_next_tgt)
            # calibration telemetry (run-9 criterion (a)): does predicted sigma
            # track realized error? corr > 0.5 = calibrated; ratio ~ 0.8 for a
            # well-calibrated Gaussian (E|N(0,1)| = 0.798)
            with torch.no_grad():
                mu, lv = self.dynamics.moments(z, a)
                std = torch.exp(0.5 * lv).flatten()
                err = (z_next_tgt - mu).abs().flatten()
                cov = ((std - std.mean()) * (err - err.mean())).mean()
                denom = (std.std(unbiased=False) * err.std(unbiased=False)).clamp_min(1e-12)
                dyn_calib = {"dyn/calib_corr": float(cov / denom),
                             "dyn/calib_ratio": float(err.mean()
                                                      / std.mean().clamp_min(1e-9)),
                             "dyn/pred_std": float(std.mean())}
        else:
            if self.dyn_ensemble:
                # deep-ensemble discipline: EVERY member regresses to the data;
                # diversity persists from independent inits (fitting only the
                # mean would leave the disagreement signal unregularized)
                preds = self.dynamics.all_members(z, a)            # (M, B, k)
                dyn_loss = F.mse_loss(preds, z_next_tgt.unsqueeze(0).expand_as(preds))
                with torch.no_grad():
                    dyn_calib = {"dyn/disagreement":
                                 float(self.dynamics.disagreement(z, a).mean())}
            else:
                dyn_loss = F.mse_loss(self.dynamics(z, a), z_next_tgt)
        # reward model predicts symlog(r) when model.symlog_reward is on;
        # imagination applies symexp to whatever it consumes (behaviour_update)
        r_target = self._reward_target(r)
        # disagreement-gated lambda (penalty.disagreement_gate): compute the
        # gate HERE — before the spectral refit below reads self.dg_gate_now via
        # the closed-form theta weights, so champion/spectral are gated too (not
        # just the MLP Hutchinson path). Signal = the reward ensemble's head-std
        # on this batch (the MLP/aux heads, trained in every stack: native in
        # mlp, encoder-grounding aux in spectral/champion). Detached, no RNG;
        # dis_ema/peak checkpointed -> bitwise resume. Updated every step (warm
        # through auto-dose warmup). self.dg_gate_now stays 1.0 when disabled.
        dg_metrics = {}
        if self.dg_enabled:
            # signal: on spectral stacks use the spectral ensemble's own head
            # spread (set last step; tracks the SPECTRAL reward's convergence so
            # the gate actually releases — the aux-MLP signal stayed pinned at
            # 1.0, throttling bandwidth all run). Falls back to the MLP/reward
            # heads (mlp stack, or spectral before the first refit builds heads).
            if self.spec_enabled and self._spec_head_dis is not None:
                d_dis = self._spec_head_dis
            else:
                with torch.no_grad():
                    d_dis = self.reward.all_heads(z.detach(), a, tau).std(0).mean().item()
            import math as _mdg
            if _mdg.isfinite(d_dis):
                self.dis_ema = (d_dis if self.dis_ema is None
                                else self.dg_decay * self.dis_ema
                                + (1 - self.dg_decay) * d_dis)
                self.dis_peak = max(self.dis_peak, self.dis_ema)
            if self.dis_ema is not None and self.dis_peak > 0:
                d_norm = min(max(self.dis_ema / self.dis_peak, 0.0), 1.0)
                self.dg_gate_now = self.dg_floor + (1.0 - self.dg_floor) * d_norm
            dg_metrics = {"penalty/disagreement": d_dis,
                          "penalty/dg_gate": self.dg_gate_now}

        spec_metrics = {}
        if self.spec_enabled:
            # Spectral reward path: no MLP reward fit loss — the closed-form
            # heads are refit from the rolling cache instead. Cache rows live
            # in the SAME coords as the penalty: cat(z.detach(), a[, tau]).
            parts_s = [z.detach(), a.detach()]
            if tau is not None:
                parts_s.append(tau.detach())
            x_spec = torch.cat(parts_s, dim=-1)
            self.spec_cache_x = torch.cat(
                [self.spec_cache_x, x_spec.cpu()])[-self.spec_cache_size:]
            self.spec_cache_y = torch.cat(
                [self.spec_cache_y, r_target.detach().cpu()])[-self.spec_cache_size:]
            self.spec_since_refit += 1
            # First refit as soon as the cache holds >= n_features rows (keeps
            # the (M, M) solve well-posed); then every refit_every updates.
            # Before the first refit the heads predict zeros — logged below.
            if (self.spec_cache_x.shape[0] >= self.spec_nf
                    and (self.spec_refits == 0
                         or self.spec_since_refit >= self.spec_refit_every)):
                self._spectral_refit()
            with torch.no_grad():  # diagnostic only: ensemble-mean fit MSE
                if self.spec_heads:
                    ph = torch.stack([h.predict(x_spec) for h in self.spec_heads])
                    pred = ph.mean(0)
                    rew_loss_val = F.mse_loss(pred, r_target).item()
                    # spectral ensemble's OWN head-spread — the disagreement
                    # signal the gate uses on spectral stacks (read by the NEXT
                    # step's early gate block; the aux-MLP-head signal does not
                    # track the spectral reward's convergence -> never releases).
                    if len(self.spec_heads) >= 2:
                        self._spec_head_dis = ph.std(0).mean().item()
                else:   # sigma_w=auto, pre-calibration: heads not built yet
                    rew_loss_val = float(r_target.pow(2).mean().item())
            if self.spec_sigma == "learned" and self.spec_refits > 0:
                # one gradient step on the bandwidths: fit error of the
                # CURRENT batch, c held fixed (re-anchored at next refit).
                # x_spec is built from detached (z, a) — only log_s moves.
                pred = torch.stack([h.predict(x_spec)
                                    for h in self.spec_heads]).mean(0)
                sig_loss = F.mse_loss(pred, r_target.detach())
                if self.spec_sigma_wd > 0:   # elastic anchor toward init
                    sig_loss = sig_loss + self.spec_sigma_wd * sum(
                        h.log_s.pow(2).sum() for h in self.spec_heads)
                self.spec_sigma_opt.zero_grad(set_to_none=True)
                sig_loss.backward()
                self.spec_sigma_opt.step()
            rew_loss = None
            spec_metrics = {"spectral/refits": self.spec_refits,
                            "spectral/cache_n": self.spec_cache_x.shape[0],
                            "spectral/fitted": float(self.spec_refits > 0)}
            if self.spec_sigma_star is not None:
                spec_metrics["spectral/sigma_star"] = self.spec_sigma_star
                spec_metrics["spectral/recal_rebuilds"] = getattr(
                    self, "spec_recal_rebuilds", 0)
            if self.spec_refits > 0:   # closed-form fit quality (diagnostic, in-sample)
                spec_metrics["spectral/fit_mse"] = self._last_fit_mse
                spec_metrics["spectral/fit_r2"] = self._last_fit_r2
            if self.spec_sigma == "learned" and self.spec_heads:
                with torch.no_grad():   # effective per-block sigmas, head 0
                    h0 = self.spec_heads[0]
                    eff = torch.exp(h0.log_s)
                    for kk in range(len(eff)):
                        spec_metrics[f"spectral/sigma_scale_{kk}"] = float(eff[kk])
            if self.spec_snr_info:   # logged for every mode (plan #4)
                inf = self.spec_snr_info
                if inf.get("band_snrs"):
                    spec_metrics["spectral/snr_min"] = min(inf["band_snrs"])
                    spec_metrics["spectral/snr_max"] = max(inf["band_snrs"])
                    for _bi, _bsnr in enumerate(inf["band_snrs"]):   # per-band SNR ladder
                        spec_metrics[f"spectral/band_snr{_bi:02d}"] = float(_bsnr)
                if "w_at_snr1" in inf:  # sigma_eff at the SNR=1 cutoff —
                    # the user's hypothesis: this sits at sigma = 1
                    spec_metrics["spectral/sigma_at_snr1"] = (
                        inf["w_at_snr1"] / float(np.sqrt(self.spec_cache_x.shape[1])))
        else:
            rew_loss = F.mse_loss(self.reward(z, a, tau), r_target)
            rew_loss_val = rew_loss.item()

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
        if self.spec_enabled:
            # The Hutchinson penalty on the reward is SKIPPED: the spectral
            # heads' H^2 penalty is EXACT and already inside the closed-form
            # refit. Only the optional dynamics term remains stochastic.
            pen = torch.zeros((), device=self.device)
        else:
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
            pen = penalty_fn(fn, x_pen, n_probes=self.cfg.penalty.n_probes,
                             generator=self.gen)
        if self.cfg.penalty.penalize_dynamics:  # optional transversal term (R8/R9)
            k = z.shape[-1]
            dyn_det = self.dynamics.mean if self.dyn_stochastic else self.dynamics
            fn_t = lambda x: dyn_det(x[..., :k], x[..., k:k + a.shape[-1]]).sum(-1)
            za = torch.cat([z.detach(), a], dim=-1)
            pen = pen + penalty_fn(fn_t, za, n_probes=self.cfg.penalty.n_probes,
                                   generator=self.gen)

        # penalty/value: spectral => EXACT mean-over-heads H^2 (auto-dose,
        # adaptive horizon, dashboards consume it unchanged); else Hutchinson.
        pen_val = self._spectral_penalty_value() if self.spec_enabled else pen.item()

        # ---- auto-dosed lambda: lam=0 during warmup, then dose lam0 once ----
        if self.ad_enabled and self.ad_count < self.ad_warmup:
            lam_t = 0.0
            self.ad_count += 1
            if self.ad_count > self.ad_tail_start:  # tail window only
                # Dose against the REWARD fit — the quantity the penalty
                # regularizes — which is a non-negative MSE in BOTH paths
                # (spectral: the closed-form heads' ensemble-mean fit MSE;
                # MLP: rew_loss). NOT the dynamics loss: on the gaussian-dynamics
                # path (champion) it is a Gaussian NLL that goes NEGATIVE as the
                # model sharpens (the `lv` term), which drove lam0_auto negative
                # -> floored -> the penalty was silently OFF (champion -247..-310
                # vs fixed-dose -160s, 2026-06-11).
                self.ad_fit_sum += rew_loss_val
                self.ad_pen_sum += pen_val
            if self.ad_count == self.ad_warmup:
                n_tail = self.ad_warmup - self.ad_tail_start
                mean_fit = self.ad_fit_sum / n_tail
                mean_pen = self.ad_pen_sum / n_tail
                self.lam0_auto = min(
                    self.ad_target_ratio * max(mean_fit, 0.0) / max(mean_pen, 1e-12),
                    self.ad_lam_max)   # max(,0): belt against any negative fit ref
                self.lam.lam0 = self.lam0_auto
        else:
            lam_t = self.lam(self.step)
        # disagreement-gated lambda (MLP path): the gate was computed early
        # (self.dg_gate_now, before the spectral refit so closed-form theta sees
        # it too); here it scales the Hutchinson penalty's lambda. The return-gate
        # (rg_gate_now, set on eval) composes multiplicatively — both in [floor,1].
        lam_t = max(lam_t * self.dg_gate_now * self.rg_gate_now, self.lambda_min)  # hard floor
        if rew_loss is None:  # spectral: the MLP reward fit is skipped entirely
            loss = dyn_loss + lam_t * pen
            # ENCODER-GROUNDING AUX (2026-06-08, HalfCheetah collapse): in
            # spectral mode the encoder's only gradient is dyn MSE — whose
            # trivial solution is a near-constant z (observed: loss/dyn
            # 2e-5, 1000x below the MLP arm, returns random). The bypassed
            # MLP reward head is trained as an auxiliary loss purely to keep
            # z reward-informative; the spectral head remains the reward used
            # everywhere. No Hutchinson on the aux (spectral penalty is exact).
            if self.spec_enabled and self.spec_aux:
                aux = F.mse_loss(self.reward(z, a, tau), r_target)
                loss = loss + self.spec_aux_weight * aux
                spec_metrics["spectral/aux_loss"] = aux.item()
        else:
            loss = dyn_loss + rew_loss + lam_t * pen  # original op order (bitwise)
        if vae_terms is not None:   # run 10: recon + KL grounding
            loss = loss + vae_terms
            spec_metrics |= vae_metrics
        if self.spec_enabled:   # collapse early-warning, ~free
            spec_metrics["latent/z_std"] = z.detach().std(0).mean().item()

        # operator-field structural priors (model.dynamics=operator): keep A(z) a
        # coherent operator bundle. Penalties on DETACHED z (founding-doc latent-
        # coord discipline, R16) -> they shape the A/B nets, not the encoder.
        # spectral_summary is logged every step (cheap, no grad); the weighted
        # penalty term is added only when some w_* > 0 (default all 0 = no-op).
        op_metrics = {}
        if self.dyn_operator:
            op_metrics = self.dynamics.spectral_summary(z.detach())
            if any(self.op_w.values()):
                sp = self.dynamics.structural_penalties(z.detach())
                loss = loss + sum(self.op_w[kk] * sp[kk] for kk in self.op_w)
                op_metrics |= {f"op/pen_{kk}": float(sp[kk].detach()) for kk in sp}

        import math as _math   # used by the channel-capacity diagnostics below
        self._model_step(loss, pen_val)   # zero_grad/backward/clip/step/ema/step++/pen_ema

        # GATED input-salience snapshot (default-OFF, self.step already advanced): a
        # self-contained ∂reward/∂obs pass on a detached obs leaf — no training
        # tensor/opt/RNG touched (NO self.gen draws). The Gram snapshot rides
        # _representation_readouts(z) below.
        if self.salience_every > 0 and (self.step % self.salience_every == 0):
            self._snapshot_input_salience(obs, a, tau)

        # ---- latent-as-channel diagnostics (rate-distortion / IB frontier) ----
        # rate  = E KL(q_φ(z|x) ‖ N(0,I)) >= I(x;z)  (only defined for the VAE
        #         stochastic channel; deterministic encoder is noiseless -> no rate).
        # task-relevant capacities via the Gaussian-channel proxy ½ln(1+SNR):
        #   I(z;r)  from the reward-fit SNR,  I(z;z') from the 1-step dynamics SNR.
        # See docs/channel_capacity_formalization_2026-06-11.md.
        info_metrics = {}
        with torch.no_grad():
            eps = 1e-8
            snr_r = float(r_target.var().item()) / max(rew_loss_val, eps)
            info_metrics["info/task_reward_nats"] = 0.5 * _math.log1p(max(snr_r, 0.0))
            mu_pred = (self.dynamics.moments(z, a)[0] if self.dyn_stochastic
                       else self.dynamics(z, a))           # deterministic mean (ensemble fwd = mean)
            mse_dyn = float(F.mse_loss(mu_pred, z_next_tgt).item())
            snr_d = float(z_next_tgt.var().item()) / max(mse_dyn, eps)
            info_metrics["info/task_dyn_nats"] = 0.5 * _math.log1p(max(snr_d, 0.0))
            info_metrics["info/dyn_mse"] = mse_dyn
            if self.enc_vae:
                rate = vae_metrics["vae/kl"]               # nats; >= I(x;z)
                info_metrics["info/rate_nats"] = rate
                info_metrics["info/rate_bits"] = rate / _math.log(2)
                # IB efficiency: task-relevant capacity earned per representation bit
                task = info_metrics["info/task_reward_nats"] + info_metrics["info/task_dyn_nats"]
                info_metrics["info/ib_efficiency"] = task / max(rate, eps)

        out = {"loss/dyn": dyn_loss.item(), "loss/reward": rew_loss_val,
               "penalty/value": pen_val, "penalty/lambda": lam_t,
               "penalty/return_gate": self.rg_gate_now,
               "loss/total": loss.item(), "step": self.step, **spec_metrics,
               **dyn_calib, **dg_metrics, **info_metrics, **op_metrics,
               **self._representation_readouts(z)}
        if self.lam0_auto is not None:
            out["penalty/lam0_auto"] = self.lam0_auto
        return out

    def _imagined_reward(self, z, a, tau=None):
        """Reward as consumed by imagination: per-head symexp (if the model is
        trained in symlog space), then ensemble mean - pessimism * std.
        Returns (reward (B,), mean head disagreement scalar).
        Spectral path: per-head closed-form predict — cos features, fully
        differentiable in (z, a) so the policy gradient flows through it."""
        if self.spec_enabled:
            parts = [z, a] if tau is None else [z, a, tau]
            x = torch.cat(parts, dim=-1)
            if not self.spec_heads:   # sigma_w=auto, pre-calibration
                return (torch.zeros(z.shape[0], device=z.device),
                        z.new_zeros(()))
            heads = torch.stack([h.predict(x) for h in self.spec_heads])  # (heads, B)
        else:
            heads = self.reward.all_heads(z, a, tau)      # (n_heads, B)
        if self.symlog:
            # Clamp BEFORE symexp: imagined rollouts extrapolate, and
            # expm1(|x| > ~89) overflows float32 -> inf rewards -> NaN policy
            # (the shiny-run crash). The bound is DATA-DRIVEN: margin * the
            # running max |symlog(r)| seen in real batches — the old fixed
            # +-20 still allowed symexp up to 4.8e8, which drove imagined
            # return variance to 1e19 in low-lambda windows.
            bound = self.symexp_margin * self.symlog_bound
            heads = symexp(heads.clamp(-bound, bound))
        if heads.shape[0] == 1:
            rew, dis = heads[0], heads.new_zeros(())
        else:
            std = heads.std(0)
            rew, dis = heads.mean(0) - self.pessimism * std, std.mean().detach()
        if self.reward_clip > 0.0:   # cf4: cap imagined reward so an expansive op_p
            with torch.no_grad():   # diagnostic: pre-clip over-threshold fraction
                self._reward_clip_over += float(
                    (rew.detach().abs() > self.reward_clip).sum())
                self._reward_clip_tot += float(rew.numel())
            rew = rew.clamp(-self.reward_clip, self.reward_clip)   # can't blow returns to inf
        return rew, dis

    def _imagination_horizon(self) -> int:
        """Curvature-certified horizon: imagine further only as the penalty EMA
        falls off its running peak (low curvature => trust longer rollouts)."""
        if self.use_planner:
            return self.planner.H        # the plan length is fixed (no adaptive H)
        if not self.ah_enabled:
            return int(self.cfg.imagination.horizon)
        if self.pen_ema is None or self.pen_peak <= 0:
            return self._horizon_ratchet(self.ah_h_min)  # no curvature evidence yet
        frac = min(max(1.0 - self.pen_ema / max(self.pen_peak, 1e-12), 0.0), 1.0)
        if not np.isfinite(frac):  # last-ditch guard: never crash the loop
            return self._horizon_ratchet(self.ah_h_min)
        H = int(round(self.ah_h_min + (self.ah_h_max - self.ah_h_min) * frac))
        return self._horizon_ratchet(H)

    def _horizon_ratchet(self, H: int) -> int:
        """cf17 monotonic horizon floor: once the adaptive H first reaches ratchet_base,
        lock a running-max floor so H can rise but never fall below its peak again —
        stability at peak convergence, no penalty-spike collapse. Idempotent (max-based),
        and its state (_ah_ratchet_floor/_on) is checkpointed for bitwise resume. Off ⇒
        returns H unchanged (the original adaptive behaviour)."""
        if not self.ah_ratchet:
            return H
        if not self._ah_ratchet_on and H >= self.ah_ratchet_base:
            self._ah_ratchet_on = True
        if self._ah_ratchet_on:
            self._ah_ratchet_floor = max(self._ah_ratchet_floor, H)
            return self._ah_ratchet_floor
        return H

    def _reward_frac(self) -> float:
        """rf ∈ [0,1] from the return EMA: 0 = at/below mid (explore), 1 = at/above mid+scale
        (exploit). Drives all three reward-adaptive policy knobs. 0 until the first eval."""
        if self.ret_ema is None:
            return 0.0
        return min(max((self.ret_ema - self.ra_mid) / max(self.ra_scale, 1e-6), 0.0), 1.0)

    def _apply_lr_schedule(self) -> None:
        """Tie the model LR to lambda's exponent (PM 2026-06-15): model_opt lr = lr_sched(step)
        (cuberoot, same exponent as lambda). No-op when off (lr stays the constant cfg value)."""
        if self.lr_sched is None:
            return
        self._model_lr_now = self.lr_sched(self.step)
        for g in self.model_opt.param_groups:
            g["lr"] = self._model_lr_now

    def _apply_logstd_floor(self) -> None:
        """cf21 (PM 2026-06-15): set the policy's HARD log_std floor from rf — high (explore)
        at low return, relaxing linearly toward `lo` (commit) as return climbs. Derived
        purely from ret_ema (checkpointed), so resume is bitwise-exact. No-op when off."""
        if not self.ra_lsf_on:
            return
        lsm = self.ra_lsf_hi + (self.ra_lsf_lo - self.ra_lsf_hi) * self._reward_frac()
        self.policy.log_std_min = lsm
        if self.policy_ema is not None:
            self.policy_ema.log_std_min = lsm

    def _policy_reg(self, entropy, base_ent_coef):
        """Reward-adaptive policy regularization (PM 2026-06-15): returns (effective entropy
        coef, entropy-floor penalty tensor, effective actor grad-clip) from rf. Off ⇒
        (base_ent_coef, 0, actor_clip) — byte-identical."""
        if self.auto_alpha:        # A3: SAC auto-temperature — α=exp(log_alpha) IS the entropy
            alpha_loss = self.log_alpha * (entropy.detach() - self.alpha_target_H)  # coef; step
            self.alpha_opt.zero_grad(set_to_none=True)                              # log_alpha
            alpha_loss.backward(); self.alpha_opt.step()                            # so H→target
            rf = self._reward_frac()
            clip = (self.actor_clip * (self.ra_clip_min + (1.0 - self.ra_clip_min) * (1.0 - rf))
                    if self.ra_clip_on else self.actor_clip)
            return self.log_alpha.exp().detach(), entropy.new_zeros(()), clip
        rf = self._reward_frac()
        ent_coef = base_ent_coef * (1.0 - rf) if self.ra_anneal else base_ent_coef
        floor_pen = entropy.new_zeros(())
        if self.ra_floor_on:                       # penalize entropy below H*(rf)=h_high·(1-rf)
            gap = self.ra_floor_h * (1.0 - rf) - entropy   # >0 ⇒ entropy under the floor
            if self.ra_floor_shape == "sigmoid":   # bounded penalty; lift peaks at the target
                floor_pen = self.ra_floor_coef * torch.sigmoid(self.ra_floor_beta * gap)
            else:                                  # relu: constant lift everywhere below
                floor_pen = self.ra_floor_coef * torch.relu(gap)
        clip = self.actor_clip
        if self.ra_clip_on:                        # tighten the grad-clip as return rises
            clip = self.actor_clip * (self.ra_clip_min + (1.0 - self.ra_clip_min) * (1.0 - rf))
        return ent_coef, floor_pen, clip

    def _returns_and_scale(self, zs, rs, tau0, H, cfg_i, gamma, lam_ret):
        """Value-target bootstrap + advantage/lambda-returns + Dreamer-V3 return
        scaling. Pure extraction of the contiguous block from behaviour_update
        (primary path); preserves every statement and order verbatim. NO self.gen
        draws. Returns (returns, adv, norm); mutates self.ret_scale in place exactly
        as before. `zs` are the imagined latents (H+1, B, k)."""
        with torch.no_grad():
            flat = zs.reshape(-1, zs.shape[-1])
            tgt_tau = tau0.repeat(H + 1, 1) if tau0 is not None else None
            v_tgt = self.value_target(flat, tgt_tau).reshape(H + 1, -1)
            if self.double_value:   # A4: min(V1,V2) clipped-double-value bootstrap
                v_tgt = torch.minimum(v_tgt, self.value2_target(flat, tgt_tau).reshape(H + 1, -1))
        # advantage estimator: "lambda" (default, unchanged) | "gae" (Schulman 2016,
        # the PPO/A2C standard). Both share gamma/lambda_; GAE's value target
        # (adv + v) IS the lambda-return (pinned by test_returns_gae), so the value
        # regression below is identical — only the POLICY weighting changes.
        advantage = str(cfg_i.get("advantage", "lambda"))
        if advantage == "gae":
            adv, returns = gae_advantages(rs, v_tgt, gamma, lam_ret)  # (H, B) x2
        else:
            returns = lambda_returns(rs, v_tgt, gamma, lam_ret)       # (H, B)
            adv = None
        if self.return_clip > 0.0:   # cf4: hard-bound the λ-returns (and GAE advantage)
            with torch.no_grad():   # diagnostic: pre-clip over-threshold fraction (returns)
                self._last_return_clip_frac = float(
                    (returns.detach().abs() > self.return_clip).float().mean())
            returns = returns.clamp(-self.return_clip, self.return_clip)  # so a diverged
            if adv is not None:                                           # imagined rollout
                adv = adv.clamp(-self.return_clip, self.return_clip)      # can't NaN the loss

        # --- return normalization (Dreamer-V3): scale-invariant policy gradient
        with torch.no_grad():
            lo = torch.quantile(returns.detach().float(), 0.05)
            hi = torch.quantile(returns.detach().float(), 0.95)
            decay = cfg_i.get("ret_scale_decay", 0.99)
            span = float(hi - lo)
            if np.isfinite(span):  # NaN hygiene: don't poison the scale EMA
                self.ret_scale = decay * self.ret_scale + (1 - decay) * span
        norm = max(1.0, self.ret_scale)
        return returns, adv, norm

    def _imagine_rollout(self, z0, tau0, H):
        """Differentiable imagination on the primary (single-latent) path: roll the
        learned dynamics under the policy/planner, scoring per-step imagined reward.
        Pure extraction of the contiguous block from behaviour_update; every statement
        and order preserved verbatim. Draws ZERO self.gen (the primary rollout uses the
        policy's own sampler, NOT self.gen — the strict primary RNG invariant). Returns
        (zs, rs, logps, dis, pen_stats); zs is (H+1, B, k), rs/logps are (H, B)."""
        # --- differentiable imagination (gradients flow through T and R) ---
        # planner: emit the whole H-step plan from z0 up front (open-loop), then
        # roll T under it; the per-step policy samples closed-loop on z_k.
        plan_a, plan_logp = (self.planner.plan(z0, tau0) if self.use_planner
                             else (None, None))
        zs, rs, logps, dis, pens = [z0], [], [], [], []
        z = z0
        for k in range(H):
            if self.use_planner:
                a, logp = plan_a[k], plan_logp[k]
            else:
                a, logp = self.policy.sample(z, tau0)
            z = self.dynamics(z, a)
            zs.append(z)
            r_im, d = self._imagined_reward(zs[-2], a, tau0)
            if self.dyn_ensemble and self.ens_pessimism > 0.0:
                # epistemic discount (PETS/MBPO-style): distrust imagined reward
                # where the dynamics ensemble disagrees about the transition
                pen = self.ens_pessimism * self.dynamics.disagreement(zs[-2], a)
                r_im = r_im - pen
                pens.append(pen.detach())
            rs.append(r_im)
            dis.append(d)
            logps.append(logp)
        zs = torch.stack(zs)                      # (H+1, B, k)
        # campaign-1 instrumentation lesson: imagine/return_var conflates the
        # discount's own variance with model-exploitation — log the penalty
        # stream separately so the two are decomposable
        pen_stats = {}
        if pens:
            pen_t = torch.stack(pens)
            pen_stats = {"imagine/penalty_mean": pen_t.mean().item(),
                         "imagine/penalty_var": pen_t.var().item()}
        rs = smooth_rewards(torch.stack(rs), self.cfg.smoothing)  # (H, B)
        logps = torch.stack(logps)                # (H, B)
        return zs, rs, logps, dis, pen_stats

    def _policy_value_step(self, zs, returns, adv, norm, logps, ent_coef, z0, tau0, H, cfg_i):
        """Actor (policy/planner) + value optimizer step on the primary path
        (incl. the cf4 stabilizers: skip_nonfinite, value_clip, double_value, and
        the policy EMA). Pure extraction of the contiguous block from
        behaviour_update — every statement and order preserved verbatim. NO self.gen
        draws (the optimizer steps / EMA updates are deterministic; _policy_reg may
        step the alpha opt but draws no self.gen). Returns
        (entropy, pi_loss, v_loss, align_val, gnorm) for the metrics dict; mutates
        self._nonfinite_skips and the value/policy nets/targets exactly as before."""
        # --- policy: maximize normalized lambda-returns (or GAE advantages) +
        #     entropy (never curvature-penalized, R10)
        entropy = -logps.mean()
        pi_signal = adv if adv is not None else returns
        ent_coef_eff, floor_pen, clip_eff = self._policy_reg(entropy, ent_coef)
        pi_loss = -(pi_signal / norm).mean() - ent_coef_eff * entropy + floor_pen
        # Imagination-latent ALIGNMENT (arXiv 2507.16450-inspired stabilizer):
        # pull the rolled-out imagined latents back onto the ENCODER's manifold
        # by matching the per-dim mean/std of z0 (the real encoded latents).
        # Combats imagination drift — the long plan wandering off-distribution
        # where the reward/value readouts break (the transformer's
        # collapse-after-peak failure mode). Grad flows to the actor (and T) via
        # the imagined zs; the target stats are detached.
        align_val = 0.0
        if self.align_weight > 0.0:
            with torch.no_grad():
                real_mu, real_sd = z0.mean(0), z0.std(0)
            imag = zs[1:].reshape(-1, zs.shape[-1])
            align = ((imag.mean(0) - real_mu).pow(2).mean()
                     + (imag.std(0) - real_sd).pow(2).mean())
            pi_loss = pi_loss + self.align_weight * align
            align_val = align.item()
        pi_loss = pi_loss + self._policy_inertia_term()    # weight inertia (off by default)
        self.policy_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        actor = self.planner if self.use_planner else self.policy
        gnorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip_eff)
        if (not self.skip_nonfinite) or torch.isfinite(gnorm):   # cf4: don't poison θ_π
            self.policy_opt.step()
        else:
            self._nonfinite_skips += 1

        # --- value: regress to lambda-returns on detached latents ---
        flat = zs[:-1].detach().reshape(-1, zs.shape[-1])
        v_tau = tau0.repeat(H, 1) if tau0 is not None else None
        v = self.value(flat, v_tau).reshape(H, -1)
        v_loss = F.mse_loss(v, returns.detach())
        self.value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        if self.value_clip > 0.0 or self.skip_nonfinite:   # cf4: clip the value grad
            vnorm = torch.nn.utils.clip_grad_norm_(       # (was unclipped) + skip on NaN
                self.value.parameters(),
                self.value_clip if self.value_clip > 0.0 else float("inf"))
            if (not self.skip_nonfinite) or torch.isfinite(vnorm):
                self.value_opt.step()
            else:
                self._nonfinite_skips += 1
        else:
            # measure-only inf-norm (no-op clip, mirrors the model-loss path) so
            # the pre-clip value grad norm is ALWAYS available as a diagnostic
            vnorm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), float("inf"))
            self.value_opt.step()
        self._last_value_grad_norm = float(vnorm)   # pre-clip; reveals value_clip hits

        # --- EMA target value + (optional) EMA policy ---
        decay = cfg_i.get("value_target_decay", 0.98)
        with torch.no_grad():
            for pt, p in zip(self.value_target.parameters(), self.value.parameters()):
                pt.lerp_(p, 1.0 - decay)
        if self.double_value:   # A4: train + EMA the second value net on the same λ-targets
            v2 = self.value2(flat, v_tau).reshape(H, -1)
            v2_loss = F.mse_loss(v2, returns.detach())
            self.value2_opt.zero_grad(set_to_none=True); v2_loss.backward()
            v2n = torch.nn.utils.clip_grad_norm_(self.value2.parameters(),
                    self.value_clip if self.value_clip > 0.0 else float("inf"))
            if (not self.skip_nonfinite) or torch.isfinite(v2n):
                self.value2_opt.step()
            with torch.no_grad():
                for pt, pp in zip(self.value2_target.parameters(), self.value2.parameters()):
                    pt.lerp_(pp, 1.0 - decay)
        self._update_policy_ema()
        return entropy, pi_loss, v_loss, align_val, gnorm

    # ---------------- behaviour learning (Dreamer lambda-returns) ----------------
    def behaviour_update(self, z0: torch.Tensor, tau0: torch.Tensor | None = None) -> dict:
        if self.dual_latent:
            return self._behaviour_update_dual(z0, tau0)
        cfg_i = self.cfg.imagination
        gamma, lam_ret = cfg_i.gamma, cfg_i.get("lambda_", 0.95)
        H = self._imagination_horizon()
        ent_coef = cfg_i.get("entropy_coef", 3e-4)
        self._reward_clip_over = self._reward_clip_tot = 0.0   # reset diag accumulators

        zs, rs, logps, dis, pen_stats = self._imagine_rollout(z0, tau0, H)

        returns, adv, norm = self._returns_and_scale(zs, rs, tau0, H, cfg_i, gamma, lam_ret)

        entropy, pi_loss, v_loss, align_val, gnorm = self._policy_value_step(
            zs, returns, adv, norm, logps, ent_coef, z0, tau0, H, cfg_i)

        return {"loss/value": v_loss.item(), "loss/policy": pi_loss.item(),
                "policy/entropy": entropy.item(),
                "policy/ret_scale": self.ret_scale,
                "model/reward_disagreement": torch.stack(dis).mean().item(),
                "imagine/horizon": H,
                "imagine/return_mean": returns.mean().item(),
                "imagine/return_var": returns.var().item(),  # R15 diagnostic
                "imagine/align": align_val, "actor/grad_norm": float(gnorm),
                "stab/nonfinite_skips": self._nonfinite_skips,  # cf4: same diagnostic as the dual path
                "stab/value_grad_norm": self._last_value_grad_norm,   # cf4: pre-clip
                **({"stab/reward_clip_frac": self._reward_clip_over
                    / max(self._reward_clip_tot, 1.0)} if self.reward_clip > 0.0 else {}),
                **({"stab/return_clip_frac": self._last_return_clip_frac}
                   if self.return_clip > 0.0 else {}),
                **pen_stats}

    # ---------------- dual-latent path (model.dual_latent.enabled) ----------------
    def _model_update_dual(self, batch) -> dict:
        """Shared encoder z; dynamics latent d=D(z) fit by the operator in d-space;
        policy latent p=P(z) carries the reward head + the curvature penalty (R10/
        R16, now in p-coords). Twin mode also fits op_p (p-consistency) and the weak
        coupling L_couple. No spectral/auto-dose/dgate — a clean fresh arm."""
        import functools, math as _math
        obs, a, r, obs_next, tau, z, z_next_tgt, vae_terms, vae_metrics = self._encode_batch(batch)
        dl = self.dual
        d, p = dl.d_of(z), dl.p_of(z)
        with torch.no_grad():
            d_next, p_next = dl.d_of(z_next_tgt), dl.p_of(z_next_tgt)

        # dynamics fit, scored in d-space (shared: roll backbone z, require D(z')
        # predictable; twin: op_d predicts d')
        if dl.mode == "shared":
            dyn_loss = F.mse_loss(dl.d_of(dl.op(z, a)), d_next)
        else:
            dyn_loss = F.mse_loss(dl.op_d(d, a), d_next)

        # reward fit in p-coords
        r_target = self._reward_target(r)
        rew_loss = F.mse_loss(self.reward(p, a, tau), r_target)

        # isotropic curvature penalty on the reward, in p-coords (R10/R16; detached p).
        # dual_penalize_reward=false skips it entirely (and the hvp compute): in twin
        # mode the policy latent p is meant to be ROUGH/bumpy (it carries sharp reward/
        # value structure) while the DYNAMICS latent d is kept smooth by op_d's priors
        # — so the reward-curvature penalty can be the wrong tool on p (PM 2026-06-13).
        lam_t = max(self.lam(self.step) * self.rg_gate_now, self.lambda_min)  # hard floor
        if self.dual_penalize_reward:
            if self.cfg.penalty.get("form", "frobenius") == "laplacian_trace":
                penalty_fn = functools.partial(
                    laplacian_trace_penalty, clamp=self.cfg.penalty.get("clamp_trace", True))
            else:
                penalty_fn = hvp_penalty
            parts = [p.detach(), a]
            if tau is not None and self.cfg.penalty.get("include_task", True):
                parts.append(tau)
            x_pen = torch.cat(parts, dim=-1)
            if tau is not None and not self.cfg.penalty.get("include_task", True):
                fn = lambda x: self.reward.on_concat(torch.cat([x, tau.detach()], dim=-1))
            else:
                fn = self.reward.on_concat
            pen = penalty_fn(fn, x_pen, n_probes=self.cfg.penalty.n_probes, generator=self.gen)
            pen_val = pen.item()
            loss = dyn_loss + rew_loss + lam_t * pen
        else:
            pen_val, lam_t = 0.0, 0.0
            loss = dyn_loss + rew_loss
        if vae_terms is not None:
            loss = loss + vae_terms

        # operator structural priors (per operator) + spectral diagnostics. In twin
        # mode op_d (dynamics) uses op_w and op_p (policy) uses op_w_p — so the
        # dynamics space can be regularized SMOOTH while the policy space is left
        # ROUGH (e.g. op_w.w_smooth>0, op_w_p.w_smooth=0).
        op_metrics = {}
        if self.rad_anneal_tau > 0.0:
            op_metrics["op/radius_ceil"] = getattr(self, "_radius_ceil", self.rad_anneal_floor)
        # PHASED SVD (model.operator.struct_every): spectral_summary + structural_penalties both
        # call svdvals(A) — O(d^3), the dominant cost at large latent. Run them only every
        # struct_every-th update (=once per episode when set to model_updates_per_iter), caching
        # the diagnostics so they still log each iteration. The Stein/lyap lever below is
        # matmul-only and stays every-update. struct_every=1 ⇒ unchanged validated behaviour.
        if self.step % self.struct_every == 0:
            _cache = {}
            for i, op in enumerate(dl.operators()):
                tag = "" if dl.mode == "shared" else ("_d" if i == 0 else "_p")
                zin = z.detach() if dl.mode == "shared" else (
                    d.detach() if i == 0 else p.detach())
                w = self.op_w_p if (dl.mode == "twin" and i == 1) else self.op_w
                _cache |= {f"{kk}{tag}": vv for kk, vv in op.spectral_summary(zin).items()}
                if any(w.values()):
                    sp = op.structural_penalties(zin)
                    loss = loss + sum(w[kk] * sp[kk] for kk in w)
                    _cache |= {f"op/pen_{kk}{tag}": float(sp[kk].detach()) for kk in sp}
            self._op_metrics_cache = _cache
        op_metrics |= self._op_metrics_cache

        # Lyapunov/Stein consistency on op_d (model.dual_latent.lyap_weight>0): the
        # empirical d second moment G must be op_d's STATIONARY covariance given the
        # measured innovation — the discrete Stein equation G = A G Aᵀ + Q̂. Forces
        # op_d to be a faithful forward model on the ACTUAL latent geometry (term (c)
        # of docs/unified_spectral_loss.md), and complements svband: the contraction
        # (svband) plus the innovation Q̂ must reconstruct G, so op_d can be neither
        # frozen NOR inconsistent. Stable: only second moments + matmuls — no
        # eigendecomposition, no Lyapunov solve. Twin only (op_d rolls d-space).
        if self.lyap_w > 0.0 and dl.mode == "twin":
            A_d, _ = dl.op_d.operators(d)                  # (N,k,k) per-sample operator
            yhat = (A_d @ d.unsqueeze(-1)).squeeze(-1)     # autonomous prop A_d(d)·d (a=0)
            N_ = float(d.shape[0])
            G = d.t() @ d / N_                             # current d second moment (k,k)
            S_auto = yhat.t() @ yhat / N_                  # E[A d dᵀ Aᵀ] (state-dep-correct)
            R = d_next - dl.op_d(d, a)                     # full dynamics innovation (d_next detached)
            Q_hat = R.t() @ R / N_                         # innovation covariance
            if self.excite_enabled:                        # EMA of innovation RMS √(tr Q̂/k) = the excite drive scale
                q_rms = float((Q_hat.diagonal().clamp_min(0.0).sum() / Q_hat.shape[0]).sqrt().detach())
                if _math.isfinite(q_rms):
                    self.innov_ema = (q_rms if self.innov_ema is None
                                      else self.excite_decay * self.innov_ema + (1 - self.excite_decay) * q_rms)
            stein = (G - S_auto - Q_hat).pow(2).mean()     # ‖G − A G Aᵀ − Q̂‖² (per-entry)
            loss = loss + self.lyap_w * stein
            op_metrics["op/lyap_stein"] = float(stein.detach())

        # det(op_p) > 0 (model.dual_latent.detpos_weight>0): require the POLICY operator
        # to be invertible AND orientation-preserving — det A_p ≥ floor > 0. Keeps op_p
        # in GL⁺ (the identity component): no policy mode collapses to a singular
        # direction, no orientation flip, so the entropy exponent log det A_p stays
        # finite and the imagined policy rollout is a proper invertible flow. The
        # conservative-op_p counterpart to svband's dissipative-op_d (op_d contracts,
        # det<1, forgets; op_p stays non-singular, det>0, preserves control directions).
        # det is real even for the rotational op_p — complex eigenvalues pair up. Twin only.
        if self.detpos_w > 0.0 and dl.mode == "twin":
            A_p, _ = dl.op_p.operators(p)                  # (N,k,k) policy operator
            det_p = torch.linalg.det(A_p)                  # (N,) real
            detpos = torch.relu(self.detpos_floor - det_p).pow(2).mean()  # barrier det ≥ floor>0
            loss = loss + self.detpos_w * detpos
            op_metrics["op/detpos"] = float(detpos.detach())
            op_metrics["op/det_p_mean"] = float(det_p.detach().mean())
            op_metrics["op/det_p_negfrac"] = float((det_p.detach() <= 0).float().mean())

        # twin: ground op_p (p-consistency) + weak coupling of the two geometries
        dual_metrics = {}
        cpl = None
        if dl.mode == "twin":
            pcons = F.mse_loss(dl.op_p(p, a), p_next)
            loss = loss + self.pconsist_w * pcons
            dual_metrics["dual/p_consistency"] = pcons.item()
            if self.couple_w > 0 or self.frame_balance:
                cpl = dl.couple(d, p)
                if not self.frame_balance:   # cf7: balance bypasses the fixed weight
                    loss = loss + self.couple_w * cpl
                dual_metrics["dual/couple"] = cpl.item()
            with torch.no_grad():   # relative-phase drift: scale-free desync of the two
                cd, cp = dl.Wd(d), dl.Wp(p)   # sectors (0 = phase-locked, →1 slipping out).
                denom = 0.5 * (cd.norm(dim=-1) + cp.norm(dim=-1)) + 1e-6   # phase stiffness
                dual_metrics["dual/phase_drift"] = float(   # readout (couple_w raises it)
                    ((cd - cp).norm(dim=-1) / denom).mean())

        frame_metrics = {}
        dissip = None
        if self.frame_enabled:   # cf5 rank-2 reward⊥energy frame
            frame_term, frame_metrics = self._rank2_frame(z, d, p, a, tau)
            loss = loss + frame_term
            if (self.frame_w_dissip > 0.0 or self.frame_balance) and self.energy is not None:
                from ..regularization.rank2_frame import dissipativity_penalty
                dissip = dissipativity_penalty(self.energy, d, self.dual.d_of(z_next_tgt), r)
                if not self.frame_balance:   # cf6 dissipativity (fixed weight)
                    loss = loss + self.frame_w_dissip * dissip
                frame_metrics["frame/dissip_resid"] = float(dissip.detach())
        # cf7 equilibrium coupling: hold alignment (couple) and energy (dissipativity) at
        # equal influence so neither outweighs the other (replaces their fixed weights)
        if self.frame_balance and cpl is not None and dissip is not None:
            bal_term, bal_m = self._balance_align_energy(cpl, dissip)
            loss = loss + bal_term
            frame_metrics |= bal_m

        self._model_step(loss, pen_val)   # zero_grad/backward/clip/[skip_nonfinite]/step/ema/step++/pen_ema

        # GATED input-salience snapshot (default-OFF, dual path): same self-contained
        # ∂reward/∂obs pass (reward reads p in dual mode; the helper handles that). No
        # training tensor/opt/RNG touched. The Gram snapshot rides _representation_readouts below.
        if self.salience_every > 0 and (self.step % self.salience_every == 0):
            self._snapshot_input_salience(obs, a, tau)

        z_std_now = z.detach().std(0).mean().item()
        if self.excite_enabled and _math.isfinite(z_std_now):   # EMA anchor for the excitation gate
            self.z_std_ema = (z_std_now if self.z_std_ema is None
                              else self.excite_decay * self.z_std_ema + (1 - self.excite_decay) * z_std_now)
        out = {"loss/dyn": dyn_loss.item(), "loss/reward": rew_loss.item(),
               "penalty/value": pen_val, "penalty/lambda": lam_t,
               "penalty/return_gate": self.rg_gate_now, "optim/model_lr": self._model_lr_now,
               "loss/total": loss.item(), "step": self.step,
               "latent/z_std": z_std_now,
               **op_metrics, **dual_metrics, **frame_metrics,
               **self._representation_readouts(z)}
        if vae_metrics is not None:
            out |= vae_metrics
        return out

    def _imagine_rollout_dual(self, z0, tau0, H):
        """Differentiable imagination on the DUAL-LATENT path: roll in the policy
        latent p (shared: roll backbone z, read p=P(z); twin: roll p with op_p),
        with the gated stochastic excitation Q-drive. Pure extraction of the
        contiguous block from _behaviour_update_dual — every statement and order
        preserved verbatim, INCLUDING the self.gen draws and their guards.
        RNG-CRITICAL: the only self.gen draws on the behaviour side live here —
        (1) ONE Bernoulli(excite_p) gate draw before the loop, then (2) one randn
        per rollout step inside the loop when the gate is open. Both stay in their
        original positions/order relative to each other and to everything else (no
        other self.gen draw exists in this method). Returns
        (ps, rs, logps, excite_now, noise_std); ps is (H+1, B, p_dim)."""
        dl = self.dual
        z = z0
        p = dl.p_of(z0)
        # discrete excitation gate: ONE Bernoulli(excite_p) draw per update, OPEN only at the
        # operating point (ema z_std within band of the anchor); when open, inject Q-scaled process
        # noise at every rollout step (parametric drive of the |λ|≈1 oscillator). self.gen = checkpointed RNG.
        excite_now = False
        if (self.excite_enabled and self.innov_ema is not None and self.z_std_ema is not None
                and abs(self.z_std_ema - self.excite_zstd_anchor) <= self.excite_zstd_band):
            excite_now = bool(torch.bernoulli(
                torch.full((), self.excite_p, device=self.device), generator=self.gen).item())
        noise_std = self.excite_scale * self.innov_ema if excite_now else 0.0
        ps, rs, logps = [p], [], []
        for _ in range(H):
            a, logp = self.policy.sample(p, tau0)
            if dl.mode == "shared":
                z = dl.op(z, a)
                p = dl.p_of(z)
            else:
                p = dl.op_p(p, a)
            if excite_now:                            # additive detached noise ⇒ gradients still flow through p
                p = p + noise_std * torch.randn(p.shape, generator=self.gen, device=self.device)
            ps.append(p)
            r_im, _ = self._imagined_reward(ps[-2], a, tau0)
            rs.append(r_im)
            logps.append(logp)
        ps = torch.stack(ps)                          # (H+1, B, p_dim)
        rs = smooth_rewards(torch.stack(rs), self.cfg.smoothing)   # (H, B)
        logps = torch.stack(logps)                    # (H, B)
        return ps, rs, logps, excite_now, noise_std

    def _policy_value_step_dual(self, ps, returns, adv, norm, logps, ent_coef, tau0, H, cfg_i):
        """Actor (policy) + value optimizer step on the DUAL-LATENT path (incl. the
        cf4 stabilizers and the policy EMA). Imagination/alignment are in p-space
        (ps), and the actor is always self.policy (the planner is never used in dual
        mode). Pure extraction of the contiguous block from _behaviour_update_dual —
        every statement and order preserved verbatim. NO self.gen draws. Returns
        (entropy, pi_loss, v_loss, align_val, gnorm); mutates self._nonfinite_skips
        and the value/policy nets/targets exactly as before."""
        entropy = -logps.mean()
        pi_signal = adv if adv is not None else returns
        ent_coef_eff, floor_pen, clip_eff = self._policy_reg(entropy, ent_coef)
        pi_loss = -(pi_signal / norm).mean() - ent_coef_eff * entropy + floor_pen
        align_val = 0.0
        if self.align_weight > 0.0:                   # 2507.16450 stabilizer, in p-space
            with torch.no_grad():
                real_mu, real_sd = ps[0].mean(0), ps[0].std(0)
            imag = ps[1:].reshape(-1, ps.shape[-1])
            align = ((imag.mean(0) - real_mu).pow(2).mean()
                     + (imag.std(0) - real_sd).pow(2).mean())
            pi_loss = pi_loss + self.align_weight * align
            align_val = align.item()
        pi_loss = pi_loss + self._policy_inertia_term()    # weight inertia (off by default)
        self.policy_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_eff)
        if (not self.skip_nonfinite) or torch.isfinite(gnorm):   # cf4: don't poison θ_π
            self.policy_opt.step()
        else:
            self._nonfinite_skips += 1

        flat = ps[:-1].detach().reshape(-1, ps.shape[-1])
        v_tau = tau0.repeat(H, 1) if tau0 is not None else None
        v = self.value(flat, v_tau).reshape(H, -1)
        v_loss = F.mse_loss(v, returns.detach())
        self.value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        if self.value_clip > 0.0 or self.skip_nonfinite:   # cf4: clip the value grad
            vnorm = torch.nn.utils.clip_grad_norm_(       # (was unclipped) + skip on NaN
                self.value.parameters(),
                self.value_clip if self.value_clip > 0.0 else float("inf"))
            if (not self.skip_nonfinite) or torch.isfinite(vnorm):
                self.value_opt.step()
            else:
                self._nonfinite_skips += 1
        else:
            # measure-only inf-norm (no-op clip) so the pre-clip value grad norm
            # is ALWAYS available as a diagnostic
            vnorm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), float("inf"))
            self.value_opt.step()
        self._last_value_grad_norm = float(vnorm)   # pre-clip; reveals value_clip hits
        decay = cfg_i.get("value_target_decay", 0.98)
        with torch.no_grad():
            for pt, pp in zip(self.value_target.parameters(), self.value.parameters()):
                pt.lerp_(pp, 1.0 - decay)
        if self.double_value:   # A4: train + EMA the second value net on the same λ-targets
            v2 = self.value2(flat, v_tau).reshape(H, -1)
            v2_loss = F.mse_loss(v2, returns.detach())
            self.value2_opt.zero_grad(set_to_none=True); v2_loss.backward()
            v2n = torch.nn.utils.clip_grad_norm_(self.value2.parameters(),
                    self.value_clip if self.value_clip > 0.0 else float("inf"))
            if (not self.skip_nonfinite) or torch.isfinite(v2n):
                self.value2_opt.step()
            with torch.no_grad():
                for pt, pp in zip(self.value2_target.parameters(), self.value2.parameters()):
                    pt.lerp_(pp, 1.0 - decay)
        self._update_policy_ema()
        return entropy, pi_loss, v_loss, align_val, gnorm

    def _behaviour_update_dual(self, z0: torch.Tensor, tau0=None) -> dict:
        """Imagine in the POLICY latent p (shared: roll backbone z, read p=P(z);
        twin: roll p with op_p), score reward/value on p, train via Dreamer
        λ-returns (R10: the actor is never curvature-penalized)."""
        cfg_i = self.cfg.imagination
        gamma, lam_ret = cfg_i.gamma, cfg_i.get("lambda_", 0.95)
        H = self._imagination_horizon()
        ent_coef = cfg_i.get("entropy_coef", 3e-4)
        self._reward_clip_over = self._reward_clip_tot = 0.0   # reset diag accumulators

        ps, rs, logps, excite_now, noise_std = self._imagine_rollout_dual(z0, tau0, H)

        returns, adv, norm = self._returns_and_scale(ps, rs, tau0, H, cfg_i, gamma, lam_ret)

        entropy, pi_loss, v_loss, align_val, gnorm = self._policy_value_step_dual(
            ps, returns, adv, norm, logps, ent_coef, tau0, H, cfg_i)

        return {"loss/value": v_loss.item(), "loss/policy": pi_loss.item(),
                "policy/entropy": entropy.item(), "policy/ret_scale": self.ret_scale,
                "imagine/horizon": H, "imagine/return_mean": returns.mean().item(),
                "imagine/return_var": returns.var().item(),
                "imagine/align": align_val, "actor/grad_norm": float(gnorm),
                "stab/nonfinite_skips": self._nonfinite_skips,
                "stab/value_grad_norm": self._last_value_grad_norm,   # cf4: pre-clip
                **({"stab/reward_clip_frac": self._reward_clip_over
                    / max(self._reward_clip_tot, 1.0)} if self.reward_clip > 0.0 else {}),
                **({"stab/return_clip_frac": self._last_return_clip_frac}
                   if self.return_clip > 0.0 else {}),
                "excite/gate": float(excite_now), "excite/noise_std": float(noise_std)}

    def _policy_inertia_term(self):
        """Soft trust-region anchor inertia*‖θ_π − θ_π^ema‖² (0.0 when off). Pulls
        the live policy toward its slow EMA so it resists big jumps toward
        exploiting transient model errors (two-timescale stabilizer)."""
        if self.policy_inertia <= 0.0 or self.policy_ema is None:
            return 0.0
        return self.policy_inertia * sum(
            (p - pe.detach()).pow(2).sum()
            for p, pe in zip(self.policy.parameters(), self.policy_ema.parameters()))

    def _update_policy_ema(self):
        """Polyak-update the slow policy EMA (no-op when off). Deterministic — no
        new RNG, so bitwise resume holds once policy_ema is checkpointed."""
        if self.policy_ema is None:
            return
        with torch.no_grad():
            for pe, p in zip(self.policy_ema.parameters(), self.policy.parameters()):
                pe.lerp_(p, 1.0 - self.policy_ema_decay)

    def observe_return(self, ep_return: float) -> None:
        """Feed the latest ACTUAL eval return to the return-gate (penalty.return_gate).
        Maps the return EMA to gate ∈ [floor,1] about the midpoint `rg_mid` (default
        0 = the do-nothing↔running boundary on HalfCheetah). `rg_shape` selects the
        gate's response curve (all full-range, smooth, slew-limited):
          - 'quadratic' (default): MONOTONE convex — λ high for return ≤ mid, relaxes
            to floor along a parabola for return > mid.
          - 'cuberoot': MONOTONE concave — relaxes λ faster (the gate-curvature axis).
          - 'sigmoid': MONOTONE 1-σ.
          - 'bump': PEAKED — MAX λ at mid (the "phase transition"), released
            quadratically both above and below (penalize hardest where unstable).
        `rg_slew` caps the per-eval change so a collapse can't spike the gate.
        No-op when disabled. ret_ema + rg_gate_now are checkpointed."""
        import math as _m
        if not _m.isfinite(ep_return) or not (self.rg_enabled or self._reward_adapt_on):
            return
        # ret_ema feeds BOTH the gate and the reward-adaptive policy knobs, so update it
        # whenever either is on; the gate math below is skipped when the gate is off.
        self.ret_ema = (ep_return if self.ret_ema is None
                        else self.rg_decay * self.ret_ema
                        + (1 - self.rg_decay) * ep_return)
        self._apply_logstd_floor()      # cf21: relax the hard variance bound as return climbs
        if not self.rg_enabled:
            return
        z = (self.ret_ema - self.rg_mid) / max(self.rg_scale, 1e-6)
        u = max(min(z, 1.0), -1.0)               # signed distance from mid, clipped
        if self.rg_shape == "sigmoid":
            relax = 1.0 - 1.0 / (1.0 + _m.exp(-max(min(z, 60.0), -60.0)))   # 1-σ
        elif self.rg_shape == "bump":
            # PEAKED at mid: MAX lambda at the "phase transition" (return ≈ mid),
            # released QUADRATICALLY both above AND below (relax = 1 - u²). Penalize
            # hardest where the policy is unstable (the do-nothing↔running boundary);
            # free it when clearly good (high +) or clearly failed (deep −). Not a
            # monotone gate — heuristic, in the vein of the trace penalty (PM).
            relax = 1.0 - u * u
        elif self.rg_shape in ("leaky_relu", "leaky"):
            # THRESHOLD gate (piecewise-linear): hold λ ~rigid below mid (release rises
            # only at the leak slope), then release SHARPLY (linear) above mid. Knee at
            # the return midpoint. relax = 1 - release; release(frac): 0→leak below mid,
            # leak→1 above mid. The sharpest monotone "hold then let go at the knee" gate.
            frac = (u + 1.0) / 2.0                    # 0 at mid-scale, 0.5 at mid, 1 at mid+scale
            lk = max(min(self.rg_leak, 1.0), 0.0)
            release = (lk + (1.0 - lk) * 2.0 * (frac - 0.5) if frac >= 0.5
                       else lk * 2.0 * frac)
            relax = 1.0 - release
        else:
            # FULL-RANGE power gate over the band [mid-scale, mid+scale]:
            # relax = 1 - frac^p, frac = (u+1)/2 ∈ [0,1]. lambda varies smoothly with
            # return on BOTH sides of mid (gate=1 below mid-scale, floor above
            # mid+scale). The exponent p is the GATE CURVATURE:
            #   quadratic (p=2):  convex — holds lambda high, relaxes late.
            #   cuberoot  (p=1/3): concave — relaxes lambda fast, then flattens.
            frac = (u + 1.0) / 2.0
            p = (1.0 / 3.0) if self.rg_shape == "cuberoot" else 2.0
            relax = 1.0 - frac ** p
        target = self.rg_floor + (1.0 - self.rg_floor) * relax
        lo, hi = self.rg_gate_now - self.rg_slew, self.rg_gate_now + self.rg_slew
        self.rg_gate_now = min(max(target, lo), hi)             # slew-rate limited
        if self.rg_ratchet:
            # cf19 (PM 2026-06-15): the horizon-ratchet logic applied to lambda. Once the
            # policy makes genuine progress (return EMA crosses mid), LOCK the gate's running
            # minimum: lambda's relaxation can deepen but never re-tighten, so a transient
            # return dip can't re-regularize a converging policy. Stability at peak convergence.
            if not self._rg_ratchet_on and self.ret_ema > self.rg_mid:
                self._rg_ratchet_on = True
            if self._rg_ratchet_on:
                self._rg_ratchet_min = min(self._rg_ratchet_min, self.rg_gate_now)
                self.rg_gate_now = self._rg_ratchet_min

    @torch.no_grad()
    def act(self, z: torch.Tensor, tau: torch.Tensor | None = None,
            deterministic: bool = False) -> torch.Tensor:
        """Execution action selection (B, act): the planner's receding-horizon
        first planned action, or a policy sample. The one seam env-facing code
        (collection, eval) uses, so the actor swap is invisible to train.py.
        deterministic=True returns the tanh-Gaussian MEAN (no action noise) — used
        only by the det-eval metric; collection/gates keep the stochastic path."""
        if self.use_planner:
            return self.planner.act(z, tau)
        # policy inertia: act/collect with the slow EMA policy when enabled
        actor = (self.policy_ema if (self.policy_ema is not None and self.policy_ema_act)
                 else self.policy)
        zin = self.dual.p_of(z) if self.dual_latent else z   # policy reads the policy latent p
        if deterministic:
            return actor.mean_action(zin, tau)
        return actor.sample(zin, tau)[0]

    # ---------------- checkpoint protocol ----------------
    def state_dict(self):
        sd = {"encoder": self.encoder.state_dict(), "ema": self.ema.state_dict(),
                "dynamics": self.dynamics.state_dict(), "reward": self.reward.state_dict(),
                "policy": self.policy.state_dict(), "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                **({"value2": self.value2.state_dict(),                 # A4 (off ⇒ absent)
                    "value2_target": self.value2_target.state_dict(),
                    "value2_opt": self.value2_opt.state_dict()} if self.double_value else {}),
                **({"log_alpha": self.log_alpha.detach().cpu(),         # A3 (off ⇒ absent)
                    "alpha_opt": self.alpha_opt.state_dict()} if self.auto_alpha else {}),
                **({"policy_ema": self.policy_ema.state_dict()}
                   if self.policy_ema is not None else {}),
                "model_opt": self.model_opt.state_dict(),
                "policy_opt": self.policy_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(), "step": self.step,
                "ret_scale": self.ret_scale,
                **({"planner": self.planner.state_dict()} if self.use_planner else {}),
                **({"dual": self.dual.state_dict()} if self.dual_latent else {}),
                **({"energy": self.energy.state_dict()} if self.energy is not None else {}),
                # data-driven symexp clamp bound (bitwise resume)
                "symlog_bound": self.symlog_bound,
                # cf4 non-finite-grad skip counter (diagnostic; bitwise resume)
                "nonfinite_skips": self._nonfinite_skips,
                # cf7 equilibrium-balance running magnitudes (bitwise resume)
                "bal_ema_align": self._bal_ema_align, "bal_ema_energy": self._bal_ema_energy,
                # auto-dose: computed lam0 + warmup accumulators (bitwise resume)
                "lam0_sched": self.lam.lam0, "lam0_auto": self.lam0_auto,
                "ad_count": self.ad_count, "ad_fit_sum": self.ad_fit_sum,
                "ad_pen_sum": self.ad_pen_sum,
                # adaptive-horizon certificate state
                "pen_ema": self.pen_ema, "pen_peak": self.pen_peak,
                "ah_ratchet_floor": self._ah_ratchet_floor,
                "ah_ratchet_on": self._ah_ratchet_on,
                # disagreement-gate EMA state (bitwise resume)
                "dis_ema": self.dis_ema, "dis_peak": self.dis_peak,
                "spec_head_dis": self._spec_head_dis,
                # return-gate state (bitwise resume)
                "ret_ema": self.ret_ema, "rg_gate_now": self.rg_gate_now,
                "rg_ratchet_on": self._rg_ratchet_on, "rg_ratchet_min": self._rg_ratchet_min,
                # gated-excitation EMA state (bitwise resume)
                "z_std_ema": self.z_std_ema, "innov_ema": self.innov_ema,
                "hutchinson_gen": self.gen.get_state()}
        if self.spec_enabled:
            # spectral reward: per-head (W, b, c) + the rolling cache + refit
            # counters — everything needed for bitwise resume of the closed-
            # form path (W/b are seed-derived but saved anyway: cheap insurance)
            sd["spectral"] = {
                "heads": [{"W": h.W.cpu(), "b": h.b.cpu(), "c": h.c.cpu(),
                           **({"W_base": h.W_base.cpu(),
                               "log_s": h.log_s.detach().cpu()}
                              if h.learn_scales else {})}
                          for h in self.spec_heads],
                "cache_x": self.spec_cache_x, "cache_y": self.spec_cache_y,
                "since_refit": self.spec_since_refit, "refits": self.spec_refits,
                # SNR-band RNG + the per-head Wiener-weight EMA: load-bearing for
                # bitwise resume (the SNR generator drives the split-half permutation
                # consumed on EVERY refit, in poly AND snr modes; the EMA carries the
                # smoothed band weights that define the reward heads in snr mode).
                # Without these, a resumed spectral run draws a different split and
                # restarts the EMA from scratch — a silent reward discontinuity.
                "snr_gen": self.spec_snr_gen.get_state(),
                "snr_ema": [t.cpu() if t is not None else None
                            for t in self.spec_snr_ema]}
            sd["spec_sigma"] = self.spec_sigma          # auto: calibrated ladder
            sd["spec_sigma_star"] = self.spec_sigma_star
            if self.spec_sigma == "learned":
                sd["spec_sigma_opt"] = self.spec_sigma_opt.state_dict()
        return sd

    def load_state_dict(self, sd):
        self.encoder.load_state_dict(sd["encoder"]); self.ema.load_state_dict(sd["ema"])
        self.dynamics.load_state_dict(sd["dynamics"]); self.reward.load_state_dict(sd["reward"])
        self.policy.load_state_dict(sd["policy"]); self.value.load_state_dict(sd["value"])
        if self.use_planner and "planner" in sd:
            self.planner.load_state_dict(sd["planner"])
        if self.dual_latent and "dual" in sd:
            self.dual.load_state_dict(sd["dual"])
        if self.energy is not None and "energy" in sd:
            self.energy.load_state_dict(sd["energy"])
        if "value_target" in sd:
            self.value_target.load_state_dict(sd["value_target"])
        if self.double_value and "value2" in sd:                       # A4
            self.value2.load_state_dict(sd["value2"])
            self.value2_target.load_state_dict(sd["value2_target"])
            self.value2_opt.load_state_dict(sd["value2_opt"])
        if self.auto_alpha and "log_alpha" in sd:                      # A3
            with torch.no_grad():
                self.log_alpha.copy_(sd["log_alpha"].to(self.log_alpha.device))
            self.alpha_opt.load_state_dict(sd["alpha_opt"])
        if self.policy_ema is not None and "policy_ema" in sd:
            self.policy_ema.load_state_dict(sd["policy_ema"])
        self.model_opt.load_state_dict(sd["model_opt"])
        self.policy_opt.load_state_dict(sd["policy_opt"])
        self.value_opt.load_state_dict(sd["value_opt"])
        self.step = sd["step"]
        self.ret_scale = sd.get("ret_scale", 1.0)
        self.symlog_bound = sd.get("symlog_bound", 1.0)
        self._nonfinite_skips = sd.get("nonfinite_skips", 0)
        self._bal_ema_align = sd.get("bal_ema_align", None)
        self._bal_ema_energy = sd.get("bal_ema_energy", None)
        if "lam0_sched" in sd:  # auto-dose may have rewritten the schedule's lam0
            self.lam.lam0 = sd["lam0_sched"]
        self.lam0_auto = sd.get("lam0_auto", None)
        self.dis_ema = sd.get("dis_ema", None)
        self.dis_peak = sd.get("dis_peak", 0.0)
        self._spec_head_dis = sd.get("spec_head_dis", None)
        self.ret_ema = sd.get("ret_ema", None)
        self._apply_logstd_floor()      # cf21: restore the variance bound from the resumed rf
        self.rg_gate_now = sd.get("rg_gate_now", 1.0)
        self._rg_ratchet_on = sd.get("rg_ratchet_on", False)
        self._rg_ratchet_min = sd.get("rg_ratchet_min", 1.0)
        self.ad_count = sd.get("ad_count", 0)
        self.ad_fit_sum = sd.get("ad_fit_sum", 0.0)
        self.ad_pen_sum = sd.get("ad_pen_sum", 0.0)
        self.pen_ema = sd.get("pen_ema", None)
        self.pen_peak = sd.get("pen_peak", 0.0)
        self._ah_ratchet_floor = sd.get("ah_ratchet_floor", 0)
        self._ah_ratchet_on = sd.get("ah_ratchet_on", False)
        self.z_std_ema = sd.get("z_std_ema", None)
        self.innov_ema = sd.get("innov_ema", None)
        if "hutchinson_gen" in sd:  # probe RNG must resume too (bitwise resume)
            self.gen.set_state(sd["hutchinson_gen"])
        if self.spec_enabled and "spectral" in sd:
            sp = sd["spectral"]
            if len(self.spec_heads) != len(sp["heads"]):
                # sigma_w=auto resuming before this instance calibrated:
                # rebuild placeholder heads; W/b/c (the calibrated basis)
                # are restored from the checkpoint below
                self.spec_heads = self._build_spec_heads(1.0)
                self.spec_sigma = sd.get("spec_sigma", self.spec_sigma)
                self.spec_sigma_star = sd.get("spec_sigma_star",
                                              self.spec_sigma_star)
            for head, hs in zip(self.spec_heads, sp["heads"]):
                head.W = hs["W"].to(head.device)
                head.b = hs["b"].to(head.device)
                head.c = hs["c"].to(head.device)
                head.w2 = head.W.pow(2).sum(-1)
                head.w4 = head.w2.pow(2)
                if head.learn_scales and "log_s" in hs:
                    head.W_base = hs["W_base"].to(head.device)
                    with torch.no_grad():
                        head.log_s.copy_(hs["log_s"].to(head.device))
            if self.spec_sigma == "learned" and "spec_sigma_opt" in sd:
                self.spec_sigma_opt.load_state_dict(sd["spec_sigma_opt"])
            self.spec_cache_x = sp["cache_x"].clone()
            self.spec_cache_y = sp["cache_y"].clone()
            self.spec_since_refit = sp["since_refit"]
            self.spec_refits = sp["refits"]
            # restore the SNR generator + Wiener-weight EMA (guarded: checkpoints
            # written before this fix lack the keys → keep the freshly-seeded state)
            if "snr_gen" in sp:
                self.spec_snr_gen.set_state(sp["snr_gen"])
            if "snr_ema" in sp:
                self.spec_snr_ema = [t.to(self.device) if t is not None else None
                                     for t in sp["snr_ema"]]


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
            a = trainer.act(z)
        a_np = a.cpu().numpy()
        obs_next, r, term, trunc, _ = env.step(a_np)
        for i in range(num_envs):
            if not autoreset[i]:  # skip the fake post-done boundary transition
                buffer.add(obs[i], a_np[i], float(r[i]), obs_next[i])
        autoreset = np.logical_or(term, trunc)
        obs = obs_next
        taken += num_envs
    return obs, autoreset, taken
