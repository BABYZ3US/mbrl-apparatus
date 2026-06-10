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

from ..models import (Encoder, EMAEncoder, VAEEncoder, CustomEncoder, AffineDynamics,
                      GaussianAffineDynamics, FullMLPDynamics, RewardModel,
                      Policy, ValueFn)
from ..models.ensemble import EnsembleAffineDynamics
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
        dyn_cls = {"affine": AffineDynamics, "gaussian": GaussianAffineDynamics,
                   "mlp": FullMLPDynamics}[_dyn_kind]
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
        else:
            self.dynamics = dyn_cls(k, action_dim, h, d).to(device)
        # epistemic discount on imagined reward: r -= coef * ensemble disagreement
        self.ens_pessimism = float((cfg.get("algo", {}) or {}).get("ensemble_pessimism", 0.0) or 0.0)
        self.symlog = bool(cfg.model.get("symlog_reward", False))
        self.reward = RewardModel(k, action_dim, h, d, task_dim=task_dim,
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
            spec_in = self.encoder.latent_dim + self.dynamics.m + self.task_dim
        return [SpectralReward(spec_in, n_features=self.spec_nf, sigma_w=sigma_w,
                               seed=int(self.cfg.seed) * 1000 + i,
                               device=str(self.device), learn_scales=learn)
                for i in range(self.spec_nheads)]
    def _spectral_band_weights(self, head, t: int) -> torch.Tensor:
        """Per-feature ridge weights at model-update time t:
        sum_d coefs[d] * lam(t + shifts[d]) * |w_j|^(2*degrees[d]).
        Per-degree time SHIFTS phase-shift the lambda schedule so different
        frequency bands clamp/release at different points of training."""
        theta = [c * self.lam(t + s) for c, s in zip(self.spec_coefs, self.spec_shifts)]
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

    # ---------------- model learning ----------------
    def model_update(self, batch) -> dict:
        if self.task_dim:
            obs, a, r, obs_next, tau = (x.to(self.device) for x in batch)
        else:
            obs, a, r, obs_next = (x.to(self.device) for x in batch)
            tau = None
        vae_terms = None
        if self.enc_vae:   # one forward: recon + KL + the z sample
            recon, kl, z = self.encoder.losses(obs)
            vae_terms = self.vae_recon_w * recon + self.vae_beta * kl
            vae_metrics = {"vae/recon": recon.item(), "vae/kl": kl.item()}
        else:
            z = self.encoder(obs)
        with torch.no_grad():
            z_next_tgt = self.ema(obs_next)

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
        r_target = symlog(r) if self.symlog else r
        if self.symlog:  # track the real-data symlog range for the imagination clamp
            batch_max = r_target.abs().max().item()
            if np.isfinite(batch_max):  # NaN hygiene: never poison the bound
                self.symlog_bound = max(self.symlog_bound, batch_max)
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
                    pred = torch.stack([h.predict(x_spec)
                                        for h in self.spec_heads]).mean(0)
                    rew_loss_val = F.mse_loss(pred, r_target).item()
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
                # spectral: the MLP reward fit doesn't exist — dose on dyn only
                self.ad_fit_sum += dyn_loss.item() + (
                    0.0 if self.spec_enabled else rew_loss.item())
                self.ad_pen_sum += pen_val
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

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 100.0)
        self.model_opt.step()
        self.ema.update(self.encoder)
        self.step += 1

        # penalty EMA + running peak — the adaptive-horizon certificate signal.
        # NaN hygiene: one non-finite penalty value must not poison the EMA
        # forever (a poisoned EMA crashed the horizon controller in the shiny run).
        import math as _math
        if _math.isfinite(pen_val):
            self.pen_ema = (pen_val if self.pen_ema is None
                            else self.ah_decay * self.pen_ema + (1 - self.ah_decay) * pen_val)
            self.pen_peak = max(self.pen_peak, self.pen_ema)

        out = {"loss/dyn": dyn_loss.item(), "loss/reward": rew_loss_val,
               "penalty/value": pen_val, "penalty/lambda": lam_t,
               "loss/total": loss.item(), "step": self.step, **spec_metrics,
               **dyn_calib}
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
            if self.dyn_ensemble and self.ens_pessimism > 0.0:
                # epistemic discount (PETS/MBPO-style): distrust imagined reward
                # where the dynamics ensemble disagrees about the transition
                r_im = r_im - self.ens_pessimism * self.dynamics.disagreement(zs[-2], a)
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

        # --- return normalization (Dreamer-V3): scale-invariant policy gradient
        with torch.no_grad():
            lo = torch.quantile(returns.detach().float(), 0.05)
            hi = torch.quantile(returns.detach().float(), 0.95)
            decay = cfg_i.get("ret_scale_decay", 0.99)
            span = float(hi - lo)
            if np.isfinite(span):  # NaN hygiene: don't poison the scale EMA
                self.ret_scale = decay * self.ret_scale + (1 - decay) * span
        norm = max(1.0, self.ret_scale)

        # --- policy: maximize normalized lambda-returns (or GAE advantages) +
        #     entropy (never curvature-penalized, R10)
        entropy = -logps.mean()
        pi_signal = adv if adv is not None else returns
        pi_loss = -(pi_signal / norm).mean() - ent_coef * entropy
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
        sd = {"encoder": self.encoder.state_dict(), "ema": self.ema.state_dict(),
                "dynamics": self.dynamics.state_dict(), "reward": self.reward.state_dict(),
                "policy": self.policy.state_dict(), "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "model_opt": self.model_opt.state_dict(),
                "policy_opt": self.policy_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(), "step": self.step,
                "ret_scale": self.ret_scale,
                # data-driven symexp clamp bound (bitwise resume)
                "symlog_bound": self.symlog_bound,
                # auto-dose: computed lam0 + warmup accumulators (bitwise resume)
                "lam0_sched": self.lam.lam0, "lam0_auto": self.lam0_auto,
                "ad_count": self.ad_count, "ad_fit_sum": self.ad_fit_sum,
                "ad_pen_sum": self.ad_pen_sum,
                # adaptive-horizon certificate state
                "pen_ema": self.pen_ema, "pen_peak": self.pen_peak,
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
                "since_refit": self.spec_since_refit, "refits": self.spec_refits}
            sd["spec_sigma"] = self.spec_sigma          # auto: calibrated ladder
            sd["spec_sigma_star"] = self.spec_sigma_star
            if self.spec_sigma == "learned":
                sd["spec_sigma_opt"] = self.spec_sigma_opt.state_dict()
        return sd

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
        self.symlog_bound = sd.get("symlog_bound", 1.0)
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
