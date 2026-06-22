"""operator_sdre — State-Dependent Riccati Equation (SDRE) control on the
OperatorDynamics local linearization (Çimen 2008, SDRE survey; Anderson & Moore
1990, optimal LQ methods; Kalman 1960b, the discrete Riccati equation).

The operator model z' = A(z) z + B(z) a is ALREADY a state-dependent linear
plant: at any latent z it hands back the exact pointwise (A, B). SDRE exploits
this — instead of linearizing a nonlinear f(z, a) by a Jacobian and incurring a
linearization residual, the (A(z), B(z)) ARE the local linear system, so the
LQR computed from them is exact for that state (the SDC factorization is unique
and exact here, not an approximation). The controller is therefore: at each z,
read (A, B) from the operator, solve the per-state discrete LQR, and apply the
first feedback law a = -K(z) (z - z_ref) — a receding-horizon / frozen-operator
regulator that re-solves the Riccati equation every step as the operator field
varies over z.

Per-state batched: every sample in the batch carries its OWN (A, B) and thus
its own gain K, so the backward Riccati recursion runs as batched matmuls /
batched linear solves over the leading N dimension. Pure torch, no RNG, no
global state — deterministic given the inputs (the resume-bitwise discipline,
matching cem.py). The operator object owns all model access; this module imports
no dynamics class and stays a standalone unit, taking any object exposing
`.operators(z) -> (A, B)`.
"""
from __future__ import annotations

import torch
from torch import Tensor


class OperatorSDRE:
    """State-Dependent Riccati Equation regulator over an OperatorDynamics field.

    horizon: number of backward Riccati steps for the finite-horizon recursion
        (the practical SDRE solve — a long-but-finite horizon converges to the
        steady DARE gain on a stabilizable pair).
    q_weight / r_weight: default state / control costs (Q = q_weight·I_k,
        R = r_weight·I_m) used when the caller passes no explicit Q / R.
    riccati_iters: cap on the fixed-point DARE iteration; the recursion stops
        early once P stops moving (convergence), so this bounds work on slow
        systems without forcing extra steps on fast ones. The effective number
        of backward sweeps is max(horizon, riccati_iters): `horizon` is the
        nominal finite-horizon depth, `riccati_iters` the convergence safety cap.
    """

    RIDGE = 1e-6           # tiny diagonal added to (R + BᵀPB) for a stable solve
    CONV_TOL = 1e-9        # P-fixed-point tolerance for early stop (DARE convergence)

    def __init__(self, horizon: int = 15, q_weight: float = 1.0, r_weight: float = 1.0,
                 riccati_iters: int = 50):
        if horizon < 1:
            raise ValueError("horizon (%d) must be >= 1" % horizon)
        if riccati_iters < 1:
            raise ValueError("riccati_iters (%d) must be >= 1" % riccati_iters)
        self.horizon = int(horizon)
        self.q_weight = float(q_weight)
        self.r_weight = float(r_weight)
        self.riccati_iters = int(riccati_iters)

    @torch.no_grad()
    def gain(self, A: Tensor, B: Tensor, Q: "Tensor | None" = None,
             R: "Tensor | None" = None) -> Tensor:
        """Per-sample discrete finite-horizon LQR gain by backward Riccati.

        A: [N, k, k], B: [N, k, m] (the pointwise operator at each state).
        Q: [k, k] or [N, k, k] (state cost, default q_weight·I_k).
        R: [m, m] or [N, m, m] (control cost, default r_weight·I_m).
        Returns the steady feedback gain K: [N, m, k] for the law a = -K z.

        Recursion (discrete, Kalman 1960b / Anderson & Moore 1990):
            P_T = Q
            K_t = (R + Bᵀ P B)⁻¹ Bᵀ P A
            P   = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
        Iterated max(horizon, riccati_iters) times (with early stop on a P
        fixed point) and the converged K returned. The inverse is never formed:
        (R + BᵀPB) X = BᵀPA is solved for X via torch.linalg.solve, X = K.
        """
        if A.dim() != 3 or B.dim() != 3:
            raise ValueError("A must be [N,k,k] and B [N,k,m]; got %r, %r"
                             % (tuple(A.shape), tuple(B.shape)))
        N, k, k2 = A.shape
        if k != k2:
            raise ValueError("A must be square [N,k,k]; got %r" % (tuple(A.shape),))
        if B.shape[0] != N or B.shape[1] != k:
            raise ValueError("B must be [N,k,m] matching A's [N,k,*]; got %r vs %r"
                             % (tuple(B.shape), tuple(A.shape)))
        m = B.shape[2]
        dev, dt = A.device, A.dtype

        Q = self._cost(Q, self.q_weight, k, N, dev, dt)              # [N,k,k]
        R = self._cost(R, self.r_weight, m, N, dev, dt)              # [N,m,m]
        Bt = B.transpose(-1, -2)                                     # [N,m,k]
        ridge = self.RIDGE * torch.eye(m, device=dev, dtype=dt)     # [m,m] -> broadcast

        P = Q.clone()                                                # P_T = Q, [N,k,k]
        K = torch.zeros(N, m, k, device=dev, dtype=dt)
        steps = max(self.horizon, self.riccati_iters)
        for _ in range(steps):
            BtP = Bt @ P                                             # [N,m,k]
            S = R + BtP @ B + ridge                                  # [N,m,m] = R+BᵀPB (+ridge)
            BtPA = BtP @ A                                           # [N,m,k] = BᵀPA
            K = torch.linalg.solve(S, BtPA)                          # [N,m,k]: S K = BᵀPA
            # P_new = Q + Aᵀ P A − Aᵀ P B K   (= Q + AᵀPA − AᵀPB(R+BᵀPB)⁻¹BᵀPA)
            AtP = A.transpose(-1, -2) @ P                            # [N,k,k] = AᵀP
            P_new = Q + AtP @ A - (AtP @ B) @ K                      # [N,k,k]
            P_new = 0.5 * (P_new + P_new.transpose(-1, -2))          # symmetrize (numerics)
            if torch.max(torch.abs(P_new - P)).item() < self.CONV_TOL:
                P = P_new
                break                                                # DARE fixed point reached
            P = P_new
        return K                                                     # [N,m,k]

    @torch.no_grad()
    def act(self, z: Tensor, operator, z_ref: "Tensor | None" = None,
            Q: "Tensor | None" = None, R: "Tensor | None" = None) -> Tensor:
        """Regulator action a = -K(z) (z - z_ref), clamped to [-1, 1].

        z: [N, k] latent batch. operator: any object with
        `.operators(z) -> (A, B)` (e.g. OperatorDynamics). z_ref: [N, k] or
        [k] target latent (default 0 — drive z to the origin). Returns a: [N, m].
        """
        if z.dim() != 2:
            raise ValueError("z must be [N,k]; got %r" % (tuple(z.shape),))
        A, B = operator.operators(z)                                # [N,k,k], [N,k,m]
        K = self.gain(A, B, Q, R)                                   # [N,m,k]
        ref = torch.zeros_like(z) if z_ref is None else z_ref
        err = (z - ref).unsqueeze(-1)                               # [N,k,1]
        a = -(K @ err).squeeze(-1)                                  # [N,m]
        return a.clamp(-1.0, 1.0)

    @staticmethod
    def _cost(M: "Tensor | None", weight: float, dim: int, N: int,
              dev: torch.device, dt: torch.dtype) -> Tensor:
        """Broadcast a cost matrix to [N, dim, dim] (default weight·I_dim)."""
        eye = torch.eye(dim, device=dev, dtype=dt)
        if M is None:
            return (weight * eye).expand(N, dim, dim)
        M = M.to(device=dev, dtype=dt)
        if M.dim() == 2:
            return M.expand(N, dim, dim)
        return M                                                     # already [N,dim,dim]
