"""Lambda schedules (R12): (t0/(t0+t))^(1/3) anneal, step-anneal, cosine, constant,
and sin2chirp — a positive oscillating profile under the theory envelope."""
from __future__ import annotations

import math


class LambdaSchedule:
    """lambda(t) profiles. State is just t; checkpoint-friendly."""

    def __init__(self, kind: str = "cuberoot", lam0: float = 1e-3,
                 t0: float = 10_000.0, floor: float = 0.0,
                 step_at: float = 0.5, step_factor: float = 0.1,
                 total_steps: int | None = None,
                 period0: float = 20_000.0, period_end: float = 2_000.0,
                 period2: float = 10_000.0):
        self.kind, self.lam0, self.t0, self.floor = kind, lam0, t0, floor
        self.step_at, self.step_factor = step_at, step_factor
        self.total_steps = total_steps
        self.period0, self.period_end = period0, period_end
        self.period2 = period2  # sincos: second oscillator (beat = 1/|1/p0-1/p2|)

    def __call__(self, t: int) -> float:
        if self.kind == "constant":
            lam = self.lam0
        elif self.kind == "cuberoot":          # R12 theory profile
            lam = self.lam0 * (self.t0 / (self.t0 + t)) ** (1.0 / 3.0)
        elif self.kind == "step":              # strong, then release
            assert self.total_steps, "step schedule needs total_steps"
            lam = self.lam0 * (self.step_factor if t >= self.step_at * self.total_steps else 1.0)
        elif self.kind == "sin2chirp":
            # lam0 * envelope(t) * sin^2(phi(t)): strictly non-negative; a
            # pow-free envelope decays the amplitude while a linear chirp raises
            # the oscillation frequency — slow clamp/release cycles early
            # (stability), faster + smaller cycles late (models are better, the
            # reward gets periodic curvature 'breathing' instead of a constant
            # squeeze). Envelope: linear ramp to the floor over total_steps
            # (rational t0/(t0+t) fallback if total_steps unset — also pow-free).
            f0 = 1.0 / self.period0
            if self.total_steps:
                frac = min(t / self.total_steps, 1.0)
                env = 1.0 - frac
                f1 = 1.0 / self.period_end
                phase = 2 * math.pi * t * (f0 + 0.5 * (f1 - f0) * frac)
            else:
                env = self.t0 / (self.t0 + t)
                phase = 2 * math.pi * f0 * t
            lam = self.lam0 * env * math.sin(phase) ** 2
        elif self.kind == "sincos":
            # Two-oscillator interference: lam0 * env * ((sin w1 t + cos w2 t)/2)^2.
            # Different periods => beats — constructive interference (full-strength
            # clamp) alternating with destructive phase cancellation (deep nulls,
            # held off exact zero by the floor). Beat period = 1/|1/p0 - 1/p2|;
            # same pow-free envelope as sin2chirp.
            if self.total_steps:
                env = max(1.0 - min(t / self.total_steps, 1.0), 0.0)
            else:
                env = self.t0 / (self.t0 + t)
            s = math.sin(2 * math.pi * t / self.period0)
            c = math.cos(2 * math.pi * t / self.period2)
            # normalize by 4: with INDEPENDENT phases, full constructive
            # interference reaches |sin + cos| = 2 (that's the beat peak), so
            # (s+c)^2 <= 4 and the schedule tops out exactly at the envelope;
            # destructive cancellation drops to the floor
            lam = self.lam0 * env * 0.25 * (s + c) ** 2
        elif self.kind == "cosine":
            assert self.total_steps, "cosine schedule needs total_steps"
            frac = min(t / self.total_steps, 1.0)
            lam = self.floor + 0.5 * (self.lam0 - self.floor) * (1 + math.cos(math.pi * frac))
        else:
            raise ValueError(f"unknown schedule kind: {self.kind}")
        return max(lam, self.floor)
