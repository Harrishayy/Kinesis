"""Viviani's curve — intersection of a sphere and a tangent cylinder.

Defined as the curve that lies simultaneously on:
- a sphere of radius R, and
- a cylinder of radius R/2 whose axis passes through the sphere's centre.

Parametrised here (centred on `center`) as

    x(τ) = cx + (R/2) cos t
    y(τ) = cy + (R/2) sin t
    z(τ) = cz +  R    sin(t/2)        with t = 4π · (τ mod T) / T

The xy projection is a circle of radius R/2; the xz projection is a figure-eight,
so the curve self-intersects in projection while remaining a single smooth loop
in 3D. The full closed curve is traced once over t ∈ [0, 4π]; we map one
period_s of wall time to that range so the policy sees a unique (sin, cos) phase
at every point along the curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VivianiTrajectory:
    center: np.ndarray  # (3,)
    sphere_radius_m: float
    period_s: float

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        object.__setattr__(self, "center", c)

    def _t(self, tau: float | np.ndarray) -> np.ndarray:
        return 4.0 * np.pi * (np.asarray(tau) % self.period_s) / self.period_s

    def target(self, t: float) -> np.ndarray:
        tt = self._t(t)
        R = self.sphere_radius_m
        return self.center + np.array(
            [0.5 * R * np.cos(tt), 0.5 * R * np.sin(tt), R * np.sin(0.5 * tt)],
            dtype=np.float64,
        )

    def lookahead(self, t: float, n: int, dt: float) -> np.ndarray:
        ts = t + dt * np.arange(1, n + 1)
        tt = self._t(ts)
        R = self.sphere_radius_m
        out = np.stack(
            [0.5 * R * np.cos(tt), 0.5 * R * np.sin(tt), R * np.sin(0.5 * tt)],
            axis=1,
        )
        return out + self.center

    def phase_sin_cos(self, t: float) -> np.ndarray:
        a = 2.0 * np.pi * (float(t) % self.period_s) / self.period_s
        return np.array([np.sin(a), np.cos(a)], dtype=np.float64)
