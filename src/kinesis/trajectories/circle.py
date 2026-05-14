"""Circle in the y-z plane at fixed x."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleTrajectory:
    center: np.ndarray  # (3,)
    radius: float
    period_s: float

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        object.__setattr__(self, "center", c)

    def _angle(self, t: float | np.ndarray) -> np.ndarray:
        return 2.0 * np.pi * (np.asarray(t) % self.period_s) / self.period_s

    def target(self, t: float) -> np.ndarray:
        a = self._angle(t)
        return self.center + np.array(
            [0.0, self.radius * np.cos(a), self.radius * np.sin(a)], dtype=np.float64
        )

    def lookahead(self, t: float, n: int, dt: float) -> np.ndarray:
        ts = t + dt * np.arange(1, n + 1)
        a = self._angle(ts)
        out = np.zeros((n, 3), dtype=np.float64)
        out[:, 0] = 0.0
        out[:, 1] = self.radius * np.cos(a)
        out[:, 2] = self.radius * np.sin(a)
        return out + self.center

    def phase_sin_cos(self, t: float) -> np.ndarray:
        a = float(self._angle(t))
        return np.array([np.sin(a), np.cos(a)], dtype=np.float64)
