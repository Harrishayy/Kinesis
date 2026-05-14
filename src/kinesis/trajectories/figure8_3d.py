"""3D figure-eight (tilted Lissajous) — engages all 7 arm joints.

Parametrised as:
    x(t) = cx + Ax * cos(ω t)        depth bob → joints 1-3 (shoulder/base)
    y(t) = cy + Ay * sin(2 ω t)      figure-eight lobe in the y axis
    z(t) = cz + Az * sin(ω t)        vertical sweep → joints 4-7 (elbow/wrist)

Projected on y-z this looks like an infinity sign; projected on x-z it is an
ellipse. The full curve is a figure-eight twisted through the workspace, so
no joint group can carry the whole tracking task on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Figure8_3DTrajectory:
    center: np.ndarray  # (3,)
    amp_x_m: float
    amp_y_m: float
    amp_z_m: float
    period_s: float

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        object.__setattr__(self, "center", c)

    def _angle(self, t: float | np.ndarray) -> np.ndarray:
        return 2.0 * np.pi * (np.asarray(t) % self.period_s) / self.period_s

    def target(self, t: float) -> np.ndarray:
        a = self._angle(t)
        return self.center + np.array(
            [
                self.amp_x_m * np.cos(a),
                self.amp_y_m * np.sin(2.0 * a),
                self.amp_z_m * np.sin(a),
            ],
            dtype=np.float64,
        )

    def lookahead(self, t: float, n: int, dt: float) -> np.ndarray:
        ts = t + dt * np.arange(1, n + 1)
        a = self._angle(ts)
        out = np.stack(
            [
                self.amp_x_m * np.cos(a),
                self.amp_y_m * np.sin(2.0 * a),
                self.amp_z_m * np.sin(a),
            ],
            axis=1,
        )
        return out + self.center

    def phase_sin_cos(self, t: float) -> np.ndarray:
        a = float(self._angle(t))
        return np.array([np.sin(a), np.cos(a)], dtype=np.float64)
