"""Trajectory protocol — the minimum surface the env relies on."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Trajectory(Protocol):
    period_s: float

    def target(self, t: float) -> np.ndarray: ...

    def lookahead(self, t: float, n: int, dt: float) -> np.ndarray: ...

    def phase_sin_cos(self, t: float) -> np.ndarray: ...

    def orientation(self, t: float) -> np.ndarray:
        """Target rotation matrix at time `t`. (3, 3) array.

        Concrete trajectories that don't define a curve-specific orientation
        target should return the constant `R_DESIRED` ("hand pointing down").
        """
        ...

    def orientation_lookahead(self, t: float, n: int, dt: float) -> np.ndarray:
        """Stacked future rotations, shape (n, 3, 3)."""
        ...
