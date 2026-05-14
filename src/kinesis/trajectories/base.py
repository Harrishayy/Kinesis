"""Trajectory protocol — the minimum surface the env relies on."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Trajectory(Protocol):
    period_s: float

    def target(self, t: float) -> np.ndarray: ...

    def lookahead(self, t: float, n: int, dt: float) -> np.ndarray: ...

    def phase_sin_cos(self, t: float) -> np.ndarray: ...
