"""Parametric Cartesian trajectories for the end-effector to track.

Every trajectory exposes:
- `target(t)`     — 3D position at time `t` (seconds).
- `lookahead(t, n, dt)` — stacked future positions, shape (n, 3).
- `phase_sin_cos(t)`    — (sin, cos) of the trajectory phase. Exposed to the
  policy so it is stateless w.r.t. which trajectory it is tracking.
"""

from __future__ import annotations

from typing import Any

from kinesis.trajectories.base import Trajectory
from kinesis.trajectories.circle import CircleTrajectory
from kinesis.trajectories.figure8_3d import Figure8_3DTrajectory


def build_trajectory(kind: str, center_xyz, **params: Any) -> Trajectory:
    """Factory used by the env to instantiate a trajectory from config."""
    if kind == "circle":
        return CircleTrajectory(
            center=center_xyz,
            radius=float(params["radius_m"]),
            period_s=float(params["period_s"]),
        )
    if kind == "figure8_3d":
        return Figure8_3DTrajectory(
            center=center_xyz,
            amp_x_m=float(params["amp_x_m"]),
            amp_y_m=float(params["amp_y_m"]),
            amp_z_m=float(params["amp_z_m"]),
            period_s=float(params["period_s"]),
        )
    raise ValueError(f"unknown trajectory kind: {kind!r}")


__all__ = [
    "Trajectory",
    "CircleTrajectory",
    "Figure8_3DTrajectory",
    "build_trajectory",
]
