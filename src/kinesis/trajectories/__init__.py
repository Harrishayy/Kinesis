"""Parametric Cartesian trajectories for the end-effector to track.

Every trajectory exposes:
- `target(t)`     — 3D position at time `t` (seconds).
- `lookahead(t, n, dt)` — stacked future positions, shape (n, 3).
- `phase_sin_cos(t)`    — (sin, cos) of the trajectory phase. Exposed to the
  policy so it is stateless w.r.t. which trajectory it is tracking.
- `orientation(t)`      — target rotation matrix at `t`. Default is the
  constant "hand pointing down" pose; subclasses override for curve-specific
  orientation tracking.
- `orientation_lookahead(t, n, dt)` — stacked future rotations, shape (n, 3, 3).
"""

from __future__ import annotations

from typing import Any

from kinesis.trajectories.base import Trajectory
from kinesis.trajectories.circle import CircleTrajectory
from kinesis.trajectories.viviani import VivianiTrajectory


def build_trajectory(kind: str, center_xyz, **params: Any) -> Trajectory:
    """Factory used by the env to instantiate a trajectory from config."""
    if kind == "circle":
        return CircleTrajectory(
            center=center_xyz,
            radius=float(params["radius_m"]),
            period_s=float(params["period_s"]),
        )
    if kind == "viviani":
        return VivianiTrajectory(
            center=center_xyz,
            sphere_radius_m=float(params["sphere_radius_m"]),
            period_s=float(params["period_s"]),
        )
    raise ValueError(f"unknown trajectory kind: {kind!r}")


__all__ = [
    "Trajectory",
    "CircleTrajectory",
    "VivianiTrajectory",
    "build_trajectory",
]
