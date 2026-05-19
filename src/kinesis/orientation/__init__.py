"""Orientation tracking support — kept in a separate package so the core env,
trajectories, and wrappers stay readable when `include_orientation=False`.

Public surface:
- `R_DESIRED`            — canonical "hand pointing down" rotation matrix.
- `geodesic_angle`       — SO(3) distance, in radians.
- `log_so3`, `exp_so3`   — Lie-algebra logarithm and exponential (axis-angle ↔ R).
- `R_to_6d`              — Zhou et al. 2019 6D continuous representation.
- `axis_angle_to_R`      — small-angle perturbation helper used by the noise wrapper.
- `look_at_R`            — build a rotation matrix that points hand-z from `pos`
  toward `anchor`, with a chosen "up" axis to fix the remaining DoF.
- `wrist_roll_R`         — R_DESIRED rotated around hand-z by A·sin(2π t/T).
  Feasible wrist-roll target for the Franka (stays well within joint-7's
  ±166° range for any A < π/2). Used as the orientation target by the
  `*_residual_orient` configs.
"""

from kinesis.orientation.common import R_DESIRED
from kinesis.orientation.so3 import (
    axis_angle_to_R,
    exp_so3,
    geodesic_angle,
    log_so3,
    look_at_R,
    R_to_6d,
    wrist_roll_R,
)

__all__ = [
    "R_DESIRED",
    "axis_angle_to_R",
    "exp_so3",
    "geodesic_angle",
    "log_so3",
    "look_at_R",
    "R_to_6d",
    "wrist_roll_R",
]
