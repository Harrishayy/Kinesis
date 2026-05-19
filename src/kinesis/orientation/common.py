"""Single source of truth for the canonical "hand pointing down" orientation.

Columns are the hand frame's axes expressed in the world frame:
  hand-x = world +x  (along workspace forward direction)
  hand-y = world −y  (right-handed orthogonal)
  hand-z = world −z  (palm faces down at the table)

Used as the default fallback for `Trajectory.orientation(t)` and inside the
residual feedforward IK when orientation tracking is disabled.
"""

from __future__ import annotations

import numpy as np

R_DESIRED: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
R_DESIRED.flags.writeable = False
