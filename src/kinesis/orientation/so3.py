"""SO(3) helpers used by orientation tracking.

Everything here is pure NumPy: trajectories, env, wrappers, and eval all import
from this module so rotation maths lives in one place.
"""

from __future__ import annotations

import numpy as np

_EPS = 1.0e-9


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: axis-angle (3-vector) → rotation matrix.

    `omega` is the axis scaled by the angle (radians). Numerically stable for
    small `|omega|` via the Taylor expansion fallback.
    """
    w = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    K = _skew(w)
    if theta < 1.0e-6:
        # Second-order Taylor: I + K + 0.5 K² is sufficient for theta < 1e-6.
        return np.eye(3) + K + 0.5 * (K @ K)
    K_unit = K / theta
    return np.eye(3) + np.sin(theta) * K_unit + (1.0 - np.cos(theta)) * (K_unit @ K_unit)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Inverse of `exp_so3`: rotation matrix → axis-angle (3-vector).

    Numerically stable at θ ≈ 0 (Taylor fallback) and θ ≈ π (sym-matrix
    fallback recovering the axis from `B = (R + I) / 2`, since `B = a aᵀ`
    when `R = exp([πa]_×)`).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1.0e-6:
        # Near-identity: ω ≈ 0.5 (R − Rᵀ)ᵛ (vee of the skew part).
        skew = 0.5 * (R - R.T)
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    if np.pi - theta < 1.0e-6:
        # Near π: B = (R + I) / 2 ≈ a aᵀ. Take magnitudes from the diagonal,
        # then resolve signs by anchoring at the largest-magnitude component
        # and reading off-diagonals of B (= a[i] * a[j]).
        B = 0.5 * (R + np.eye(3))
        diag = np.clip(np.diag(B), 0.0, 1.0)
        axis = np.sqrt(diag)
        i_max = int(np.argmax(axis))
        for j in range(3):
            if j == i_max:
                continue
            if B[i_max, j] < 0.0:
                axis[j] = -axis[j]
        return theta * axis

    # Generic case.
    factor = theta / (2.0 * np.sin(theta))
    skew = factor * (R - R.T)
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic distance on SO(3): the angle of the relative rotation R_aᵀ R_b.

    `acos((trace − 1) / 2)`, clamped to `[−1, 1]` so float rounding overshoot
    doesn't return NaN. Boundary values produce 0 or π exactly.
    """
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R_a.T @ R_b))
    cos_theta = float(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
    return float(np.arccos(cos_theta))


def axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convenience: explicit axis (will be normalised) + angle (radians)."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        return np.eye(3)
    return exp_so3((angle / n) * a)


def R_to_6d(R: np.ndarray) -> np.ndarray:
    """6D continuous representation (Zhou et al. 2019).

    The first two columns of `R`, concatenated into a 6-vector. Avoids the
    discontinuities of Euler angles and the double-cover of quaternions —
    standard input encoding for rotations into a neural network.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]]).copy()


def wrist_roll_R(t: float, period_s: float, amplitude_rad: float) -> np.ndarray:
    """`R_DESIRED` rotated around hand-z by `A · sin(2π t / T)`.

    The hand-z axis stays aligned with world −z (palm down); only the
    rotation about hand-z varies sinusoidally. This is a *feasible*
    orientation-tracking target on the Franka — for any `A < π/2` the wrist
    roll stays inside joint 7's ±2.9 rad range — and it generalises to any
    "EE follows a curve while continuously rotating about its tool axis"
    task (polishing, painting, ultrasound, sanding).

    Geodesic-sweep range from `t = 0`: 0 → A → 0 → A → 0 over one period, so
    the metric `sweep_range_deg` reports `A` in degrees.
    """
    # Lazy import to avoid pulling common.py into the SO(3) math namespace
    # for callers that don't need orientation-target construction.
    from kinesis.orientation.common import R_DESIRED

    angle = amplitude_rad * np.sin(2.0 * np.pi * (float(t) % period_s) / period_s)
    c, s = np.cos(angle), np.sin(angle)
    R_z = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return R_DESIRED @ R_z


def look_at_R(pos: np.ndarray, anchor: np.ndarray, up_axis: np.ndarray) -> np.ndarray:
    """Rotation that points hand-z from `pos` toward `anchor`, with `up_axis`
    fixing the remaining DoF.

    hand-z = unit(anchor − pos)
    hand-y = unit(up_axis projected orthogonal to hand-z)  (Gram-Schmidt)
    hand-x = hand-y × hand-z

    Returns a constant when the look-at direction is degenerate (zero length)
    or parallel to `up_axis` — caller should pick a non-degenerate `up_axis`
    for the trajectory; circle/Viviani are validated in unit tests.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    up = np.asarray(up_axis, dtype=np.float64).reshape(3)

    z = anchor - pos
    z_norm = float(np.linalg.norm(z))
    if z_norm < _EPS:
        return np.eye(3)
    z = z / z_norm

    up_proj = up - (up @ z) * z
    up_norm = float(np.linalg.norm(up_proj))
    if up_norm < _EPS:
        # `up_axis` is parallel to the look-at direction; fall back to identity
        # rather than silently returning a malformed frame.
        return np.eye(3)
    y = up_proj / up_norm
    x = np.cross(y, z)

    return np.column_stack([x, y, z])
