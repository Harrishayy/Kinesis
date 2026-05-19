"""Tests for the SO(3) helpers in `kinesis.orientation.so3`."""

from __future__ import annotations

import numpy as np

from kinesis.orientation import (
    R_DESIRED,
    axis_angle_to_R,
    exp_so3,
    geodesic_angle,
    log_so3,
    R_to_6d,
)


def _is_so3(R: np.ndarray, atol: float = 1e-9) -> bool:
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
    )


def test_exp_identity():
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3))


def test_exp_z_pi_is_180_rotation():
    R = exp_so3(np.array([0.0, 0.0, np.pi]))
    expected = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(R, expected, atol=1e-9)


def test_log_exp_round_trip_random():
    rng = np.random.default_rng(0)
    for _ in range(32):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        # Sample angle in (0, π) — log is single-valued away from the boundary.
        angle = rng.uniform(1e-3, np.pi - 1e-3)
        omega = angle * axis
        R = exp_so3(omega)
        assert _is_so3(R)
        omega_back = log_so3(R)
        # Same axis (up to sign) and same magnitude.
        assert np.isclose(np.linalg.norm(omega_back), angle, atol=1e-9)
        # Reconstruct R and compare — robust to axis-sign ambiguity at θ→π.
        assert np.allclose(exp_so3(omega_back), R, atol=1e-9)


def test_log_near_identity_small_angle():
    omega = np.array([1e-8, 2e-8, -3e-8])
    R = exp_so3(omega)
    omega_back = log_so3(R)
    assert np.allclose(omega_back, omega, atol=1e-12)


def test_log_at_pi_recovers_axis():
    for axis in [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    ]:
        R = exp_so3(np.pi * axis)
        omega_back = log_so3(R)
        assert np.isclose(np.linalg.norm(omega_back), np.pi, atol=1e-6)
        assert np.allclose(exp_so3(omega_back), R, atol=1e-6)


def test_geodesic_angle_identity_is_zero():
    assert np.isclose(geodesic_angle(np.eye(3), np.eye(3)), 0.0)


def test_geodesic_angle_pi():
    R_pi = exp_so3(np.array([0.0, 0.0, np.pi]))
    assert np.isclose(geodesic_angle(np.eye(3), R_pi), np.pi, atol=1e-6)


def test_geodesic_angle_matches_log():
    rng = np.random.default_rng(1)
    for _ in range(16):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, np.pi - 1e-3)
        R = exp_so3(angle * axis)
        assert np.isclose(geodesic_angle(np.eye(3), R), angle, atol=1e-9)


def test_geodesic_angle_frame_invariant():
    rng = np.random.default_rng(2)
    R_a = exp_so3(rng.normal(size=3))
    R_b = exp_so3(rng.normal(size=3))
    R_q = exp_so3(rng.normal(size=3))  # arbitrary frame
    d1 = geodesic_angle(R_a, R_b)
    d2 = geodesic_angle(R_q @ R_a, R_q @ R_b)
    assert np.isclose(d1, d2, atol=1e-9)


def test_axis_angle_to_R_normalises():
    R1 = axis_angle_to_R(np.array([0.0, 0.0, 2.0]), angle=np.pi / 2)
    R2 = axis_angle_to_R(np.array([0.0, 0.0, 1.0]), angle=np.pi / 2)
    assert np.allclose(R1, R2)


def test_R_to_6d_shape_and_round_trip():
    rng = np.random.default_rng(3)
    R = exp_so3(rng.normal(size=3))
    v = R_to_6d(R)
    assert v.shape == (6,)
    # The first two columns alone uniquely determine R (third = first × second).
    a = v[:3]
    b = v[3:]
    a_unit = a / np.linalg.norm(a)
    b_orth = b - (b @ a_unit) * a_unit
    b_unit = b_orth / np.linalg.norm(b_orth)
    c_unit = np.cross(a_unit, b_unit)
    R_back = np.column_stack([a_unit, b_unit, c_unit])
    assert np.allclose(R_back, R, atol=1e-9)


def test_R_desired_is_valid_so3():
    assert _is_so3(R_DESIRED)
