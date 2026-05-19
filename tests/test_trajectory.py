import numpy as np

from kinesis.orientation import geodesic_angle
from kinesis.trajectories import (
    CircleTrajectory,
    VivianiTrajectory,
    build_trajectory,
)


def _is_so3(R: np.ndarray, atol: float = 1e-9) -> bool:
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
    )


def _circle() -> CircleTrajectory:
    return CircleTrajectory(center=np.array([0.5, 0.0, 0.4]), radius=0.15, period_s=4.0)


def test_target_lies_on_circle():
    traj = _circle()
    for t in np.linspace(0.0, 8.0, 17):
        p = traj.target(float(t))
        assert np.isclose(p[0], 0.5)
        r = np.linalg.norm(p[1:] - np.array([0.0, 0.4]))
        assert np.isclose(r, 0.15, atol=1e-9)


def test_period_closes_loop():
    traj = _circle()
    assert np.allclose(traj.target(0.0), traj.target(traj.period_s))


def test_lookahead_shape_and_consistency():
    traj = _circle()
    la = traj.lookahead(t=0.0, n=4, dt=0.1)
    assert la.shape == (4, 3)
    assert np.allclose(la[0], traj.target(0.1))


def test_phase_unit_circle():
    traj = _circle()
    for t in np.linspace(0.0, 4.0, 9):
        s, c = traj.phase_sin_cos(float(t))
        assert np.isclose(s * s + c * c, 1.0)


def _viviani() -> VivianiTrajectory:
    return VivianiTrajectory(
        center=np.array([0.5, 0.0, 0.4]),
        sphere_radius_m=0.12,
        period_s=4.0,
    )


def test_viviani_lies_on_sphere():
    # Sphere of radius R centred at (cx - R/2, cy, cz).
    traj = _viviani()
    R = traj.sphere_radius_m
    sphere_center = traj.center + np.array([-0.5 * R, 0.0, 0.0])
    for t in np.linspace(0.0, 2.0 * traj.period_s, 65):
        p = traj.target(float(t))
        assert np.isclose(np.linalg.norm(p - sphere_center), R, atol=1e-12)


def test_viviani_lies_on_cylinder():
    # Cylinder of radius R/2 with axis through (cx, cy) parallel to z.
    traj = _viviani()
    R = traj.sphere_radius_m
    for t in np.linspace(0.0, traj.period_s, 33):
        p = traj.target(float(t))
        r_xy = np.linalg.norm(p[:2] - traj.center[:2])
        assert np.isclose(r_xy, 0.5 * R, atol=1e-12)


def test_viviani_period_closes_loop():
    traj = _viviani()
    assert np.allclose(traj.target(0.0), traj.target(traj.period_s))


def test_viviani_self_intersects_in_xz_projection():
    # The defining figure-eight property: t and t + 2π give the same (x, z)
    # but opposite y. At τ = T/2 we are π out of phase from τ = 0 in xy and
    # at the same z (sin(t/2 + π) = -sin(t/2) but at τ=0 sin(0)=0 so z=0 too).
    traj = _viviani()
    p0 = traj.target(0.0)
    p_half = traj.target(traj.period_s / 2.0)
    # In the centred parametrisation the crossing happens at the +x end of the
    # cylinder: (cx + R/2, cy, cz). Both τ=0 and τ=T/2 land there.
    expected = traj.center + np.array([0.5 * traj.sphere_radius_m, 0.0, 0.0])
    assert np.allclose(p0, expected, atol=1e-12)
    assert np.allclose(p_half, expected, atol=1e-12)


def test_viviani_lookahead_matches_target():
    traj = _viviani()
    la = traj.lookahead(t=0.0, n=4, dt=0.1)
    assert la.shape == (4, 3)
    for i in range(4):
        assert np.allclose(la[i], traj.target(0.1 * (i + 1)))


def test_viviani_phase_unit_circle_and_unique_per_period():
    traj = _viviani()
    # Unit circle property.
    for t in np.linspace(0.0, traj.period_s, 9):
        s, c = traj.phase_sin_cos(float(t))
        assert np.isclose(s * s + c * c, 1.0)
    # Phase is unique over one period (matches τ ∈ [0, T) one-to-one with
    # angle ∈ [0, 2π)) so the policy can disambiguate the two figure-eight
    # leaves at the self-crossing.
    phases = [tuple(traj.phase_sin_cos(float(t))) for t in np.linspace(0.0, traj.period_s, 33)[:-1]]
    assert len(set(phases)) == len(phases)


def test_circle_orientation_is_valid_so3_everywhere():
    traj = _circle()
    for t in np.linspace(0.0, traj.period_s, 32, endpoint=False):
        R = traj.orientation(float(t))
        assert _is_so3(R, atol=1e-9)


def test_circle_orientation_hand_z_stays_palm_down():
    """Sinusoidal wrist-roll target keeps hand-z aligned with world −z; only
    rotation about hand-z varies."""
    traj = _circle()
    for t in np.linspace(0.0, traj.period_s, 16, endpoint=False):
        R = traj.orientation(float(t))
        assert np.allclose(R[:, 2], np.array([0.0, 0.0, -1.0]), atol=1e-9)


def test_circle_orientation_continuous():
    traj = _circle()
    eps = 1e-3
    for t in np.linspace(0.0, traj.period_s, 8, endpoint=False):
        R0 = traj.orientation(float(t))
        R1 = traj.orientation(float(t) + eps)
        # Geodesic angle between samples ε apart should be tiny.
        assert geodesic_angle(R0, R1) < 1e-2


def test_circle_orientation_amplitude_matches_60_degrees():
    """The wrist roll target has 60° amplitude — peak deviation from t=0 is
    at the sine peaks, where geodesic_angle should equal π/3."""
    traj = _circle()
    R0 = traj.orientation(0.0)
    R_quarter = traj.orientation(traj.period_s / 4.0)  # sin peak
    assert np.isclose(geodesic_angle(R0, R_quarter), np.pi / 3, atol=1e-6)
    # Trough.
    R_three_quarter = traj.orientation(3.0 * traj.period_s / 4.0)
    assert np.isclose(geodesic_angle(R0, R_three_quarter), np.pi / 3, atol=1e-6)


def test_circle_orientation_lookahead_matches_per_step():
    traj = _circle()
    la = traj.orientation_lookahead(t=0.0, n=4, dt=0.1)
    assert la.shape == (4, 3, 3)
    for i in range(4):
        assert np.allclose(la[i], traj.orientation(0.1 * (i + 1)), atol=1e-12)


def test_viviani_orientation_is_valid_so3_everywhere():
    traj = _viviani()
    for t in np.linspace(0.0, traj.period_s, 64, endpoint=False):
        R = traj.orientation(float(t))
        assert _is_so3(R, atol=1e-9)


def test_viviani_orientation_hand_z_stays_palm_down():
    traj = _viviani()
    for t in np.linspace(0.0, traj.period_s, 16, endpoint=False):
        R = traj.orientation(float(t))
        assert np.allclose(R[:, 2], np.array([0.0, 0.0, -1.0]), atol=1e-9)


def test_viviani_orientation_continuous():
    traj = _viviani()
    eps = 1e-3
    for t in np.linspace(0.0, traj.period_s, 16, endpoint=False):
        R0 = traj.orientation(float(t))
        R1 = traj.orientation(float(t) + eps)
        assert geodesic_angle(R0, R1) < 1e-2


def test_viviani_orientation_lookahead_matches_per_step():
    traj = _viviani()
    la = traj.orientation_lookahead(t=0.0, n=4, dt=0.1)
    assert la.shape == (4, 3, 3)
    for i in range(4):
        assert np.allclose(la[i], traj.orientation(0.1 * (i + 1)), atol=1e-12)


def test_build_trajectory_dispatches():
    c = build_trajectory("circle", np.array([0.5, 0.0, 0.4]), radius_m=0.15, period_s=4.0)
    assert isinstance(c, CircleTrajectory)
    v = build_trajectory(
        "viviani",
        np.array([0.5, 0.0, 0.4]),
        sphere_radius_m=0.12,
        period_s=4.0,
    )
    assert isinstance(v, VivianiTrajectory)
