import numpy as np

from kinesis.trajectories import (
    CircleTrajectory,
    Figure8_3DTrajectory,
    VivianiTrajectory,
    build_trajectory,
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


def _figure8() -> Figure8_3DTrajectory:
    return Figure8_3DTrajectory(
        center=np.array([0.5, 0.0, 0.4]),
        amp_x_m=0.10,
        amp_y_m=0.15,
        amp_z_m=0.10,
        period_s=4.0,
    )


def test_figure8_period_closes_loop():
    traj = _figure8()
    assert np.allclose(traj.target(0.0), traj.target(traj.period_s))


def test_figure8_crosses_center_in_y_at_quarter_period():
    # y = amp_y * sin(2ωt); at t = T/2 → 2ωt = 2π → sin = 0. The figure-eight
    # crosses the centre line at half-period.
    traj = _figure8()
    p = traj.target(traj.period_s / 2.0)
    assert np.isclose(p[1], traj.center[1], atol=1e-12)


def test_figure8_spans_all_three_axes():
    traj = _figure8()
    samples = np.stack([traj.target(float(t)) for t in np.linspace(0.0, traj.period_s, 64)])
    span = samples.max(axis=0) - samples.min(axis=0)
    assert span[0] > 0.15  # ~2 * amp_x
    assert span[1] > 0.25  # ~2 * amp_y
    assert span[2] > 0.15  # ~2 * amp_z


def test_figure8_lookahead_matches_target():
    traj = _figure8()
    la = traj.lookahead(t=0.0, n=3, dt=0.1)
    assert la.shape == (3, 3)
    for i in range(3):
        assert np.allclose(la[i], traj.target(0.1 * (i + 1)))


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


def test_build_trajectory_dispatches():
    c = build_trajectory("circle", np.array([0.5, 0.0, 0.4]), radius_m=0.15, period_s=4.0)
    assert isinstance(c, CircleTrajectory)
    f = build_trajectory(
        "figure8_3d",
        np.array([0.5, 0.0, 0.4]),
        amp_x_m=0.1,
        amp_y_m=0.15,
        amp_z_m=0.1,
        period_s=4.0,
    )
    assert isinstance(f, Figure8_3DTrajectory)
    v = build_trajectory(
        "viviani",
        np.array([0.5, 0.0, 0.4]),
        sphere_radius_m=0.12,
        period_s=4.0,
    )
    assert isinstance(v, VivianiTrajectory)
