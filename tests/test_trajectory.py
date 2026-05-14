import numpy as np

from kinesis.trajectories import CircleTrajectory, Figure8_3DTrajectory, build_trajectory


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
