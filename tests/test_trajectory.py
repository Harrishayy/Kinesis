import numpy as np

from kinesis.utils.trajectory import CircleTrajectory


def _traj() -> CircleTrajectory:
    return CircleTrajectory(center=np.array([0.5, 0.0, 0.4]), radius=0.15, period_s=4.0)


def test_target_lies_on_circle():
    traj = _traj()
    for t in np.linspace(0.0, 8.0, 17):
        p = traj.target(float(t))
        assert np.isclose(p[0], 0.5)
        r = np.linalg.norm(p[1:] - np.array([0.0, 0.4]))
        assert np.isclose(r, 0.15, atol=1e-9)


def test_period_closes_loop():
    traj = _traj()
    assert np.allclose(traj.target(0.0), traj.target(traj.period_s))


def test_lookahead_shape_and_consistency():
    traj = _traj()
    la = traj.lookahead(t=0.0, n=4, dt=0.1)
    assert la.shape == (4, 3)
    # The first lookahead element should equal target(0 + dt).
    assert np.allclose(la[0], traj.target(0.1))


def test_phase_unit_circle():
    traj = _traj()
    for t in np.linspace(0.0, 4.0, 9):
        s, c = traj.phase_sin_cos(float(t))
        assert np.isclose(s * s + c * c, 1.0)
