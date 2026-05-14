import numpy as np
import pytest

from kinesis.envs.panda_track import N_ARM_JOINTS, PandaTrackConfig, PandaTrackEnv


@pytest.fixture(scope="module")
def env() -> PandaTrackEnv:
    return PandaTrackEnv(config=PandaTrackConfig(), seed=0)


def test_observation_shape_and_dtype(env: PandaTrackEnv) -> None:
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert info == {}


def test_action_space(env: PandaTrackEnv) -> None:
    assert env.action_space.shape == (N_ARM_JOINTS,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_random_rollout_finite(env: PandaTrackEnv) -> None:
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(123)
    for _ in range(100):
        a = rng.uniform(-1.0, 1.0, size=N_ARM_JOINTS).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(a)
        assert np.all(np.isfinite(obs)), "obs contains NaN/Inf"
        assert np.isfinite(reward)
        assert "ee_error_m" in info and np.isfinite(info["ee_error_m"])


def test_reset_places_ee_near_target_center(env: PandaTrackEnv) -> None:
    env.reset(seed=0)
    ee = env._ee_pos()
    center = np.asarray(env.cfg.trajectory_center_xyz)
    assert np.linalg.norm(ee - center) < 0.10, (
        f"EE at {ee} is too far from trajectory center {center}; "
        "home_qpos may be wrong."
    )


def test_reward_inband_dominates_when_on_target(env: PandaTrackEnv) -> None:
    zero7 = np.zeros(7)
    on_target = np.array([0.5, 0.0, 0.4])
    r, terms = env._reward(
        ee=on_target,
        target=on_target,
        action=zero7,
        prev_action=zero7,
        qdot=zero7,
    )
    assert terms["r_track"] == 0.0
    assert terms["r_action_rate"] == 0.0
    assert terms["r_qdot"] == 0.0
    assert terms["r_inband"] == env.cfg.w_inband
    assert r == env.cfg.w_inband


def test_reward_strongly_negative_when_far(env: PandaTrackEnv) -> None:
    zero7 = np.zeros(7)
    target = np.array([0.5, 0.0, 0.4])
    far_ee = target + np.array([0.5, 0.0, 0.0])  # 50 cm off
    r, terms = env._reward(
        ee=far_ee,
        target=target,
        action=zero7,
        prev_action=zero7,
        qdot=zero7,
    )
    # 0.5 m error → tracking penalty = -10 * 0.25 = -2.5
    assert terms["r_track"] == pytest.approx(-2.5)
    assert terms["r_inband"] == 0.0
    assert r < -2.0


def test_reward_penalizes_action_rate(env: PandaTrackEnv) -> None:
    on_target = np.array([0.5, 0.0, 0.4])
    action = np.ones(7)
    prev = np.zeros(7)
    _, terms = env._reward(
        ee=on_target,
        target=on_target,
        action=action,
        prev_action=prev,
        qdot=np.zeros(7),
    )
    # ||1-0||² summed over 7 dims = 7 → -0.1 * 7 = -0.7
    assert terms["r_action_rate"] == pytest.approx(-0.7)


def test_step_respects_joint_limits(env: PandaTrackEnv) -> None:
    env.reset(seed=2)
    # Push hard against +action for many steps; must remain within joint range.
    a = np.ones(N_ARM_JOINTS, dtype=np.float32)
    # MuJoCo's joint limit constraints can show ~1e-3 rad of solver slop
    # when the controller is driving hard into the limit; 5e-3 (0.3°) is a
    # safe physical tolerance.
    tol = 5e-3
    for _ in range(200):
        env.step(a)
        q = env.data.qpos[:N_ARM_JOINTS]
        assert np.all(q <= env._joint_hi + tol)
        assert np.all(q >= env._joint_lo - tol)
