import numpy as np
import pytest

from kinesis.envs.panda_track import N_ARM_JOINTS, PandaTrackEnv
from kinesis.envs.wrappers import (
    EE_POS_DIM,
    EE_POS_OFFSET,
    ActionDelayWrapper,
    ObsNoiseWrapper,
)


@pytest.fixture()
def base_env() -> PandaTrackEnv:
    return PandaTrackEnv(seed=0)


def test_obs_noise_sigma_zero_is_identity(base_env: PandaTrackEnv) -> None:
    wrapped = ObsNoiseWrapper(base_env, sigma_m=0.0, seed=0)
    obs_w, _ = wrapped.reset(seed=0)
    base_env_2 = PandaTrackEnv(seed=0)
    obs_b, _ = base_env_2.reset(seed=0)
    assert np.array_equal(obs_w, obs_b)


def test_obs_noise_only_affects_ee_pos_field(base_env: PandaTrackEnv) -> None:
    wrapped = ObsNoiseWrapper(base_env, sigma_m=0.05, seed=42)
    obs_w, _ = wrapped.reset(seed=0)

    # Reproduce unwrapped obs from a fresh env with the same seed.
    base_env_2 = PandaTrackEnv(seed=0)
    obs_b, _ = base_env_2.reset(seed=0)

    diff = obs_w - obs_b
    # Only the EE-pos slice should differ.
    mask = np.ones_like(diff, dtype=bool)
    mask[EE_POS_OFFSET : EE_POS_OFFSET + EE_POS_DIM] = False
    assert np.all(diff[mask] == 0.0)
    assert not np.all(diff[EE_POS_OFFSET : EE_POS_OFFSET + EE_POS_DIM] == 0.0)


def test_action_delay_zero_is_identity(base_env: PandaTrackEnv) -> None:
    wrapped = ActionDelayWrapper(base_env, delay_steps=0)
    wrapped.reset(seed=0)
    a = np.full(N_ARM_JOINTS, 0.3, dtype=np.float32)
    # If delay is 0, the env should receive `a` directly — verify by checking
    # that the resulting prev_action matches.
    wrapped.step(a)
    assert np.allclose(base_env._prev_action, a, atol=1e-6)


def test_action_delay_queues_correctly() -> None:
    """A queue of length k means the underlying env sees `a_t` only at step t+k.

    For k=2: external [a1, a2, a3, a4] → internal [0, 0, a1, a2].
    """

    class _RecordEnv(PandaTrackEnv):
        def __init__(self) -> None:
            super().__init__(seed=0)
            self.recorded: list[np.ndarray] = []

        def step(self, action):  # type: ignore[override]
            self.recorded.append(np.asarray(action, dtype=np.float32).copy())
            return super().step(action)

    inner = _RecordEnv()
    wrapped = ActionDelayWrapper(inner, delay_steps=2)
    wrapped.reset(seed=0)

    inputs = [
        np.full(N_ARM_JOINTS, 0.1, dtype=np.float32),
        np.full(N_ARM_JOINTS, 0.2, dtype=np.float32),
        np.full(N_ARM_JOINTS, 0.3, dtype=np.float32),
        np.full(N_ARM_JOINTS, 0.4, dtype=np.float32),
    ]
    for a in inputs:
        wrapped.step(a)

    assert len(inner.recorded) == 4
    assert np.allclose(inner.recorded[0], 0.0)
    assert np.allclose(inner.recorded[1], 0.0)
    assert np.allclose(inner.recorded[2], 0.1)
    assert np.allclose(inner.recorded[3], 0.2)


def test_combined_wrappers_run_clean(base_env: PandaTrackEnv) -> None:
    env = ActionDelayWrapper(
        ObsNoiseWrapper(base_env, sigma_m=0.02, seed=0), delay_steps=2
    )
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = rng.uniform(-1.0, 1.0, size=N_ARM_JOINTS).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
