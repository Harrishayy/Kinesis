"""Gymnasium wrappers modelling the two uncertainty sources required by the brief:

- `ObsNoiseWrapper`: additive Gaussian noise on the measured end-effector
  position only. Targets and proprioception stay clean — the agent has noisy
  perception of where it is, not where it's going.
- `ActionDelayWrapper`: applies the action commanded `k` control steps ago.
  Initial queue is zeros so the first `k` env steps see no command.

The `ee_pos` field starts at offset 14 in the observation produced by
`PandaTrackEnv._obs()` (see that file for layout).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

EE_POS_OFFSET = 14  # q(7) + qdot(7)
EE_POS_DIM = 3


class ObsNoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to the measured EE position in obs."""

    def __init__(self, env: gym.Env, sigma_m: float, seed: int | None = None) -> None:
        super().__init__(env)
        if sigma_m < 0:
            raise ValueError(f"sigma_m must be non-negative, got {sigma_m}")
        self.sigma_m = float(sigma_m)
        self._rng = np.random.default_rng(seed)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.sigma_m == 0.0:
            return obs
        out = obs.copy()
        noise = self._rng.normal(0.0, self.sigma_m, size=EE_POS_DIM).astype(out.dtype)
        out[EE_POS_OFFSET : EE_POS_OFFSET + EE_POS_DIM] += noise
        return out

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return super().reset(seed=seed, options=options)


class ActionDelayWrapper(gym.Wrapper):
    """Delay the action by `delay_steps` control steps."""

    def __init__(self, env: gym.Env, delay_steps: int) -> None:
        super().__init__(env)
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self.delay_steps = int(delay_steps)
        self._queue: deque[np.ndarray] = deque()

    def _prefill(self) -> None:
        self._queue = deque(
            [np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)]
            * self.delay_steps,
            maxlen=max(self.delay_steps, 1),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._prefill()
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.delay_steps == 0:
            return self.env.step(action)
        applied = self._queue.popleft()
        self._queue.append(np.asarray(action, dtype=self.env.action_space.dtype))
        return self.env.step(applied)
