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


class _PinkNoise:
    """Voss-McCartney 1/f noise generator (online, no FFT).

    Sums `n_octaves` independent zero-mean Gaussian sources updated at
    progressively halved rates: source i is refreshed every 2^i steps. With
    enough octaves the power spectrum is approximately 1/f across the
    (1/2^N, 0.5) sample-rate band, so the lowest frequency content has a
    period of 2^N samples (≈ 1.3 s at 50 Hz with N=6).
    """

    def __init__(
        self,
        dim: int,
        sigma: float,
        n_octaves: int = 6,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_octaves < 1:
            raise ValueError("n_octaves must be >= 1")
        self.dim = int(dim)
        self.sigma = float(sigma)
        self.n_octaves = int(n_octaves)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._per_source_sigma = self.sigma / np.sqrt(self.n_octaves)
        self._sources = self._rng.normal(
            0.0, self._per_source_sigma, size=(self.n_octaves, self.dim)
        )
        self._step = 0

    def sample(self) -> np.ndarray:
        self._step += 1
        for i in range(self.n_octaves):
            if self._step % (1 << i) == 0:
                self._sources[i] = self._rng.normal(0.0, self._per_source_sigma, size=self.dim)
        return self._sources.sum(axis=0)


class ObsNoiseWrapper(gym.ObservationWrapper):
    """Add noise to the measured EE position in obs. `color` switches between
    IID Gaussian ("white") and Voss-McCartney 1/f ("pink"). Pink noise has
    non-zero autocorrelation across the lookahead window so the policy can't
    defeat it with a single-step low-pass — it has to learn broadband rejection.
    """

    def __init__(
        self,
        env: gym.Env,
        sigma_m: float,
        seed: int | None = None,
        color: str = "white",
        n_octaves: int = 6,
    ) -> None:
        super().__init__(env)
        if sigma_m < 0:
            raise ValueError(f"sigma_m must be non-negative, got {sigma_m}")
        if color not in {"white", "pink"}:
            raise ValueError(f"color must be 'white' or 'pink', got {color!r}")
        self.sigma_m = float(sigma_m)
        self.color = color
        self.n_octaves = int(n_octaves)
        self._rng = np.random.default_rng(seed)
        self._pink: _PinkNoise | None = (
            _PinkNoise(EE_POS_DIM, self.sigma_m, n_octaves=self.n_octaves, rng=self._rng)
            if color == "pink"
            else None
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.sigma_m == 0.0:
            return obs
        out = obs.copy()
        if self._pink is not None:
            noise = self._pink.sample().astype(out.dtype)
        else:
            noise = self._rng.normal(0.0, self.sigma_m, size=EE_POS_DIM).astype(out.dtype)
        out[EE_POS_OFFSET : EE_POS_OFFSET + EE_POS_DIM] += noise
        return out

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if self.color == "pink":
                self._pink = _PinkNoise(
                    EE_POS_DIM, self.sigma_m, n_octaves=self.n_octaves, rng=self._rng
                )
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

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.delay_steps == 0:
            return self.env.step(action)
        applied = self._queue.popleft()
        self._queue.append(np.asarray(action, dtype=self.env.action_space.dtype))
        return self.env.step(applied)
