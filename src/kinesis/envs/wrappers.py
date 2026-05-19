"""Gymnasium wrappers modelling the two uncertainty sources required by the brief:

- `ObsNoiseWrapper`: additive Gaussian noise on the measured end-effector
  position (always), and, when the underlying env exposes a rotation block,
  small axis-angle perturbation on the measured EE rotation. Targets and
  proprioception stay clean — the agent has noisy perception of where it is
  (and how it's oriented), not where it's going.
- `ActionDelayWrapper`: applies the action commanded `k` control steps ago.
  Initial queue is zeros so the first `k` env steps see no command. Because the
  delay is on the *action*, all downstream effects (position, orientation,
  contact) inherit it — no separate orientation-delay wrapper required.

The wrapper looks up obs-block offsets via `PandaTrackEnv.obs_layout()` so a
new optional obs flag doesn't break it. `EE_POS_OFFSET` / `EE_POS_DIM` remain
exported for legacy callers.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from kinesis.envs.panda_track import PandaTrackEnv
from kinesis.orientation import R_to_6d, axis_angle_to_R

# Legacy constants — preserved for callers that import them directly. The
# wrapper itself prefers `_locate_block` (defined below).
EE_POS_OFFSET = 14  # q(7) + qdot(7)
EE_POS_DIM = 3


def _unwrap_to_panda(env: gym.Env) -> PandaTrackEnv:
    """Walk down `env.env` chain until we reach the `PandaTrackEnv` instance."""
    cur: Any = env
    while not isinstance(cur, PandaTrackEnv):
        cur = cur.env
    return cur


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
    """Add noise to the measured EE position and (optionally) EE rotation.

    `color` switches between IID Gaussian ("white") and Voss-McCartney 1/f
    ("pink"). Pink noise has non-zero autocorrelation across the lookahead
    window so the policy can't defeat it with a single-step low-pass — it has
    to learn broadband rejection.

    `sigma_R_rad` controls orientation noise: at each step we sample an
    axis-angle perturbation `δ ~ N(0, σ_R²)` per component and right-multiply
    the measured rotation by `exp([δ]_×)`. The convention "noise in the body
    frame of the measurement" mirrors how a real wrist-mounted IMU's
    rotation reading is corrupted in its own frame. The corrupted rotation
    is written back to the 6D continuous slice in the obs; the target
    rotation slice and the rotation lookahead stay clean.

    The wrapper is a no-op for orientation when the underlying env was built
    with `include_orientation=False` — it just won't find an `R_ee_6d` slice
    in the obs layout, and only the position noise applies.
    """

    def __init__(
        self,
        env: gym.Env,
        sigma_m: float,
        seed: int | None = None,
        color: str = "white",
        n_octaves: int = 6,
        sigma_R_rad: float = 0.0,
    ) -> None:
        super().__init__(env)
        if sigma_m < 0:
            raise ValueError(f"sigma_m must be non-negative, got {sigma_m}")
        if sigma_R_rad < 0:
            raise ValueError(f"sigma_R_rad must be non-negative, got {sigma_R_rad}")
        if color not in {"white", "pink"}:
            raise ValueError(f"color must be 'white' or 'pink', got {color!r}")
        self.sigma_m = float(sigma_m)
        self.sigma_R_rad = float(sigma_R_rad)
        self.color = color
        self.n_octaves = int(n_octaves)
        self._rng = np.random.default_rng(seed)
        self._pink_pos: _PinkNoise | None = (
            _PinkNoise(EE_POS_DIM, self.sigma_m, n_octaves=self.n_octaves, rng=self._rng)
            if color == "pink"
            else None
        )
        # Three axis-angle components, sampled either IID or with pink-noise
        # correlation matching the position-noise channel.
        self._pink_rot: _PinkNoise | None = (
            _PinkNoise(3, self.sigma_R_rad, n_octaves=self.n_octaves, rng=self._rng)
            if color == "pink" and self.sigma_R_rad > 0.0
            else None
        )

        panda = _unwrap_to_panda(env)
        layout = panda.obs_layout()
        self._ee_slice = layout["ee_pos"]
        self._R_slice = layout.get("R_ee_6d")  # None if orientation tracking off

    def _sample_pos_noise(self, dtype) -> np.ndarray:
        if self._pink_pos is not None:
            return self._pink_pos.sample().astype(dtype)
        return self._rng.normal(0.0, self.sigma_m, size=EE_POS_DIM).astype(dtype)

    def _sample_rot_axis_angle(self) -> np.ndarray:
        if self._pink_rot is not None:
            return self._pink_rot.sample()
        return self._rng.normal(0.0, self.sigma_R_rad, size=3)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.sigma_m == 0.0 and (self.sigma_R_rad == 0.0 or self._R_slice is None):
            return obs
        out = obs.copy()
        if self.sigma_m > 0.0:
            s, e = self._ee_slice
            out[s:e] += self._sample_pos_noise(out.dtype)
        if self.sigma_R_rad > 0.0 and self._R_slice is not None:
            s, e = self._R_slice
            # Recover R from the 6D rep: first two columns, Gram-Schmidted in
            # case of float drift, third column by cross product.
            v = out[s:e].astype(np.float64)
            a = v[:3]
            b = v[3:]
            a /= max(float(np.linalg.norm(a)), 1e-12)
            b = b - (b @ a) * a
            b /= max(float(np.linalg.norm(b)), 1e-12)
            c = np.cross(a, b)
            R = np.column_stack([a, b, c])
            # Right-multiply by the perturbation — "noise in the body frame".
            delta = self._sample_rot_axis_angle()
            R_noisy = R @ axis_angle_to_R(delta, angle=float(np.linalg.norm(delta)))
            out[s:e] = R_to_6d(R_noisy).astype(out.dtype)
        return out

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if self.color == "pink":
                self._pink_pos = _PinkNoise(
                    EE_POS_DIM, self.sigma_m, n_octaves=self.n_octaves, rng=self._rng
                )
                if self.sigma_R_rad > 0.0:
                    self._pink_rot = _PinkNoise(
                        3, self.sigma_R_rad, n_octaves=self.n_octaves, rng=self._rng
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
