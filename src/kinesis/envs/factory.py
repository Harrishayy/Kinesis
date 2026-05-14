"""Factories for building configured envs and vectorised envs.

Centralises the wrapper stack so training and evaluation use identical envs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import yaml

from kinesis.envs.panda_track import PandaTrackConfig, PandaTrackEnv
from kinesis.envs.wrappers import ActionDelayWrapper, ObsNoiseWrapper

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or _DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def _panda_config(cfg: dict) -> PandaTrackConfig:
    env_c = cfg.get("env", {})
    traj_c = cfg.get("trajectory", {})
    rew_c = cfg.get("reward", {})
    return PandaTrackConfig(
        control_hz=float(env_c.get("control_hz", 50.0)),
        max_steps=int(env_c.get("max_steps", 500)),
        max_delta_rad=float(env_c.get("max_delta_rad", 0.0873)),
        reset_noise_rad=float(env_c.get("reset_noise_rad", 0.02)),
        lookahead_n=int(env_c.get("lookahead_n", 4)),
        lookahead_dt_s=float(env_c.get("lookahead_dt_s", 0.1)),
        home_qpos=tuple(env_c.get("home_qpos", PandaTrackConfig().home_qpos)),
        trajectory_period_s=float(traj_c.get("period_s", 4.0)),
        trajectory_radius_m=float(traj_c.get("radius_m", 0.15)),
        trajectory_center_xyz=tuple(traj_c.get("center_xyz", (0.5, 0.0, 0.4))),
        w_track=float(rew_c.get("w_track", 10.0)),
        w_action_rate=float(rew_c.get("w_action_rate", 0.1)),
        w_qdot=float(rew_c.get("w_qdot", 0.001)),
        w_inband=float(rew_c.get("w_inband", 0.5)),
    )


def make_env(
    cfg: dict,
    seed: int = 0,
    *,
    apply_wrappers: bool = True,
) -> gym.Env:
    """Build a single, optionally-wrapped, env. Use `apply_wrappers=False` for clean eval."""

    env: gym.Env = PandaTrackEnv(config=_panda_config(cfg), seed=seed)
    if apply_wrappers:
        wcfg = cfg.get("wrappers", {})
        sigma = float(wcfg.get("obs_noise_sigma_m", 0.0))
        delay = int(wcfg.get("action_delay_steps", 0))
        if sigma > 0:
            env = ObsNoiseWrapper(env, sigma_m=sigma, seed=seed)
        if delay > 0:
            env = ActionDelayWrapper(env, delay_steps=delay)
    return env


def env_thunk(cfg: dict, seed: int, *, apply_wrappers: bool = True) -> Callable[[], gym.Env]:
    """Returns a zero-arg callable that builds an env. Required by SB3 VecEnv."""

    def _thunk() -> gym.Env:
        return make_env(cfg, seed=seed, apply_wrappers=apply_wrappers)

    return _thunk
