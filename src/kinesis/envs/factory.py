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

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT_CONFIG = CONFIGS_DIR / "naive" / "circle.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config.

    `path` may be an absolute/relative file path, or a bare trajectory name
    such as ``"circle"`` / ``"viviani_residual"``. Bare names are resolved by
    recursively searching ``configs/`` (so ``naive/`` and ``residual/`` subdirs
    are both visible without specifying the subdir).
    """
    if path is None:
        path = _DEFAULT_CONFIG
    else:
        p = Path(path)
        if not p.exists() and not p.suffix:
            matches = sorted(CONFIGS_DIR.rglob(f"{p.name}.yaml"))
            if not matches:
                raise FileNotFoundError(f"No config '{p.name}.yaml' under {CONFIGS_DIR}")
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous config '{p.name}': {[str(m.relative_to(CONFIGS_DIR)) for m in matches]}"
                )
            p = matches[0]
        path = p
    with open(path) as f:
        return yaml.safe_load(f)


def _panda_config(cfg: dict) -> PandaTrackConfig:
    env_c = cfg.get("env", {})
    traj_c = cfg.get("trajectory", {})
    rew_c = cfg.get("reward", {})
    defaults = PandaTrackConfig()
    return PandaTrackConfig(
        control_hz=float(env_c.get("control_hz", 50.0)),
        max_steps=int(env_c.get("max_steps", 500)),
        max_delta_rad=float(env_c.get("max_delta_rad", 0.0873)),
        reset_noise_rad=float(env_c.get("reset_noise_rad", 0.02)),
        lookahead_n=int(env_c.get("lookahead_n", 4)),
        lookahead_dt_s=float(env_c.get("lookahead_dt_s", 0.1)),
        home_qpos=tuple(env_c.get("home_qpos", defaults.home_qpos)),
        start_at_target=bool(env_c.get("start_at_target", defaults.start_at_target)),
        tip_offset_m=float(env_c.get("tip_offset_m", defaults.tip_offset_m)),
        include_cartesian_velocities=bool(
            env_c.get("include_cartesian_velocities", defaults.include_cartesian_velocities)
        ),
        trajectory_kind=str(traj_c.get("kind", "circle")),
        trajectory_period_s=float(traj_c.get("period_s", 4.0)),
        trajectory_center_xyz=tuple(traj_c.get("center_xyz", (0.5, 0.0, 0.4))),
        trajectory_radius_m=float(traj_c.get("radius_m", defaults.trajectory_radius_m)),
        trajectory_sphere_radius_m=float(
            traj_c.get("sphere_radius_m", defaults.trajectory_sphere_radius_m)
        ),
        w_track=float(rew_c.get("w_track", 10.0)),
        w_action_rate=float(rew_c.get("w_action_rate", 0.1)),
        w_qdot=float(rew_c.get("w_qdot", 0.001)),
        w_inband=float(rew_c.get("w_inband", 0.5)),
        w_orient=float(rew_c.get("w_orient", defaults.w_orient)),
        w_omega=float(rew_c.get("w_omega", defaults.w_omega)),
        r_pos_scale=float(rew_c.get("r_pos_scale", defaults.r_pos_scale)),
        r_ori_scale=float(rew_c.get("r_ori_scale", defaults.r_ori_scale)),
        include_orientation=bool(env_c.get("include_orientation", defaults.include_orientation)),
        orient_lookahead_n=int(env_c.get("orient_lookahead_n", defaults.orient_lookahead_n)),
        orient_lookahead_dt_s=float(
            env_c.get("orient_lookahead_dt_s", defaults.orient_lookahead_dt_s)
        ),
        residual_ff_enabled=bool(cfg.get("residual_ff", {}).get("enabled", False)),
        residual_ff_p_gain=float(
            cfg.get("residual_ff", {}).get("p_gain", defaults.residual_ff_p_gain)
        ),
        residual_ff_damping=float(
            cfg.get("residual_ff", {}).get("damping", defaults.residual_ff_damping)
        ),
        residual_ff_clip=float(cfg.get("residual_ff", {}).get("clip", defaults.residual_ff_clip)),
        residual_ff_orient_gain=float(
            cfg.get("residual_ff", {}).get("orient_gain", defaults.residual_ff_orient_gain)
        ),
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
        sigma_R = float(wcfg.get("obs_noise_sigma_R_rad", 0.0))
        delay = int(wcfg.get("action_delay_steps", 0))
        color = str(wcfg.get("noise_color", "white"))
        n_octaves = int(wcfg.get("noise_octaves", 6))
        if sigma > 0 or sigma_R > 0:
            env = ObsNoiseWrapper(
                env,
                sigma_m=sigma,
                seed=seed,
                color=color,
                n_octaves=n_octaves,
                sigma_R_rad=sigma_R,
            )
        if delay > 0:
            env = ActionDelayWrapper(env, delay_steps=delay)
    return env


def env_thunk(cfg: dict, seed: int, *, apply_wrappers: bool = True) -> Callable[[], gym.Env]:
    """Returns a zero-arg callable that builds an env. Required by SB3 VecEnv."""

    def _thunk() -> gym.Env:
        return make_env(cfg, seed=seed, apply_wrappers=apply_wrappers)

    return _thunk
