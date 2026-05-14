"""Gymnasium environment: Franka Panda end-effector tracking a Cartesian trajectory.

Action: 7-D delta in joint position, scaled to ±`max_delta_rad`. Joint limits enforced.
Observation: see `_obs()` — q, q̇, ee position, target, lookahead, phase, prev action.
Reward: stubbed to 0.0 in this milestone — populated in M3.

Loads the Franka model from `assets/mujoco_menagerie/franka_emika_panda/scene.xml`.
The end-effector frame is the `hand` body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from kinesis.trajectories import Trajectory, build_trajectory

_DEFAULT_SCENE = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

N_ARM_JOINTS = 7


@dataclass(frozen=True)
class PandaTrackConfig:
    control_hz: float = 50.0
    max_steps: int = 500
    max_delta_rad: float = 0.0873
    reset_noise_rad: float = 0.02
    lookahead_n: int = 4
    lookahead_dt_s: float = 0.1
    home_qpos: tuple[float, ...] = (
        0.257813,
        0.163813,
        -0.335467,
        -1.865636,
        0.389176,
        1.429329,
        -0.965127,
    )
    trajectory_kind: str = "circle"
    trajectory_period_s: float = 4.0
    trajectory_center_xyz: tuple[float, float, float] = (0.5, 0.0, 0.4)
    # Circle-only:
    trajectory_radius_m: float = 0.15
    # Figure8_3D-only:
    trajectory_amp_x_m: float = 0.10
    trajectory_amp_y_m: float = 0.15
    trajectory_amp_z_m: float = 0.10
    # Reward weights (see CLAUDE.md). Keep prior values in comments when tuning.
    w_track: float = 10.0
    w_action_rate: float = 0.1
    w_qdot: float = 0.001
    w_inband: float = 0.5
    inband_threshold_m: float = 0.02


class PandaTrackEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: PandaTrackConfig | None = None,
        scene_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or PandaTrackConfig()
        self.model = mujoco.MjModel.from_xml_path(str(scene_path or _DEFAULT_SCENE))
        self.data = mujoco.MjData(self.model)

        sim_dt = float(self.model.opt.timestep)
        control_dt = 1.0 / self.cfg.control_hz
        n_substeps = int(round(control_dt / sim_dt))
        if n_substeps < 1:
            raise ValueError(
                f"sim timestep {sim_dt}s coarser than control period {control_dt}s"
            )
        self.n_substeps = n_substeps

        center = np.asarray(self.cfg.trajectory_center_xyz, dtype=np.float64)
        traj_params: dict[str, float] = {"period_s": self.cfg.trajectory_period_s}
        if self.cfg.trajectory_kind == "circle":
            traj_params["radius_m"] = self.cfg.trajectory_radius_m
        elif self.cfg.trajectory_kind == "figure8_3d":
            traj_params["amp_x_m"] = self.cfg.trajectory_amp_x_m
            traj_params["amp_y_m"] = self.cfg.trajectory_amp_y_m
            traj_params["amp_z_m"] = self.cfg.trajectory_amp_z_m
        self.trajectory: Trajectory = build_trajectory(
            self.cfg.trajectory_kind, center, **traj_params
        )

        joint_lo, joint_hi = self._joint_limits()
        self._joint_lo = joint_lo
        self._joint_hi = joint_hi

        obs_dim = (
            N_ARM_JOINTS  # q
            + N_ARM_JOINTS  # qdot
            + 3  # ee_pos
            + 3  # target
            + 3 * self.cfg.lookahead_n  # lookahead targets
            + 2  # phase sin/cos
            + N_ARM_JOINTS  # prev_action
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ARM_JOINTS,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._prev_action = np.zeros(N_ARM_JOINTS, dtype=np.float64)

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(N_ARM_JOINTS)
        hi = np.zeros(N_ARM_JOINTS)
        for i in range(N_ARM_JOINTS):
            jnt = self.model.joint(f"joint{i+1}")
            lo[i], hi[i] = jnt.range
        return lo, hi

    def _t(self) -> float:
        return self._step_idx / self.cfg.control_hz

    def _ee_pos(self) -> np.ndarray:
        return self.data.body("hand").xpos.copy()

    def _obs(self) -> np.ndarray:
        t = self._t()
        q = self.data.qpos[:N_ARM_JOINTS].copy()
        qdot = self.data.qvel[:N_ARM_JOINTS].copy()
        ee = self._ee_pos()
        target = self.trajectory.target(t)
        lookahead = self.trajectory.lookahead(
            t=t, n=self.cfg.lookahead_n, dt=self.cfg.lookahead_dt_s
        ).reshape(-1)
        phase = self.trajectory.phase_sin_cos(t)
        return np.concatenate(
            [q, qdot, ee, target, lookahead, phase, self._prev_action]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        home = np.asarray(self.cfg.home_qpos, dtype=np.float64)
        noise = self._rng.normal(0.0, self.cfg.reset_noise_rad, size=N_ARM_JOINTS)
        q0 = np.clip(home + noise, self._joint_lo, self._joint_hi)
        self.data.qpos[:N_ARM_JOINTS] = q0
        # Fingers open; we don't actuate them.
        if self.model.nq >= 9:
            self.data.qpos[7:9] = 0.04
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_idx = 0
        self._prev_action[:] = 0.0
        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64).reshape(N_ARM_JOINTS)
        a = np.clip(a, -1.0, 1.0)
        delta_q = a * self.cfg.max_delta_rad

        target_q = np.clip(
            self.data.qpos[:N_ARM_JOINTS] + delta_q, self._joint_lo, self._joint_hi
        )
        # Drive the actuators by setting ctrl directly to the target joint angles
        # (the panda model uses position actuators with ctrlrange == joint range).
        self.data.ctrl[:N_ARM_JOINTS] = target_q
        if self.model.nu >= 8:
            self.data.ctrl[7] = 0.0  # gripper closed-but-unused

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        prev_action = self._prev_action
        self._prev_action = a.copy()
        self._step_idx += 1

        ee = self._ee_pos()
        target = self.trajectory.target(self._t())
        qdot = self.data.qvel[:N_ARM_JOINTS]
        reward, reward_terms = self._reward(
            ee=ee, target=target, action=a, prev_action=prev_action, qdot=qdot
        )

        obs = self._obs()
        terminated = False
        truncated = self._step_idx >= self.cfg.max_steps
        info = {
            "ee_pos": ee,
            "target": target,
            "ee_error_m": float(np.linalg.norm(ee - target)),
            **reward_terms,
        }
        return obs, float(reward), terminated, truncated, info

    def _reward(
        self,
        *,
        ee: np.ndarray,
        target: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        qdot: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        err = float(np.linalg.norm(ee - target))
        track = -self.cfg.w_track * (err * err)
        action_rate = -self.cfg.w_action_rate * float(np.sum((action - prev_action) ** 2))
        qdot_pen = -self.cfg.w_qdot * float(np.sum(qdot * qdot))
        inband = self.cfg.w_inband if err < self.cfg.inband_threshold_m else 0.0
        total = track + action_rate + qdot_pen + inband
        return total, {
            "r_track": track,
            "r_action_rate": action_rate,
            "r_qdot": qdot_pen,
            "r_inband": inband,
        }

    def close(self) -> None:
        # MjData/MjModel hold no OS resources requiring explicit cleanup.
        pass
