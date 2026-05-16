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

# Canonical "hand pointing down at the workspace" orientation. Columns are the
# hand frame's axes expressed in the world frame: hand-z = world −z (palm faces
# down at the table), hand-x = world +x (along workspace forward direction),
# hand-y = world −y (right-handed orthogonal). Used by the 6-DoF residual FF
# IK to lock the arm out of folded null-space configurations.
_R_DESIRED = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PandaTrackConfig:
    control_hz: float = 50.0
    max_steps: int = 500
    max_delta_rad: float = 0.0873
    reset_noise_rad: float = 0.02
    lookahead_n: int = 4
    lookahead_dt_s: float = 0.1
    # Distance from the `hand` body origin to the tool-centre point along the
    # hand-z axis. The Franka official spec is 103.4 mm — this places the
    # tracked point at the midpoint between the gripper fingertips, which is
    # what people mean by "end-effector position" in robotics. Setting to 0
    # reverts to tracking the wrist origin (pre-TCP behaviour).
    tip_offset_m: float = 0.0
    # When True, reset runs a damped-LS IK to drive EE to trajectory.target(0)
    # before applying reset_noise_rad. Mirrors how real robots use a motion
    # planner to reach the approach pose before engaging the tracking
    # controller, so plots aren't dominated by a cold-start reach transient.
    start_at_target: bool = False
    # When True, _obs() includes Cartesian EE velocity (finite-diff on clean
    # internal position) and target velocity (analytic, via first lookahead
    # sample), adding 6 dims. Gated for back-compat with checkpoints trained
    # before this obs upgrade landed.
    include_cartesian_velocities: bool = False
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
    # Viviani-only:
    trajectory_sphere_radius_m: float = 0.12
    # Reward weights. Keep prior values in comments when tuning.
    w_track: float = 10.0
    w_action_rate: float = 0.1
    w_qdot: float = 0.001
    w_inband: float = 0.5
    inband_threshold_m: float = 0.02
    # Orientation alignment penalty: keep hand-z aligned with world −z (hand
    # pointing down). For non-residual configs this is the only signal that
    # collapses the arm's 4-DoF position-only null space; otherwise the policy
    # is free to drift into folded configurations that satisfy position
    # tracking but are mechanically unrealistic.
    w_orient: float = 0.0
    # Residual RL: when enabled, the env adds a Jacobian-pseudoinverse feedforward
    # action to the policy's action each step, and exposes that feedforward to
    # the policy via the observation. Policy then learns only the residual.
    residual_ff_enabled: bool = False
    residual_ff_p_gain: float = 0.5
    residual_ff_damping: float = 0.01
    residual_ff_clip: float = 0.8  # headroom left for the residual
    # When residual FF is enabled, also lock the EE orientation to a canonical
    # "hand pointing down" pose (hand-z = world −z, hand-x = world +x). The
    # 6-DoF IK eliminates the arm's 4-DoF null space, so the policy can't fold
    # the elbow over the shoulder while satisfying position-only tracking.
    residual_ff_orient_gain: float = 0.5


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
            raise ValueError(f"sim timestep {sim_dt}s coarser than control period {control_dt}s")
        self.n_substeps = n_substeps

        center = np.asarray(self.cfg.trajectory_center_xyz, dtype=np.float64)
        traj_params: dict[str, float] = {"period_s": self.cfg.trajectory_period_s}
        if self.cfg.trajectory_kind == "circle":
            traj_params["radius_m"] = self.cfg.trajectory_radius_m
        elif self.cfg.trajectory_kind == "figure8_3d":
            traj_params["amp_x_m"] = self.cfg.trajectory_amp_x_m
            traj_params["amp_y_m"] = self.cfg.trajectory_amp_y_m
            traj_params["amp_z_m"] = self.cfg.trajectory_amp_z_m
        elif self.cfg.trajectory_kind == "viviani":
            traj_params["sphere_radius_m"] = self.cfg.trajectory_sphere_radius_m
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
        if self.cfg.include_cartesian_velocities:
            # ee_vel (finite-diff on clean internal position; noise wrapper only
            # corrupts ee_pos, mirroring an encoder-based velocity) + target_vel.
            obs_dim += 6
        if self.cfg.residual_ff_enabled:
            obs_dim += N_ARM_JOINTS  # a_ff so the policy sees what it is correcting
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ARM_JOINTS,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._prev_action = np.zeros(N_ARM_JOINTS, dtype=np.float64)
        self._prev_ee = np.zeros(3, dtype=np.float64)

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(N_ARM_JOINTS)
        hi = np.zeros(N_ARM_JOINTS)
        for i in range(N_ARM_JOINTS):
            jnt = self.model.joint(f"joint{i + 1}")
            lo[i], hi[i] = jnt.range
        return lo, hi

    def _t(self) -> float:
        return self._step_idx / self.cfg.control_hz

    def _ee_pos(self) -> np.ndarray:
        """Tracked end-effector point in world coordinates.

        When `tip_offset_m > 0` this returns the gripper TCP — the wrist
        origin offset along the hand's local z-axis by `tip_offset_m` (Franka's
        103.4 mm spec puts this at the fingertip midpoint). With offset 0 it
        returns the wrist origin, preserving pre-TCP behaviour.
        """
        hand = self.data.body("hand")
        if self.cfg.tip_offset_m == 0.0:
            return hand.xpos.copy()
        hand_z_world = hand.xmat.reshape(3, 3)[:, 2]
        return hand.xpos + self.cfg.tip_offset_m * hand_z_world

    def _ee_jacobian(self, jacp: np.ndarray, jacr: np.ndarray | None) -> None:
        """Fill jacp (and optionally jacr) with the Jacobian of the tracked
        EE point, accounting for `tip_offset_m`. Uses `mj_jac` at the world-
        coordinate location of the tip so the resulting Jacobian moves the
        tip — not the wrist — through the desired Cartesian step."""
        if self.cfg.tip_offset_m == 0.0:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.model.body("hand").id)
        else:
            point = self._ee_pos()
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.model.body("hand").id)

    def _ik_to_target(
        self,
        target_xyz: np.ndarray,
        *,
        max_iters: int = 200,
        tol_m: float = 1e-4,
        damping: float = 0.01,
        max_step_rad: float = 0.1,
    ) -> float:
        """Damped-LS IK on the arm joints to drive EE to `target_xyz`.

        Warm-starts from the current qpos. When residual_ff_enabled is True,
        also locks the EE orientation to _R_DESIRED (hand pointing down),
        which collapses the arm's null space so the policy cannot select a
        folded-elbow IK branch at reset.
        """
        if self.cfg.residual_ff_enabled:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            eye6 = np.eye(6)
            for _ in range(max_iters):
                mujoco.mj_forward(self.model, self.data)
                pos_err = target_xyz - self._ee_pos()
                R = self.data.body("hand").xmat.reshape(3, 3)
                R_err = _R_DESIRED @ R.T
                omega = 0.5 * np.array(
                    [
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ]
                )
                if np.linalg.norm(pos_err) < tol_m and np.linalg.norm(omega) < 1e-3:
                    break
                self._ee_jacobian(jacp, jacr)
                J = np.vstack([jacp[:, :N_ARM_JOINTS], jacr[:, :N_ARM_JOINTS]])
                full_err = np.concatenate([pos_err, omega])
                dq = J.T @ np.linalg.solve(J @ J.T + damping * eye6, full_err)
                norm = float(np.linalg.norm(dq))
                if norm > max_step_rad:
                    dq *= max_step_rad / norm
                self.data.qpos[:N_ARM_JOINTS] = np.clip(
                    self.data.qpos[:N_ARM_JOINTS] + dq, self._joint_lo, self._joint_hi
                )
        else:
            jacp = np.zeros((3, self.model.nv))
            eye3 = np.eye(3)
            for _ in range(max_iters):
                mujoco.mj_forward(self.model, self.data)
                err = target_xyz - self._ee_pos()
                if np.linalg.norm(err) < tol_m:
                    break
                self._ee_jacobian(jacp, None)
                J = jacp[:, :N_ARM_JOINTS]
                dq = J.T @ np.linalg.solve(J @ J.T + damping * eye3, err)
                norm = float(np.linalg.norm(dq))
                if norm > max_step_rad:
                    dq *= max_step_rad / norm
                self.data.qpos[:N_ARM_JOINTS] = np.clip(
                    self.data.qpos[:N_ARM_JOINTS] + dq, self._joint_lo, self._joint_hi
                )
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(target_xyz - self._ee_pos()))

    def _feedforward_action(self) -> np.ndarray:
        """6-DoF damped Jacobian-pseudoinverse feedforward in scaled action units.

        Solves J_full · Δq = [Δee_pos; ω_orient] with Tikhonov damping λ to stay
        sane near singular configurations, then scales Δq to the env's [−1, 1]
        action. The position term combines an exact one-step trajectory
        displacement with a proportional pull toward the current target. The
        orientation term pulls hand-z toward world −z (canonical "hand pointing
        down"), which collapses the arm's null space so the policy cannot fold
        the elbow over the shoulder while still satisfying position tracking.
        """
        t = self._t()
        dt = 1.0 / self.cfg.control_hz
        target = self.trajectory.target(t)
        next_target = self.trajectory.lookahead(t=t, n=1, dt=dt)[0]
        ee = self._ee_pos()
        delta_pos = (next_target - target) + self.cfg.residual_ff_p_gain * (target - ee)

        R = self.data.body("hand").xmat.reshape(3, 3)
        R_err = _R_DESIRED @ R.T
        omega = 0.5 * np.array(
            [
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]
        )
        delta_full = np.concatenate([delta_pos, self.cfg.residual_ff_orient_gain * omega])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self._ee_jacobian(jacp, jacr)
        J = np.vstack([jacp[:, :N_ARM_JOINTS], jacr[:, :N_ARM_JOINTS]])  # 6 × 7

        lam = self.cfg.residual_ff_damping
        JJt = J @ J.T + lam * np.eye(6)
        delta_q = J.T @ np.linalg.solve(JJt, delta_full)

        a_ff = delta_q / self.cfg.max_delta_rad
        clip = self.cfg.residual_ff_clip
        return np.clip(a_ff, -clip, clip)

    def _obs(self) -> np.ndarray:
        t = self._t()
        q = self.data.qpos[:N_ARM_JOINTS].copy()
        qdot = self.data.qvel[:N_ARM_JOINTS].copy()
        ee = self._ee_pos()
        target = self.trajectory.target(t)
        lookahead_mat = self.trajectory.lookahead(
            t=t, n=self.cfg.lookahead_n, dt=self.cfg.lookahead_dt_s
        )
        lookahead = lookahead_mat.reshape(-1)
        phase = self.trajectory.phase_sin_cos(t)
        parts: list[np.ndarray] = [q, qdot, ee]
        if self.cfg.include_cartesian_velocities:
            ee_vel = (ee - self._prev_ee) * self.cfg.control_hz
            target_vel = (lookahead_mat[0] - target) / self.cfg.lookahead_dt_s
            parts += [ee_vel, target, target_vel]
        else:
            parts.append(target)
        parts += [lookahead, phase, self._prev_action]
        if self.cfg.residual_ff_enabled:
            parts.append(self._feedforward_action())
        return np.concatenate(parts).astype(np.float32)

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
        # Fingers open; we don't actuate them.
        if self.model.nq >= 9:
            self.data.qpos[7:9] = 0.04
        self.data.qvel[:] = 0.0

        if self.cfg.start_at_target:
            # "Planner brings EE to approach pose" — IK warm-started from home,
            # then small joint noise on top represents residual calibration /
            # sensor offset, which is the relevant initial-state mismatch for a
            # tracking benchmark (138 mm cold-starts are a reach problem, not a
            # tracking one).
            self.data.qpos[:N_ARM_JOINTS] = home
            self._ik_to_target(self.trajectory.target(0.0))
            if self.cfg.reset_noise_rad > 0.0:
                noise = self._rng.normal(0.0, self.cfg.reset_noise_rad, size=N_ARM_JOINTS)
                self.data.qpos[:N_ARM_JOINTS] = np.clip(
                    self.data.qpos[:N_ARM_JOINTS] + noise, self._joint_lo, self._joint_hi
                )
                mujoco.mj_forward(self.model, self.data)
        else:
            noise = self._rng.normal(0.0, self.cfg.reset_noise_rad, size=N_ARM_JOINTS)
            q0 = np.clip(home + noise, self._joint_lo, self._joint_hi)
            self.data.qpos[:N_ARM_JOINTS] = q0
            mujoco.mj_forward(self.model, self.data)

        self._step_idx = 0
        self._prev_action[:] = 0.0
        self._prev_ee = self._ee_pos()  # zero EE-velocity on first obs.
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a_in = np.asarray(action, dtype=np.float64).reshape(N_ARM_JOINTS)
        if self.cfg.residual_ff_enabled:
            a_ff = self._feedforward_action()
            a = np.clip(a_in + a_ff, -1.0, 1.0)
        else:
            a = np.clip(a_in, -1.0, 1.0)
        delta_q = a * self.cfg.max_delta_rad

        target_q = np.clip(self.data.qpos[:N_ARM_JOINTS] + delta_q, self._joint_lo, self._joint_hi)
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
        # _obs uses self._prev_ee; update only after the obs is built so the
        # finite-difference uses (this-step ee) − (last-step ee).
        self._prev_ee = ee.copy()
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
        if self.cfg.w_orient > 0.0:
            # hand-z (z-axis of hand frame in world) should equal world (0,0,-1).
            # R[:, 2] = hand-z; R[2, 2] = z-component of hand-z in world; want -1.
            hand_z2 = float(self.data.body("hand").xmat.reshape(3, 3)[2, 2])
            orient = -self.cfg.w_orient * (1.0 + hand_z2) ** 2
        else:
            orient = 0.0
        total = track + action_rate + qdot_pen + inband + orient
        return total, {
            "r_track": track,
            "r_action_rate": action_rate,
            "r_qdot": qdot_pen,
            "r_inband": inband,
            "r_orient": orient,
        }

    def close(self) -> None:
        # MjData/MjModel hold no OS resources requiring explicit cleanup.
        pass
