"""Gymnasium environment: Franka Panda end-effector tracking a Cartesian trajectory.

Action: 7-D delta in joint position, scaled to ±`max_delta_rad`. Joint limits enforced.
Observation: see `_obs()` — q, q̇, ee position, target, lookahead, phase, prev action,
optionally a feedforward IK action (residual configs) and an orientation block
(orientation-tracking configs).

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

from kinesis.orientation import R_DESIRED, R_to_6d, geodesic_angle, log_so3
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
    # Viviani-only:
    trajectory_sphere_radius_m: float = 0.12
    # Reward weights. Keep prior values in comments when tuning.
    w_track: float = 10.0
    w_action_rate: float = 0.1
    w_qdot: float = 0.001
    w_inband: float = 0.5
    inband_threshold_m: float = 0.02
    # Legacy orientation alignment penalty: keep hand-z aligned with world −z
    # (hand pointing down). For non-residual configs this is the only signal
    # that collapses the arm's 4-DoF position-only null space; otherwise the
    # policy is free to drift into folded configurations that satisfy position
    # tracking but are mechanically unrealistic. Subsumed (and overridden) by
    # `include_orientation = True` below — orientation-tracking configs leave
    # `w_orient = 0`.
    w_orient: float = 0.0
    # Residual RL: when enabled, the env adds a Jacobian-pseudoinverse feedforward
    # action to the policy's action each step, and exposes that feedforward to
    # the policy via the observation. Policy then learns only the residual.
    residual_ff_enabled: bool = False
    residual_ff_p_gain: float = 0.5
    residual_ff_damping: float = 0.01
    residual_ff_clip: float = 0.8  # headroom left for the residual
    # When residual FF is enabled, also locks the EE orientation to the
    # `trajectory.orientation(t)` target via the 6-DoF damped-LS IK. If
    # `include_orientation = False` that target is the constant
    # `R_DESIRED` (palm down); if True it follows the curve.
    residual_ff_orient_gain: float = 0.5
    # Orientation tracking (optional, kept modular). When False the
    # trajectory orientation target is the constant `R_DESIRED` and the reward
    # / obs revert to the position-only quadratic form. When True:
    #  - obs gains a rotation block (see `obs_layout`)
    #  - reward switches to the multiplicative exponential form proposed in
    #    arXiv:2412.03012: r_track = w_track · r_pos · (1 + r_ori), where
    #    r_pos = exp(-||err||/σ_p) and r_ori = exp(-||log(R*Rᵀ)||/σ_R). This
    #    bounds rewards in (0, 2·w_track] and gates orientation behind
    #    position progress, which is what kept PPO stable for 6-DoF pose
    #    tracking. The angular-rate smoothness penalty (`w_omega`) and the
    #    legacy quadratic position term `w_track` are still active, but
    #    `w_track` scales the *exponential* form here.
    #  - the residual FF (if enabled) tracks `trajectory.orientation(t)` with
    #    `residual_ff_orient_gain` scaling the FF's orientation correction.
    include_orientation: bool = False
    # σ_p in metres — position error at which r_pos = 1/e. A 5cm scale puts the
    # gradient where we want it (steep below 5cm, gentle above).
    r_pos_scale: float = 0.05
    # σ_R in radians — orientation error at which r_ori = 1/e. A 0.5 rad
    # (~28.6°) scale matches the position scale in "effort to fix" units.
    r_ori_scale: float = 0.5
    w_omega: float = 0.05
    orient_lookahead_n: int = 4
    orient_lookahead_dt_s: float = 0.1


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
        elif self.cfg.trajectory_kind == "viviani":
            traj_params["sphere_radius_m"] = self.cfg.trajectory_sphere_radius_m
        self.trajectory: Trajectory = build_trajectory(
            self.cfg.trajectory_kind, center, **traj_params
        )

        joint_lo, joint_hi = self._joint_limits()
        self._joint_lo = joint_lo
        self._joint_hi = joint_hi

        # Build the obs layout once so wrappers (and tests) can look up slices
        # by name instead of recomputing offsets. Layout matches `_obs()` exactly.
        self._obs_layout = self._build_obs_layout()
        obs_dim = self._obs_layout["__total__"][1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ARM_JOINTS,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._prev_action = np.zeros(N_ARM_JOINTS, dtype=np.float64)
        self._prev_ee = np.zeros(3, dtype=np.float64)
        self._prev_R_ee = np.eye(3, dtype=np.float64)

    def _build_obs_layout(self) -> dict[str, tuple[int, int]]:
        """Map each obs block to its `(start, end)` slice within the obs vector.

        Wrappers (e.g. `ObsNoiseWrapper`) use this to find the EE-position and
        rotation slices regardless of which optional flags are enabled.
        """
        layout: dict[str, tuple[int, int]] = {}
        idx = 0

        def _add(name: str, width: int) -> None:
            nonlocal idx
            layout[name] = (idx, idx + width)
            idx += width

        _add("q", N_ARM_JOINTS)
        _add("qdot", N_ARM_JOINTS)
        _add("ee_pos", 3)
        if self.cfg.include_cartesian_velocities:
            _add("ee_vel", 3)
            _add("target", 3)
            _add("target_vel", 3)
        else:
            _add("target", 3)
        _add("lookahead", 3 * self.cfg.lookahead_n)
        _add("phase", 2)
        _add("prev_action", N_ARM_JOINTS)
        if self.cfg.residual_ff_enabled:
            _add("a_ff", N_ARM_JOINTS)
        if self.cfg.include_orientation:
            _add("R_ee_6d", 6)
            _add("R_target_6d", 6)
            _add("R_target_lookahead_6d", 6 * self.cfg.orient_lookahead_n)
        layout["__total__"] = (0, idx)
        return layout

    def obs_layout(self) -> dict[str, tuple[int, int]]:
        """Public read-only view of the obs layout."""
        return dict(self._obs_layout)

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

    def _R_ee(self) -> np.ndarray:
        """Current EE rotation matrix (3, 3), copied so callers can mutate."""
        return self.data.body("hand").xmat.reshape(3, 3).copy()

    def _R_target(self, t: float | None = None) -> np.ndarray:
        """Time-varying orientation target.

        When orientation tracking is enabled, delegates to the trajectory's
        `orientation(t)`; otherwise returns the constant palm-down pose so
        non-orient configs preserve their pre-change behaviour exactly.
        """
        if not self.cfg.include_orientation:
            return R_DESIRED
        if t is None:
            t = self._t()
        return self.trajectory.orientation(float(t))

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

        Warm-starts from the current qpos. When `residual_ff_enabled` is True,
        also locks the EE orientation to `_R_target(0)` (constant palm-down
        when orientation tracking is off; the trajectory's start-pose
        otherwise), collapsing the arm's null space so the policy cannot
        select a folded-elbow IK branch at reset.
        """
        if self.cfg.residual_ff_enabled:
            R_target_reset = self._R_target(0.0)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            eye6 = np.eye(6)
            for _ in range(max_iters):
                mujoco.mj_forward(self.model, self.data)
                pos_err = target_xyz - self._ee_pos()
                R = self._R_ee()
                R_err = R_target_reset @ R.T
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
        orientation term pulls the current EE rotation toward `_R_target(t)`:
        the constant palm-down pose when orientation tracking is off (just
        collapses the arm's null space), or the trajectory's time-varying
        rotation when it is on (also tracks the orientation target).
        """
        t = self._t()
        dt = 1.0 / self.cfg.control_hz
        target = self.trajectory.target(t)
        next_target = self.trajectory.lookahead(t=t, n=1, dt=dt)[0]
        ee = self._ee_pos()
        delta_pos = (next_target - target) + self.cfg.residual_ff_p_gain * (target - ee)

        R = self._R_ee()
        R_err = self._R_target(t) @ R.T
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
        if self.cfg.include_orientation:
            R_ee = self._R_ee()
            R_target = self._R_target(t)
            R_target_la = self.trajectory.orientation_lookahead(
                t=t,
                n=self.cfg.orient_lookahead_n,
                dt=self.cfg.orient_lookahead_dt_s,
            )
            la_6d = np.concatenate([R_to_6d(R_target_la[i]) for i in range(R_target_la.shape[0])])
            parts += [R_to_6d(R_ee), R_to_6d(R_target), la_6d]
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
        self._prev_R_ee = self._R_ee()  # zero angular-velocity on first step.
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
        prev_R_ee = self._prev_R_ee
        R_ee_now = self._R_ee()
        reward, reward_terms, orient_info = self._reward(
            ee=ee,
            target=target,
            action=a,
            prev_action=prev_action,
            qdot=qdot,
            R_ee=R_ee_now,
            prev_R_ee=prev_R_ee,
        )

        obs = self._obs()
        # _obs uses self._prev_ee / self._prev_R_ee; update only after the obs
        # is built so finite differences use (this-step) − (last-step).
        self._prev_ee = ee.copy()
        self._prev_R_ee = R_ee_now
        terminated = False
        truncated = self._step_idx >= self.cfg.max_steps
        info = {
            "ee_pos": ee,
            "target": target,
            "ee_error_m": float(np.linalg.norm(ee - target)),
            **reward_terms,
            **orient_info,
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
        R_ee: np.ndarray,
        prev_R_ee: np.ndarray,
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        err = float(np.linalg.norm(ee - target))
        action_rate = -self.cfg.w_action_rate * float(np.sum((action - prev_action) ** 2))
        qdot_pen = -self.cfg.w_qdot * float(np.sum(qdot * qdot))
        orient_info: dict[str, Any] = {}

        if self.cfg.include_orientation:
            # 6-DoF tracking: multiplicative-exponential form (arXiv:2412.03012).
            # Bounded in (0, 2·w_track], so PPO advantages and KL stay tame.
            R_target = self._R_target()
            theta = geodesic_angle(R_ee, R_target)
            r_pos = float(np.exp(-err / self.cfg.r_pos_scale))
            r_ori = float(np.exp(-theta / self.cfg.r_ori_scale))
            track = self.cfg.w_track * r_pos * (1.0 + r_ori)
            inband = 0.0  # subsumed by r_pos
            orient = 0.0  # legacy palm-down regulariser off when tracking
            # Angular velocity from successive clean EE rotations.
            dR = R_ee @ prev_R_ee.T
            omega = log_so3(dR) * self.cfg.control_hz
            r_omega = -self.cfg.w_omega * float(omega @ omega)
            orient_info["orient_err_rad"] = float(theta)
            orient_info["omega_ee"] = omega
            orient_info["R_ee"] = R_ee.copy()
            orient_info["R_target"] = R_target.copy()
        else:
            # Position-only configs preserve the legacy quadratic reward
            # exactly — `viviani_residual` and friends must keep printing the
            # same numbers reported in RESULTS.md.
            track = -self.cfg.w_track * (err * err)
            inband = self.cfg.w_inband if err < self.cfg.inband_threshold_m else 0.0
            if self.cfg.w_orient > 0.0:
                hand_z2 = float(R_ee[2, 2])
                orient = -self.cfg.w_orient * (1.0 + hand_z2) ** 2
            else:
                orient = 0.0
            r_omega = 0.0

        total = track + action_rate + qdot_pen + inband + orient + r_omega
        return (
            total,
            {
                "r_track": track,
                "r_action_rate": action_rate,
                "r_qdot": qdot_pen,
                "r_inband": inband,
                "r_orient": orient,
                "r_omega": r_omega,
            },
            orient_info,
        )

    def close(self) -> None:
        # MjData/MjModel hold no OS resources requiring explicit cleanup.
        pass
