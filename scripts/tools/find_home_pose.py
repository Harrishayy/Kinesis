"""M1 — solve for a Panda home qpos that places the hand near a target point.

Seeds with the `home` keyframe shipped in `panda.xml` and does a short random
search in joint space, picking the qpos that minimizes ||hand.xpos - target||.
Prints the resulting 7-vector for paste into the env config.

Run:
    uv run python scripts/find_home_pose.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSET = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)
TARGET = np.array([0.5, 0.0, 0.4])
N_TRIALS = 8000
NOISE_SIGMA = 0.5  # rad, per joint
SEED = 0


JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ]
)


def ee_pos(model: mujoco.MjModel, data: mujoco.MjData, qpos7: np.ndarray) -> np.ndarray:
    data.qpos[:7] = qpos7
    data.qpos[7:9] = 0.04  # fingers open, doesn't matter for hand frame
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.body("hand").xpos.copy()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)

    home_kf = model.keyframe("home").qpos[:7].copy()
    rng = np.random.default_rng(SEED)

    best_q = home_kf.copy()
    best_err = float(np.linalg.norm(ee_pos(model, data, best_q) - TARGET))

    for _ in range(N_TRIALS):
        q = home_kf + rng.normal(0.0, NOISE_SIGMA, size=7)
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        ee = ee_pos(model, data, q)
        err = float(np.linalg.norm(ee - TARGET))
        if err < best_err:
            best_err = err
            best_q = q.copy()

    best_ee = ee_pos(model, data, best_q)
    print("target_xyz:", TARGET.tolist())
    print(f"best_ee_xyz: {best_ee.tolist()}")
    print(f"best_ee_error_m: {best_err:.5f}")
    print("home_qpos:")
    print("  [" + ", ".join(f"{v:.6f}" for v in best_q) + "]")


if __name__ == "__main__":
    main()
