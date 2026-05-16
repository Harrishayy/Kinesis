"""Deterministic evaluation of a trained Kinesis PPO policy.

Outputs (under per-trajectory subdirectories):
- results/<traj>/plots/yz_trace.png        target vs achieved EE (y-z plane)
- results/<traj>/plots/xz_trace.png        only for 3D trajectories
- results/<traj>/plots/error_vs_time.png   tracking error magnitude vs time
- results/<traj>/videos/rollout.mp4        offscreen-rendered rollout video
- prints a one-line metric summary (RMS / max / jerk)

Loads checkpoints/<traj>/best/best_model.zip by default (the best-by-eval
checkpoint that produces the numbers in RESULTS.md). Falls back to
ppo_panda_final.zip if best/ does not exist.

Usage:
    uv run python scripts/eval.py
    uv run python scripts/eval.py --config viviani_residual
    uv run python scripts/eval.py --checkpoint checkpoints/circle/best/best_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from kinesis.envs.factory import load_config, make_env  # noqa: E402
from kinesis.envs.panda_track import PandaTrackEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 50


def _unwrap_to_panda(env) -> PandaTrackEnv:
    while not isinstance(env, PandaTrackEnv):
        env = env.env
    return env


def rollout(model: PPO, env, n_steps: int) -> dict[str, np.ndarray]:
    """Rollout the policy for `n_steps`. The caller is responsible for not
    exceeding the env's max_steps — this function does not reset mid-rollout
    so the trace stays free of synthetic transients."""
    obs, _ = env.reset(seed=0)
    ee_pos = np.zeros((n_steps, 3))
    target = np.zeros((n_steps, 3))
    actions = np.zeros((n_steps, env.action_space.shape[0]))
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        ee_pos[i] = info["ee_pos"]
        target[i] = info["target"]
        actions[i] = action
        if terminated or truncated:
            # Should not happen if caller sized n_steps <= env.max_steps;
            # if it does, stop so we don't double-count a startup transient.
            ee_pos = ee_pos[: i + 1]
            target = target[: i + 1]
            actions = actions[: i + 1]
            break
    return {"ee_pos": ee_pos, "target": target, "actions": actions}


def _add_marker(scene, *, pos, size, rgba, geom_type=None) -> None:
    """Append a sphere/line marker to a mjvScene before rendering. Reads the
    current ngeom, initialises geoms[ngeom] in-place, and increments ngeom."""
    if scene.ngeom >= scene.maxgeom:
        return  # silently drop overflows so a long trail can't crash a render
    if geom_type is None:
        geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=geom_type,
        size=np.asarray(size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def render_video(env, model: PPO, n_steps: int, out_path: Path) -> None:
    """Render a deterministic rollout to video with two trajectory annotations:
    a blue dotted trail showing the full closed curve, and a red "laser-
    pointer" sphere at the current target so a viewer can see what the EE is
    chasing each frame."""
    panda = _unwrap_to_panda(env)
    renderer = mujoco.Renderer(panda.model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Static dotted preview of the full closed curve (so a viewer can see the
    # shape before / after the EE traces it).
    trail_ts = np.linspace(0.0, panda.trajectory.period_s, 96, endpoint=False)
    trail_pts = [panda.trajectory.target(float(t)) for t in trail_ts]

    # Dynamic past-EE trail — grows each frame so the rendered video shows the
    # actual path the gripper TCP has taken so far.
    ee_history: list[np.ndarray] = []
    obs, _ = env.reset(seed=0)
    with imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec="libx264") as writer:
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ee_history.append(np.asarray(info["ee_pos"], dtype=np.float64))
            renderer.update_scene(panda.data, camera=0 if panda.model.ncam else -1)
            scene = renderer.scene
            for pt in trail_pts:
                _add_marker(scene, pos=pt, size=[0.0035, 0, 0], rgba=[0.30, 0.55, 1.0, 0.45])
            for pt in ee_history:
                _add_marker(scene, pos=pt, size=[0.0035, 0, 0], rgba=[1.0, 0.85, 0.10, 0.9])
            current_target = panda.trajectory.target(panda._t())
            _add_marker(
                scene,
                pos=current_target,
                size=[0.013, 0, 0],
                rgba=[1.0, 0.15, 0.15, 1.0],
            )
            writer.append_data(renderer.render())
            if terminated or truncated:
                obs, _ = env.reset(seed=0)
                ee_history.clear()
    renderer.close()


def metrics(
    traces: dict[str, np.ndarray], control_hz: float, settle_s: float = 1.0
) -> dict[str, float]:
    err = np.linalg.norm(traces["ee_pos"] - traces["target"], axis=1)
    rms_full = float(np.sqrt(np.mean(err**2)))
    mx_full = float(err.max())
    settle_idx = int(round(settle_s * control_hz))
    err_ss = err[settle_idx:] if err.size > settle_idx else err
    rms_ss = float(np.sqrt(np.mean(err_ss**2)))
    mx_ss = float(err_ss.max())
    # Jerk: 3rd derivative of EE position, finite-differenced.
    dt = 1.0 / control_hz
    vel = np.diff(traces["ee_pos"], axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    rms_jerk = float(np.sqrt(np.mean(np.linalg.norm(jerk, axis=1) ** 2)))
    return {
        "rms_m": rms_full,
        "max_m": mx_full,
        "rms_steady_m": rms_ss,
        "max_steady_m": mx_ss,
        "rms_jerk_m_per_s3": rms_jerk,
    }


def _plot_projection(
    traces: dict[str, np.ndarray],
    path: Path,
    axes: tuple[int, int],
    labels: tuple[str, str],
) -> None:
    ee = traces["ee_pos"]
    tg = traces["target"]
    a, b = axes
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(tg[:, a], tg[:, b], label="target", linewidth=2, alpha=0.6)
    ax.plot(ee[:, a], ee[:, b], label="achieved", linewidth=1)
    ax.set_xlabel(f"{labels[0]} (m)")
    ax.set_ylabel(f"{labels[1]} (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"End-effector trace ({labels[0]}-{labels[1]} plane)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_yz_trace(traces: dict[str, np.ndarray], path: Path) -> None:
    _plot_projection(traces, path, axes=(1, 2), labels=("y", "z"))


def plot_xz_trace(traces: dict[str, np.ndarray], path: Path) -> None:
    _plot_projection(traces, path, axes=(0, 2), labels=("x", "z"))


def plot_error(traces: dict[str, np.ndarray], control_hz: float, path: Path) -> None:
    err = np.linalg.norm(traces["ee_pos"] - traces["target"], axis=1) * 1000.0
    t = np.arange(len(err)) / control_hz
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, err, linewidth=1)
    ax.axhline(10.0, color="C3", linestyle="--", linewidth=1, label="1 cm target")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tracking error (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("EE tracking error over time")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="trajectory name or YAML path")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "defaults to checkpoints/<traj>/best/best_model.zip "
            "(falls back to ppo_panda_final.zip if best/ is missing)"
        ),
    )
    parser.add_argument("--periods", type=float, default=3.0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kind = str(cfg.get("trajectory", {}).get("kind", "circle"))
    name = str(cfg.get("name", kind))
    plots_dir = RESULTS / name / "plots"
    videos_dir = RESULTS / name / "videos"
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        best = REPO / "checkpoints" / name / "best" / "best_model.zip"
        checkpoint = str(
            best if best.exists() else REPO / "checkpoints" / name / "ppo_panda_final.zip"
        )

    env = make_env(cfg, seed=0, apply_wrappers=True)
    panda = _unwrap_to_panda(env)
    control_hz = panda.cfg.control_hz
    period_s = panda.cfg.trajectory_period_s
    n_steps = int(round(args.periods * period_s * control_hz))
    # Stay within one episode so we don't get a synthetic mid-rollout transient.
    n_steps = min(n_steps, panda.cfg.max_steps)

    print(f"[eval] traj={kind} checkpoint={checkpoint} steps={n_steps}")
    model = PPO.load(checkpoint, env=None, device="cpu")

    traces = rollout(model, env, n_steps=n_steps)
    m = metrics(traces, control_hz=control_hz)

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_yz_trace(traces, plots_dir / "yz_trace.png")
    plot_error(traces, control_hz, plots_dir / "error_vs_time.png")
    # x is only informative when the trajectory actually moves in depth.
    if (traces["target"][:, 0].max() - traces["target"][:, 0].min()) > 1e-3:
        plot_xz_trace(traces, plots_dir / "xz_trace.png")

    if not args.no_video:
        videos_dir.mkdir(parents=True, exist_ok=True)
        render_video(env, model, n_steps=n_steps, out_path=videos_dir / "rollout.mp4")

    print(
        f"[eval] RMS={m['rms_m'] * 1000:.2f} mm  MAX={m['max_m'] * 1000:.2f} mm  "
        f"steady(t>1s) RMS={m['rms_steady_m'] * 1000:.2f} mm  "
        f"MAX={m['max_steady_m'] * 1000:.2f} mm  "
        f"RMS_jerk={m['rms_jerk_m_per_s3']:.1f} m/s^3"
    )


if __name__ == "__main__":
    main()
