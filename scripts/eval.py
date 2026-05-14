"""Deterministic evaluation of a trained Kinesis PPO policy.

Outputs:
- results/plots/yz_trace.png        — target vs achieved EE trace (y-z plane)
- results/plots/error_vs_time.png   — tracking error magnitude vs time
- results/videos/rollout.mp4        — offscreen-rendered rollout video
- prints a one-line metric summary (RMS / max / jerk)

Usage:
    uv run python scripts/eval.py
    uv run python scripts/eval.py --checkpoint checkpoints/best/best_model.zip
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
PLOTS = RESULTS / "plots"
VIDEOS = RESULTS / "videos"

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


def render_video(env, model: PPO, n_steps: int, out_path: Path) -> None:
    panda = _unwrap_to_panda(env)
    renderer = mujoco.Renderer(panda.model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=0)
    with imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec="libx264") as writer:
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            renderer.update_scene(panda.data, camera=0 if panda.model.ncam else -1)
            writer.append_data(renderer.render())
            if terminated or truncated:
                obs, _ = env.reset(seed=0)
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


def plot_yz_trace(traces: dict[str, np.ndarray], path: Path) -> None:
    ee = traces["ee_pos"]
    tg = traces["target"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(tg[:, 1], tg[:, 2], label="target", linewidth=2, alpha=0.6)
    ax.plot(ee[:, 1], ee[:, 2], label="achieved", linewidth=1)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("End-effector trace (y-z plane)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "checkpoints" / "ppo_panda_final.zip"),
    )
    parser.add_argument("--periods", type=float, default=3.0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    env = make_env(cfg, seed=0, apply_wrappers=True)
    panda = _unwrap_to_panda(env)
    control_hz = panda.cfg.control_hz
    period_s = panda.cfg.trajectory_period_s
    n_steps = int(round(args.periods * period_s * control_hz))
    # Stay within one episode so we don't get a synthetic mid-rollout transient.
    n_steps = min(n_steps, panda.cfg.max_steps)

    print(f"[eval] checkpoint={args.checkpoint} steps={n_steps}")
    model = PPO.load(args.checkpoint, env=None, device="cpu")

    traces = rollout(model, env, n_steps=n_steps)
    m = metrics(traces, control_hz=control_hz)

    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_yz_trace(traces, PLOTS / "yz_trace.png")
    plot_error(traces, control_hz, PLOTS / "error_vs_time.png")

    if not args.no_video:
        VIDEOS.mkdir(parents=True, exist_ok=True)
        render_video(env, model, n_steps=n_steps, out_path=VIDEOS / "rollout.mp4")

    print(
        f"[eval] RMS={m['rms_m']*1000:.2f} mm  MAX={m['max_m']*1000:.2f} mm  "
        f"steady(t>1s) RMS={m['rms_steady_m']*1000:.2f} mm  "
        f"MAX={m['max_steady_m']*1000:.2f} mm  "
        f"RMS_jerk={m['rms_jerk_m_per_s3']:.1f} m/s^3"
    )


if __name__ == "__main__":
    main()
