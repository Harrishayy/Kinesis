"""Render multi-camera videos of a trained policy rolling out one trajectory.

Each camera produces its own video file. All cameras see the same deterministic
rollout — every step is rendered once per camera against the same physics state,
so frame N is the same moment of policy execution across all views.

Output layout:
    results/videos/<name>/rollout_<view>.mp4

Cameras (named):
    side    — 3/4 angle from the operator's right
    front   — looking at the EE from the workspace-side (along world +x)
    bottom  — from underneath, looking up at the gripper
    top     — straight down

Usage:
    uv run python scripts/render_views.py --config viviani_residual
    uv run python scripts/render_views.py --config viviani_residual \\
        --views front,bottom
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO

from kinesis.envs.factory import load_config, make_env
from kinesis.envs.panda_track import PandaTrackEnv

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 50


@dataclass(frozen=True)
class ViewSpec:
    name: str
    azimuth: float
    elevation: float
    distance: float


# Workspace centre for the Panda configs is (0.5, 0, 0.4). All views orbit
# around that point so the camera framing is consistent across rollouts.
DEFAULT_LOOKAT = np.array([0.5, 0.0, 0.4], dtype=np.float64)

# MuJoCo angle convention (verified empirically):
#   azimuth = 180 puts the camera in the +x half-space looking back toward the
#   base; azimuth = 90 puts it in the -y half-space looking toward +y.
#   elevation < 0  → camera above the lookat point, looking down.
#   elevation > 0  → camera below the lookat point, looking up.
_VIEWS: dict[str, ViewSpec] = {
    "side": ViewSpec("side", azimuth=135.0, elevation=-15.0, distance=1.6),
    "front": ViewSpec("front", azimuth=180.0, elevation=-8.0, distance=1.2),
    "bottom": ViewSpec("bottom", azimuth=180.0, elevation=70.0, distance=0.9),
    "top": ViewSpec("top", azimuth=180.0, elevation=-85.0, distance=0.9),
}


def _make_camera(spec: ViewSpec) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = spec.azimuth
    cam.elevation = spec.elevation
    cam.distance = spec.distance
    cam.lookat = DEFAULT_LOOKAT.copy()
    return cam


def _add_marker(scene, *, pos, size, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.asarray(size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_arrow(scene, *, origin, direction, length, radius, rgba) -> None:
    """Arrow geom along `direction` (will be normalised), length `length`.
    Mirrors `_add_arrow` in scripts/eval.py — kept duplicated so the tools/
    directory stays independent of the main script."""
    if scene.ngeom >= scene.maxgeom:
        return
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return
    z = d / n
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = up - z * (up @ z)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z])
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([radius, radius, float(length)], dtype=np.float64),
        pos=np.asarray(origin, dtype=np.float64),
        mat=mat.flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _unwrap_to_panda(env) -> PandaTrackEnv:
    while not isinstance(env, PandaTrackEnv):
        env = env.env
    return env


def render_multiview(
    env,
    model: PPO,
    n_steps: int,
    views: list[ViewSpec],
    out_dir: Path,
) -> None:
    panda = _unwrap_to_panda(env)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderers = {
        v.name: mujoco.Renderer(panda.model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH) for v in views
    }
    cameras = {v.name: _make_camera(v) for v in views}
    writers = {
        v.name: imageio.get_writer(
            str(out_dir / f"rollout_{v.name}.mp4"), fps=VIDEO_FPS, codec="libx264"
        )
        for v in views
    }

    trail_ts = np.linspace(0.0, panda.trajectory.period_s, 96, endpoint=False)
    trail_pts = [panda.trajectory.target(float(t)) for t in trail_ts]
    ee_history: list[np.ndarray] = []

    has_orient = panda.cfg.include_orientation
    arrow_length = 0.30
    arrow_radius_target = 0.009
    arrow_radius_realised = 0.006

    try:
        obs, _ = env.reset(seed=0)
        for _ in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ee = np.asarray(info["ee_pos"], dtype=np.float64)
            ee_history.append(ee)
            current_target = panda.trajectory.target(panda._t())
            R_ee = np.asarray(info["R_ee"], dtype=np.float64) if has_orient else None
            R_target = np.asarray(info["R_target"], dtype=np.float64) if has_orient else None
            for name, renderer in renderers.items():
                renderer.update_scene(panda.data, camera=cameras[name])
                scene = renderer.scene
                for pt in trail_pts:
                    _add_marker(scene, pos=pt, size=[0.0035, 0, 0], rgba=[0.30, 0.55, 1.0, 0.45])
                for pt in ee_history:
                    _add_marker(scene, pos=pt, size=[0.0035, 0, 0], rgba=[1.0, 0.85, 0.10, 0.9])
                _add_marker(
                    scene, pos=current_target, size=[0.013, 0, 0], rgba=[1.0, 0.15, 0.15, 1.0]
                )
                if has_orient:
                    _add_arrow(
                        scene,
                        origin=ee,
                        direction=R_target[:, 0],
                        length=arrow_length,
                        radius=arrow_radius_target,
                        rgba=[1.0, 0.20, 0.85, 0.90],
                    )
                    _add_arrow(
                        scene,
                        origin=ee,
                        direction=R_ee[:, 0],
                        length=arrow_length,
                        radius=arrow_radius_realised,
                        rgba=[0.10, 0.85, 1.0, 1.00],
                    )
                writers[name].append_data(renderer.render())
            if terminated or truncated:
                obs, _ = env.reset(seed=0)
                ee_history.clear()
    finally:
        for r in renderers.values():
            r.close()
        for w in writers.values():
            w.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="viviani_residual", help="trajectory config name or YAML path"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="defaults to checkpoints/<name>/best/best_model.zip"
    )
    parser.add_argument(
        "--periods", type=float, default=3.0, help="how many trajectory periods to roll out"
    )
    parser.add_argument(
        "--views",
        default="side,front,bottom,top",
        help=f"comma-separated subset of {sorted(_VIEWS.keys())}",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = str(cfg.get("name", cfg.get("trajectory", {}).get("kind", "circle")))
    checkpoint = args.checkpoint or str(REPO / "checkpoints" / name / "best" / "best_model.zip")

    env = make_env(cfg, seed=0, apply_wrappers=True)
    panda = _unwrap_to_panda(env)
    n_steps = min(
        int(round(args.periods * panda.cfg.trajectory_period_s * panda.cfg.control_hz)),
        panda.cfg.max_steps,
    )

    selected = []
    for v in args.views.split(","):
        v = v.strip()
        if v not in _VIEWS:
            raise SystemExit(f"unknown view {v!r}; choices: {sorted(_VIEWS)}")
        selected.append(_VIEWS[v])

    print(
        f"[render_views] config={name} checkpoint={checkpoint} steps={n_steps} views={[v.name for v in selected]}"
    )
    model = PPO.load(checkpoint, env=None, device="cpu")
    out_dir = RESULTS / name / "videos"
    render_multiview(env, model, n_steps=n_steps, views=selected, out_dir=out_dir)
    for v in selected:
        print(f"  wrote {out_dir / f'rollout_{v.name}.mp4'}")


if __name__ == "__main__":
    main()
