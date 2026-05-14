"""Run the trained policy in MuJoCo's live (passive) viewer.

Opens a draggable interactive window with the Panda tracking the trajectory
under the trained policy. Same env + wrappers as eval.

Usage:
    uv run python scripts/play.py
    uv run python scripts/play.py --checkpoint checkpoints/best/best_model.zip

Window controls (MuJoCo viewer defaults):
- Drag with right-mouse to rotate camera
- Drag with shift+right-mouse to pan
- Scroll to zoom
- Space to pause
- Close the window to exit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from kinesis.envs.factory import load_config, make_env
from kinesis.envs.panda_track import PandaTrackEnv

REPO = Path(__file__).resolve().parents[1]


def _unwrap_to_panda(env) -> PandaTrackEnv:
    while not isinstance(env, PandaTrackEnv):
        env = env.env
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="trajectory name or YAML path")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="defaults to checkpoints/<traj>/best/best_model.zip",
    )
    parser.add_argument(
        "--no-wrappers",
        action="store_true",
        help="Run on a clean env (no obs noise / control delay)",
    )
    parser.add_argument("--realtime", action="store_true", help="Sleep to wall-clock")
    args = parser.parse_args()

    cfg = load_config(args.config)
    kind = str(cfg.get("trajectory", {}).get("kind", "circle"))
    checkpoint = checkpoint or str(
        REPO / "checkpoints" / kind / "best" / "best_model.zip"
    )
    env = make_env(cfg, seed=0, apply_wrappers=not args.no_wrappers)
    panda = _unwrap_to_panda(env)
    control_dt = 1.0 / panda.cfg.control_hz

    model = PPO.load(checkpoint, device="cpu")
    obs, _ = env.reset(seed=0)
    print(
        f"[play] checkpoint={checkpoint}  "
        f"wrappers={'on' if not args.no_wrappers else 'off'}  "
        f"control_dt={control_dt*1000:.0f}ms"
    )
    print("[play] window controls: right-drag rotate, shift+right-drag pan, "
          "scroll zoom, space pause")

    with mujoco.viewer.launch_passive(panda.model, panda.data) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            if terminated or truncated:
                obs, _ = env.reset(seed=0)
            if args.realtime:
                elapsed = time.perf_counter() - t0
                slack = control_dt - elapsed
                if slack > 0:
                    time.sleep(slack)


if __name__ == "__main__":
    main()
