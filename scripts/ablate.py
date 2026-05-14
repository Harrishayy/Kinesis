"""Ablation: evaluate the trained policy under four uncertainty conditions.

For each of {clean, noise-only, delay-only, noise+delay} we run one
deterministic episode and report full / steady-state RMS, max, and jerk.

Writes a markdown table to results/ablation.md.

Usage:
    uv run python scripts/ablate.py
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from kinesis.envs.factory import load_config, make_env
from kinesis.envs.panda_track import PandaTrackEnv

REPO = Path(__file__).resolve().parents[1]


def _unwrap_to_panda(env) -> PandaTrackEnv:
    while not isinstance(env, PandaTrackEnv):
        env = env.env
    return env


def rollout_metrics(env, model: PPO, n_steps: int, settle_s: float = 1.0) -> dict:
    panda = _unwrap_to_panda(env)
    control_hz = panda.cfg.control_hz
    obs, _ = env.reset(seed=0)
    ee_pos = np.zeros((n_steps, 3))
    target = np.zeros((n_steps, 3))
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        ee_pos[i] = info["ee_pos"]
        target[i] = info["target"]
        if term or trunc:
            ee_pos = ee_pos[: i + 1]
            target = target[: i + 1]
            break
    err = np.linalg.norm(ee_pos - target, axis=1)
    settle = int(round(settle_s * control_hz))
    err_ss = err[settle:] if err.size > settle else err
    dt = 1.0 / control_hz
    vel = np.diff(ee_pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return {
        "rms_mm": float(np.sqrt(np.mean(err**2)) * 1000),
        "max_mm": float(err.max() * 1000),
        "rms_steady_mm": float(np.sqrt(np.mean(err_ss**2)) * 1000),
        "max_steady_mm": float(err_ss.max() * 1000),
        "rms_jerk": float(np.sqrt(np.mean(np.linalg.norm(jerk, axis=1) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="trajectory name or YAML path")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="defaults to checkpoints/<traj>/best/best_model.zip",
    )
    parser.add_argument("--periods", type=float, default=2.0)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    kind = str(base_cfg.get("trajectory", {}).get("kind", "circle"))
    checkpoint = checkpoint or str(
        REPO / "checkpoints" / kind / "best" / "best_model.zip"
    )
    panda = _unwrap_to_panda(make_env(base_cfg, seed=0, apply_wrappers=False))
    n_steps = min(
        int(round(args.periods * panda.cfg.trajectory_period_s * panda.cfg.control_hz)),
        panda.cfg.max_steps,
    )
    model = PPO.load(checkpoint, device="cpu")
    print(f"[ablate] traj={kind} checkpoint={checkpoint}")

    conditions: list[tuple[str, dict]] = [
        ("clean",        {"obs_noise_sigma_m": 0.0,  "action_delay_steps": 0}),
        ("noise only",   {"obs_noise_sigma_m": 0.02, "action_delay_steps": 0}),
        ("delay only",   {"obs_noise_sigma_m": 0.0,  "action_delay_steps": 2}),
        ("noise+delay",  {"obs_noise_sigma_m": 0.02, "action_delay_steps": 2}),
    ]

    rows = []
    print(f"[ablate] checkpoint={checkpoint} steps={n_steps} per condition")
    for label, w_override in conditions:
        cfg = copy.deepcopy(base_cfg)
        cfg["wrappers"] = w_override
        env = make_env(cfg, seed=0, apply_wrappers=True)
        m = rollout_metrics(env, model, n_steps=n_steps)
        rows.append((label, m))
        print(
            f"  {label:14s}  RMS={m['rms_mm']:6.2f}mm  MAX={m['max_mm']:6.2f}mm  "
            f"steady_RMS={m['rms_steady_mm']:6.2f}mm  "
            f"steady_MAX={m['max_steady_mm']:6.2f}mm  "
            f"jerk={m['rms_jerk']:6.1f}"
        )

    out = REPO / "results" / kind / "ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(
            f"# Uncertainty ablation ({kind})\n\n"
            f"Deterministic policy from `{checkpoint}` evaluated on "
            f"{n_steps} steps ({n_steps / panda.cfg.control_hz:.1f} s) of "
            f"{kind} tracking under four uncertainty conditions.\n\n"
            "| Condition | Full RMS (mm) | Full MAX (mm) | "
            "Steady RMS (mm) | Steady MAX (mm) | RMS jerk (m/s³) |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for label, m in rows:
            f.write(
                f"| {label} | {m['rms_mm']:.2f} | {m['max_mm']:.2f} | "
                f"{m['rms_steady_mm']:.2f} | {m['max_steady_mm']:.2f} | "
                f"{m['rms_jerk']:.1f} |\n"
            )
        f.write(
            "\nObservation noise σ = 2 cm (applied to EE position in obs only). "
            "Control delay = 2 steps × 20 ms = 40 ms.\n"
        )
    print(f"[ablate] wrote {out}")


if __name__ == "__main__":
    main()
