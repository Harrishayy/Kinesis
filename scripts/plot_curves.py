"""Read TensorBoard event files and save the learning-curve PNG.

Plots `rollout/ep_rew_mean` and `rollout/ee_error_rms_m` against
total_timesteps on a twin-axis figure. Writes results/plots/learning_curve.png.

Usage:
    uv run python scripts/plot_curves.py
    uv run python scripts/plot_curves.py --tb-dir logs/tb/PPO_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing import event_accumulator  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_scalar(acc: event_accumulator.EventAccumulator, tag: str):
    if tag not in acc.Tags().get("scalars", []):
        return None, None
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj",
        default="circle",
        help="trajectory subdir under logs/tb/ and results/plots/",
    )
    parser.add_argument("--tb-dir", default=None, help="TB run dir; default = latest")
    parser.add_argument(
        "--out",
        default=None,
        help="defaults to results/plots/<traj>/learning_curve.png",
    )
    args = parser.parse_args()

    if args.tb_dir is None:
        runs = sorted((REPO / "logs" / "tb" / args.traj).glob("PPO_*"))
        if not runs:
            raise SystemExit(f"no TB runs found under logs/tb/{args.traj}/")
        tb_dir = runs[-1]
    else:
        tb_dir = Path(args.tb_dir)
    out_path = Path(
        args.out or (REPO / "results" / "plots" / args.traj / "learning_curve.png")
    )
    print(f"[plot] reading {tb_dir}")

    acc = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # 0 = load all
    )
    acc.Reload()

    s_rew, v_rew = load_scalar(acc, "rollout/ep_rew_mean")
    s_err, v_err = load_scalar(acc, "rollout/ee_error_rms_m")
    s_eval, v_eval = load_scalar(acc, "eval/mean_reward")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))

    color_rew = "C0"
    if s_rew is not None:
        ax1.plot(s_rew / 1e6, v_rew, color=color_rew, alpha=0.6, label="rollout reward")
    if s_eval is not None:
        ax1.plot(
            s_eval / 1e6,
            v_eval,
            color=color_rew,
            linewidth=2.0,
            label="eval reward (deterministic)",
        )
    ax1.set_xlabel("training steps (M)")
    ax1.set_ylabel("episode reward", color=color_rew)
    ax1.tick_params(axis="y", labelcolor=color_rew)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_err = "C3"
    if s_err is not None:
        ax2.plot(
            s_err / 1e6,
            v_err * 1000.0,
            color=color_err,
            linewidth=1.2,
            label="rollout EE RMS",
        )
    ax2.set_ylabel("EE tracking RMS (mm)", color=color_err)
    ax2.tick_params(axis="y", labelcolor=color_err)
    ax2.axhline(10.0, color=color_err, linestyle="--", linewidth=0.8, alpha=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    ax1.set_title(f"PPO learning curve ({args.traj})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
