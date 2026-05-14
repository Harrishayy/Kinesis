# Kinesis

[![CI](https://github.com/Harrishayy/Kinesis/actions/workflows/ci.yml/badge.svg)](https://github.com/Harrishayy/Kinesis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Reinforcement-learning end-effector trajectory tracking for the **Franka Emika Panda** in MuJoCo. A PPO policy follows a smooth Cartesian reference (circle, 15 cm radius at 0.25 Hz) under observation noise and control delay.

<p align="center">
  <img src="results/plots/yz_trace.png" width="320"/>
  <img src="results/plots/error_vs_time.png" width="480"/>
</p>

**Result:** 5.5 mm steady-state RMS, 11.9 mm max — well under the 1 cm target — at 0.25 Hz on a 15 cm circle, under ±2 cm observation noise and 2-step control delay. Trained in ~7 min on a MacBook Pro M5 Pro (CPU).

---

## Quickstart

```bash
git clone https://github.com/Harrishayy/Kinesis.git
cd Kinesis
make setup           # uv venv + editable install + pre-commit
make test            # run the test suite (should print 18 passed)
make smoke           # 200-step SubprocVecEnv throughput check
make train           # PPO, 2M steps, ~12 min on M5 Pro (CPU)
make eval            # plots + MP4 to results/, prints RMS/max/jerk
```

Open `logs/tb/` in TensorBoard to watch training curves
(`rollout/ee_error_rms_m`, `eval/mean_reward`).

## Project structure

```
.
├── src/kinesis/        # Python package (envs, configs, utils)
├── tests/              # pytest suite
├── scripts/            # public, reproducible entrypoints (smoke, render)
├── assets/             # vendored mujoco_menagerie Panda assets
├── docs/               # design notes and figures
└── results/            # curated plots and videos
```

## Design

- **Simulator:** MuJoCo (open-source Python bindings).
- **Robot:** Franka Emika Panda from `mujoco_menagerie`.
- **RL:** stable-baselines3 PPO, 16-worker `SubprocVecEnv`.
- **Action space:** joint position deltas (∆q), clipped to a safe per-step range.
- **Observation:** joint state, end-effector position, current + lookahead trajectory targets, phase variable, previous action.
- **Reward:** weighted L2 tracking error + action-rate and joint-velocity smoothness penalties.
- **Uncertainty:** observation noise + control delay (both, configurable).

Full design notes live in [`docs/design.md`](docs/design.md).

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and code-style notes.

## License

[MIT](LICENSE).
