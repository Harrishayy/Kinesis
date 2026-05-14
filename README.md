# Kinesis

[![CI](https://github.com/Harrishayy/Kinesis/actions/workflows/ci.yml/badge.svg)](https://github.com/Harrishayy/Kinesis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

Reinforcement-learning end-effector trajectory tracking for the **Franka Emika Panda** in MuJoCo. The policy learns to follow time-varying Cartesian paths (circles, figure-eights, lissajous, moving targets) under observation noise and control delay.

> _Status: scaffold — environment, training, and evaluation land in subsequent milestones._

<!-- Headline media: replace with final video / GIF after M5. -->
<p align="center">
  <em>tracking demo video — coming soon</em>
</p>

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/Harrishayy/Kinesis.git
cd Kinesis
make setup     # uv venv + editable install + submodules + pre-commit
make smoke     # 10k-step PPO sanity check (available after M1)
```

Run the full pipeline:

```bash
make train     # PPO training, logs to TensorBoard
make eval      # load latest checkpoint, write plots + MP4 to results/
```

## Project structure

```
.
├── src/kinesis/        # Python package (envs, configs, utils)
├── tests/              # pytest suite
├── scripts/            # public, reproducible entrypoints (smoke, render)
├── assets/             # mujoco_menagerie (git submodule)
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
