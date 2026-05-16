# scripts/

User-facing entrypoints. All accept `--config <name>` where `<name>` resolves to `src/kinesis/configs/{naive,residual}/<name>.yaml`.

- `train.py` -- PPO training; writes TensorBoard to `logs/tb/<name>/` and checkpoints to `checkpoints/<name>/`.
- `eval.py` -- deterministic rollout; writes plots to `results/<name>/plots/` and a side-view MP4 to `results/<name>/videos/`.
- `play.py` -- open the trained policy in MuJoCo's interactive viewer (`mjpython` required on macOS).

Utility scripts (ablations, curve plotting, multi-view rendering, IK home-pose search) live under `tools/`.
