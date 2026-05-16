"""PPO training for Kinesis.

Usage:
    uv run python scripts/train.py --smoke                     # VecEnv smoke only
    uv run python scripts/train.py --timesteps 200000          # short pilot (circle)
    uv run python scripts/train.py --config figure8_3d         # train on figure-8
    uv run python scripts/train.py                             # full circle run

TensorBoard logs to logs/tb/<traj>/, checkpoints to checkpoints/<traj>/, best
model to checkpoints/<traj>/best/. Console gets a per-rollout summary line
including the running mean episodic EE-tracking RMS so the pilot can be eyeballed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from kinesis.envs.factory import env_thunk, load_config

REPO = Path(__file__).resolve().parents[1]


def _traj_dirs(cfg: dict) -> tuple[Path, Path, Path]:
    kind = str(cfg.get("trajectory", {}).get("kind", "circle"))
    name = str(cfg.get("name", kind))
    log_dir = REPO / "logs" / "tb" / name
    ckpt_dir = REPO / "checkpoints" / name
    return log_dir, ckpt_dir, ckpt_dir / "best"


def _build_vec(cfg: dict, n_envs: int, use_subproc: bool, seed_base: int = 0):
    thunks = [env_thunk(cfg, seed=seed_base + i) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    vec = vec_cls(thunks)
    return VecMonitor(vec)


def smoke(cfg: dict, n_envs: int, n_steps: int, use_subproc: bool) -> None:
    vec = _build_vec(cfg, n_envs=n_envs, use_subproc=use_subproc)
    try:
        vec.reset()
        rng = np.random.default_rng(0)
        act_shape = (n_envs, vec.action_space.shape[0])
        t0 = time.perf_counter()
        for _ in range(n_steps):
            a = rng.uniform(-1.0, 1.0, size=act_shape).astype(np.float32)
            vec.step(a)
        dt = time.perf_counter() - t0
        total = n_steps * n_envs
        print(
            f"vec={type(vec.venv).__name__} n_envs={n_envs} n_steps={n_steps} "
            f"wall={dt:.2f}s total_env_steps={total} "
            f"throughput={total / dt:.0f} env-steps/s"
        )
    finally:
        vec.close()


class TrackingErrorCallback(BaseCallback):
    """Logs the rolling mean per-step EE tracking error to TensorBoard."""

    def __init__(self, window: int = 2048) -> None:
        super().__init__()
        self._buf: list[float] = []
        self._window = window

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            err = info.get("ee_error_m")
            if err is not None:
                self._buf.append(float(err))
                if len(self._buf) > self._window:
                    self._buf = self._buf[-self._window :]
        if self._buf and self.num_timesteps % 2048 == 0:
            arr = np.asarray(self._buf)
            self.logger.record("rollout/ee_error_mean_m", float(arr.mean()))
            self.logger.record("rollout/ee_error_rms_m", float(np.sqrt((arr**2).mean())))
        return True


def train(cfg: dict, timesteps: int, use_subproc: bool, device: str) -> None:
    ppo_cfg = cfg.get("ppo", {})
    n_envs = int(ppo_cfg.get("n_envs", 16))

    log_dir, ckpt_dir, best_dir = _traj_dirs(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    vec = _build_vec(cfg, n_envs=n_envs, use_subproc=use_subproc, seed_base=0)
    eval_vec = _build_vec(cfg, n_envs=1, use_subproc=False, seed_base=10_000)

    net_arch = ppo_cfg.get("net_arch")
    policy_kwargs = {"net_arch": list(net_arch)} if net_arch is not None else None

    model = PPO(
        policy="MlpPolicy",
        env=vec,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
        device=device,
        seed=0,
    )

    callbacks = CallbackList(
        [
            TrackingErrorCallback(),
            CheckpointCallback(
                save_freq=max(200_000 // n_envs, 1),
                save_path=str(ckpt_dir),
                name_prefix="ppo_panda",
            ),
            EvalCallback(
                eval_vec,
                best_model_save_path=str(best_dir),
                log_path=str(log_dir / "eval"),
                eval_freq=max(50_000 // n_envs, 1),
                n_eval_episodes=3,
                deterministic=True,
                render=False,
            ),
        ]
    )

    print(
        f"[train] timesteps={timesteps} n_envs={n_envs} device={device} "
        f"vec={type(vec.venv).__name__}"
    )
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    finally:
        vec.close()
        eval_vec.close()
    dt = time.perf_counter() - t0
    final_path = ckpt_dir / "ppo_panda_final.zip"
    model.save(str(final_path))
    print(f"[train] done in {dt:.1f}s — saved {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="VecEnv smoke test only")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="override ppo.total_timesteps from config",
    )
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=200, help="smoke: steps per env")
    parser.add_argument("--dummy", action="store_true", help="Use DummyVecEnv (debug fallback)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--config",
        default=None,
        help="trajectory name (e.g. 'circle', 'figure8_3d') or path to a YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_envs = args.n_envs or int(cfg.get("ppo", {}).get("n_envs", 16))

    if args.smoke:
        smoke(cfg, n_envs=n_envs, n_steps=args.n_steps, use_subproc=not args.dummy)
        return

    timesteps = args.timesteps or int(cfg.get("ppo", {}).get("total_timesteps", 2_000_000))
    train(cfg, timesteps=timesteps, use_subproc=not args.dummy, device=args.device)


if __name__ == "__main__":
    main()
