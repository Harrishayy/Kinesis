"""PPO training entry point for Kinesis (M5/M6).

For now this only implements the M5 smoke path:
    uv run python scripts/train.py --smoke
which builds a SubprocVecEnv with the full wrapper stack, runs 200 random
steps across all workers, prints throughput, and exits. PPO is wired in M6.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from kinesis.envs.factory import env_thunk, load_config


def smoke(cfg: dict, n_envs: int, n_steps: int, use_subproc: bool) -> None:
    thunks = [env_thunk(cfg, seed=i) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec = vec_cls(thunks)
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
            f"vec={vec_cls.__name__} n_envs={n_envs} n_steps={n_steps} "
            f"wall={dt:.2f}s total_env_steps={total} "
            f"throughput={total / dt:.0f} env-steps/s"
        )
    finally:
        vec.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="VecEnv smoke test only")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument(
        "--dummy", action="store_true", help="Use DummyVecEnv (debug fallback)"
    )
    args = parser.parse_args()

    cfg = load_config()
    n_envs = args.n_envs or int(cfg.get("ppo", {}).get("n_envs", 16))

    if args.smoke:
        smoke(cfg, n_envs=n_envs, n_steps=args.n_steps, use_subproc=not args.dummy)
        return

    raise SystemExit("PPO training arrives in M6. Use --smoke for now.")


if __name__ == "__main__":
    main()
