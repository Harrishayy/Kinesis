# Design Note — Kinesis

End-effector trajectory tracking on a Franka Emika Panda in MuJoCo, trained with PPO. This note documents the decisions behind the system, the evaluation methodology, and the ablations.

## Problem

Given a smooth Cartesian reference trajectory in workspace, drive the Panda's end-effector to track it with low position error and smooth motion, under realistic perception and actuation uncertainty.

## State, action, reward

### Observation (51-D)
The policy sees a single concatenated vector:

| Field            | Dim  | Notes                                                      |
| ---------------- | ---- | ---------------------------------------------------------- |
| `q`              | 7    | joint angles                                               |
| `q̇`             | 7    | joint velocities                                           |
| `ee_pos`         | 3    | end-effector position (the `hand` body)                    |
| `target`         | 3    | reference position at current time                         |
| `lookahead`      | 12   | 4 future reference positions, spaced 0.1 s apart            |
| `phase`          | 2    | `(sin, cos)` of trajectory phase — keeps policy stateless |
| `prev_action`    | 7    | last commanded action (helps the policy reason about Δa)   |

Targets and lookahead are passed in world frame. `phase` makes the policy
agnostic to which trajectory instance it is tracking — useful if/when we
swap circles for figure-eights or random Lissajous curves.

### Action (7-D)
Per-joint position **deltas**, `a ∈ [-1, 1]^7`, mapped to `Δq = a · 0.0873 rad`
(≈5° per 20 ms control step). The commanded target `q + Δq` is clipped to joint
limits before being written to the MuJoCo position actuators. Rationale: delta
commands keep training stable (no large jumps), bound the per-step motion
without needing a torque controller, and let the policy output a comfortable
unit-bounded action — exactly what PPO likes.

### Reward
```
r = -w_track · ‖ee − target‖²
    - w_action_rate · ‖a_t − a_{t−1}‖²
    - w_qdot · ‖q̇‖²
    + w_inband · 𝟙[‖ee − target‖ < 0.02]
```
with weights `(w_track, w_action_rate, w_qdot, w_inband) = (10, 0.1, 0.001, 0.5)`.
The squared tracking term gives a smooth gradient toward the target. The
action-rate term keeps the commanded motion physically smooth. The velocity
penalty is small but rules out the worst high-frequency chatter. The in-band
bonus is a shaping term — once you're close enough, *stay* close, don't
oscillate. Keeping the reward to four terms is deliberate; CLAUDE.md flags
"too many shaped terms" as a known failure mode.

## Trajectory representation

Circle of radius 0.15 m at 0.25 Hz (period 4 s), in the *y–z plane* at fixed
x = 0.5 m, centered at `(0.5, 0.0, 0.4)`. The circle was chosen for the headline
result because it has full-cycle predictability (good for visual evaluation of
smoothness), and because the y–z plane keeps the trajectory in a natural sweep
of the Panda's workspace. The home `qpos` was solved (`scripts/find_home_pose.py`)
by random-search in joint space starting from the model's `home` keyframe,
picking the qpos that minimizes ‖hand.xpos − center‖; the result places the EE
3.3 cm from the trajectory center.

## Training

- **Algorithm:** PPO (stable-baselines3 2.x), `MlpPolicy` (default 2×64 tanh).
- **Parallelism:** 16-way `SubprocVecEnv` on CPU. Throughput ≈ 20k env-steps/s
  without learning; learning-bottlenecked at ≈ 3k env-steps/s.
- **Hyperparameters:** lr 3e-4, n_steps 2048 per env, batch 64, 10 epochs,
  γ 0.99, λ 0.95. Defaults — no tuning was needed.
- **Total interaction:** 2,000,000 env-steps. Wall-clock: see results below.
- **Device:** CPU. With this network size, MPS shows no speed-up.

## Uncertainty model

Two stacking wrappers are applied during *both* training and evaluation,
deliberately — robustness is trained in, not bolted on:

- `ObsNoiseWrapper(σ=0.02 m)` — Gaussian noise on the *measured* EE position
  only. Targets and proprioception stay clean (the agent doesn't have noisy
  knowledge of where it's *going*, only where it currently *is*).
- `ActionDelayWrapper(k=2)` — applies the action commanded two control steps
  earlier. Initial queue is zeros, so the first 40 ms is no-op.

Going beyond the brief's "noise *or* delay" with both is intentional: a policy
that survives both is much closer to something you'd put on hardware.

## Evaluation

- **Protocol:** deterministic rollout of `model.predict(obs, deterministic=True)`
  for 3 trajectory periods (12 s, 600 control steps) under the same noise + delay
  wrappers used in training. Seed fixed.
- **Metrics:**
  - **RMS error** — primary metric. Cartesian distance between EE and target,
    integrated over the rollout, root-mean-square.
  - **Max error** — worst-case position error during the rollout.
  - **RMS jerk** — third derivative of EE position, magnitude RMS. Quantifies
    motion smoothness independent of position accuracy.
- **Artifacts:** `results/plots/yz_trace.png`, `results/plots/error_vs_time.png`,
  `results/videos/rollout.mp4`.

### Headline numbers
3 M PPO steps (≈ 7 min on M5 Pro, CPU), 16-way SubprocVecEnv, deterministic
eval over 10 s (= 2.5 trajectory periods) under obs-noise + control-delay
wrappers:

| Metric              | Value       | Notes                                           |
| ------------------- | ----------- | ----------------------------------------------- |
| **RMS error**       | **5.5 mm**  | steady state (`t > 1 s`, after startup)         |
| **Max error**       | **11.9 mm** | steady state                                    |
| Full-rollout RMS    | 14.6 mm     | includes the 137 mm startup transient at t = 0  |
| RMS jerk            | 161 m/s³    | smooth — qualitatively confirmed by video       |

The startup transient (home → first target) is inherent to the chosen home
pose; see the next section.

## What was tried and discarded

- **Torque / OSC action spaces** — rejected up-front. Joint-delta keeps the
  exploration manifold tight and the dynamics gentle. With torque, PPO spends
  the first ~500k steps just learning not to throw the arm around.
- **Direct joint-position targets (no delta)** — rejected for the same
  reason. Without per-step clipping, a single bad action causes a violent jump
  and the policy never recovers gradient signal.
- **More than 4 reward terms** — explicitly avoided. Adding an orientation
  term, a workspace-bounds term, etc. is tempting but tends to fight the
  primary tracking objective and inflates tuning surface area.
- **"On-trajectory" home pose to eliminate the startup transient.** Tried
  solving for a home `qpos` whose EE sits at `target(t=0)` so the rollout
  starts on the circle. This *did* remove the 137 mm spike but placed joint 4
  at −1.47 rad — close to its upper limit of −0.07 — which collapsed
  manipulability around the leftmost extreme of the trajectory and caused
  monotonic drift after the second revolution (median error grew from 10 mm
  early to >100 mm by 10 s). Reverted to a centered home pose with joint 4
  near the middle of its range; the startup transient is reported separately
  from steady-state RMS rather than hidden by a worse policy.

## Next steps

- Orientation tracking (4-D quaternion error → reward + obs).
- Trajectory diversity: train on a *family* of circles / figure-eights / random
  splines, with the trajectory parameters in the observation. Tests the policy's
  ability to *track* rather than memorise a single curve.
- Sim-to-real: domain randomisation on link masses, friction, and actuator gain.
