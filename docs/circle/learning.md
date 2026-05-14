# Learning walkthrough: RL trajectory tracking with MuJoCo

A long-form, pedagogical companion to this repo. Walks through every concept the project depends on — MuJoCo basics, RL fundamentals, the Gymnasium + SB3 contract — and then through each design decision in this codebase, with reasoning and pointers to the relevant lines.

Read this once end-to-end before reading the code. Then read the code; everything will click.

---

## Table of contents

1. [MuJoCo fundamentals](#1-mujoco-fundamentals)
2. [RL fundamentals (just enough to use SB3)](#2-rl-fundamentals-just-enough)
3. [The Gymnasium and SB3 contracts](#3-gymnasium-and-sb3)
4. [The project, step by step](#4-the-project-step-by-step)
5. [How to view the simulation](#5-how-to-view-the-simulation)
6. [Where to look next — videos, papers, codebases](#6-where-to-look-next)

---

## 1. MuJoCo fundamentals

### 1.1 What MuJoCo is and isn't

MuJoCo (Multi-Joint dynamics with Contact) is a **rigid-body physics simulator** with a fast, well-conditioned implicit solver for contact dynamics. It is *not* a rendering engine, a learning library, or a robot SDK. It does exactly one thing well: given a description of a robot or scene, it integrates physics forward in time.

What you give MuJoCo:
- An **XML file** (`.xml` or `.mjcf`) describing the kinematic tree, geometry, joints, actuators, and contacts.

What MuJoCo gives you:
- A function (`mj_step`) that, called repeatedly, advances physics.
- Read access to every state quantity (joint angles, body positions, contact forces).
- A built-in renderer for visualisation.

This minimalism is a feature. Every higher-level abstraction (Gymnasium env, RL training loop, motion planner) is built on top of those primitives.

### 1.2 `MjModel` vs `MjData` — the most important distinction

```python
model = mujoco.MjModel.from_xml_path("scene.xml")  # static, loaded once
data  = mujoco.MjData(model)                        # dynamic, mutated each step
```

- **`MjModel`** — *static* description: link masses, inertia tensors, joint axes, actuator gains, joint limits, geometry. Read-only in practice. Stays the same for the whole episode.
- **`MjData`** — *dynamic* state: `qpos` (joint angles), `qvel` (joint velocities), `qacc`, every body's `xpos`/`xmat` in world frame, `ctrl` (the input you write each step), contact forces. Mutated by `mj_step`.

Mental model: `MjModel` is the schematic, `MjData` is the snapshot.

### 1.3 The XML format (MJCF)

A minimal scene:

```xml
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="arm">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
      <geom type="capsule" size="0.05 0.3"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="hinge" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
```

Key elements you will encounter in `panda.xml`:
- `<option timestep="...">` — the physics step size. Smaller is more accurate but slower.
- `<body>` — a rigid body in the kinematic tree. Has a pose, inertia, child bodies, and any number of geoms.
- `<joint>` — relative degree of freedom between a body and its parent. The Panda's `joint1..joint7` are all `<hinge>` joints with explicit `range` (joint limits, enforced by MuJoCo as soft constraints).
- `<geom>` — collision and visual geometry attached to a body.
- `<actuator>` — what you can command. The Panda uses `<general>` actuators in **position mode**: writing `ctrl[i]` sets a desired joint angle, and the actuator's internal PD bias drives the joint toward it.
- `<keyframe>` — named saved poses. The Panda XML defines a `home` keyframe we used as the seed for our home-pose search.
- `<site>` — a massless "marker" attached to a body. Useful for end-effector frames, sensors, attachment points. The Panda model doesn't define one for the gripper; we use the `hand` body's frame directly.

### 1.4 The simulation loop: `mj_step`

```python
data.ctrl[:] = my_action      # write inputs (actuator commands)
mujoco.mj_step(model, data)   # advance physics by model.opt.timestep
# now data.qpos, data.xpos, data.qvel etc. reflect the new state
```

That's the whole loop. Everything else — RL, control, visualisation — sits on top.

### 1.5 `env.step`, sim time vs control time, and `n_substeps`

This is the single most important detail in MuJoCo + RL. People skip past it; then their policies don't learn and they don't know why. Slow down on this one.

#### What "one step" means at each layer

There are **two clocks** running in your project. Knowing exactly which clock each piece of code lives in is most of the battle.

| Clock | What ticks | Tick rate | Where |
| --- | --- | --- | --- |
| **Sim clock** | `mujoco.mj_step` — one integration of the physics ODEs | 500 Hz (`opt.timestep = 0.002 s`) | inside MuJoCo |
| **Control clock** | `env.step` — one decision by the policy | 50 Hz (`control_dt = 0.020 s`) | the Gymnasium env |

The sim clock is fast because **physics is stiff**: smaller integrator steps → numerically stable contacts, joint limits, and actuator dynamics. Try setting `opt.timestep = 0.02` and the arm will jitter or explode the first time it touches anything.

The control clock is slow because **the policy doesn't need to react that fast**. A neural net firing every 20 ms is plenty for arm control. A policy at 500 Hz would have to learn a vastly tighter loop, and the per-step action would be 10× smaller, so it would take 10× more decisions to do anything useful.

So you **decouple them**: physics runs at its natural rate inside `env.step`, and the policy reads/writes once per call.

#### `n_substeps`: the bridge

```python
sim_dt     = model.opt.timestep        # 0.002 s — the physics clock
control_dt = 1.0 / 50.0                # 0.020 s — your policy clock
n_substeps = round(control_dt / sim_dt) # = 10  — physics ticks per policy tick
```

`n_substeps = 10` says: every time the policy makes one decision, run 10 physics steps before asking it for the next one.

#### What one call to `env.step(action)` actually does

Pseudocode, with the order of operations spelled out:

```python
def step(self, action):
    # (1) Map the policy's [-1, 1]^7 action to a joint-angle target.
    delta_q  = np.clip(action, -1, 1) * max_delta_rad   # ±5° per step
    target_q = np.clip(self.data.qpos[:7] + delta_q, joint_lo, joint_hi)

    # (2) Write the actuator command ONCE. This sits constant for the next 10 sim steps.
    self.data.ctrl[:7] = target_q

    # (3) Inner physics loop — n_substeps sim ticks, ctrl held fixed.
    for _ in range(self.n_substeps):
        mujoco.mj_step(self.model, self.data)

    # (4) Build the observation from the FINAL state (post-substep state).
    obs = self._obs()

    # (5) Compute reward from the same final state.
    reward = self._reward(...)

    # (6) Return to the agent. Agent now picks the next action; loop repeats.
    return obs, reward, terminated, truncated, info
```

Six numbered things. Notice what is **not** there:

- The policy is not consulted inside the inner loop. `action` is set once at step (2), and the actuator's PD bias chases that target for the full 20 ms.
- The policy does not see intermediate states. The states after substeps 1, 2, …, 9 happen, but they are never observed by the policy. Only the state after substep 10 — the **post-substep state** — is what `_obs()` reads.
- The reward is not averaged across substeps. It is computed once, from the post-substep state.

#### Concrete walkthrough, 200 ms

You call `env.reset()` and then `env.step(action_1)`. What happens in wall-clock terms:

- t = 0.000 s. `qpos` = home pose. Policy picks `action_1`.
- `data.ctrl[:7] = home_q + delta_q_1`. Held constant.
- Substep 1 (t → 0.002 s): physics integrates 2 ms. Joint angles move slightly toward target.
- Substep 2 (t → 0.004 s): another 2 ms.
- …
- Substep 10 (t → 0.020 s): final state.
- `obs_1 = _obs(post-substep state)` is returned. Policy sees `qpos` at t = 0.020 s.
- Policy now picks `action_2` based on `obs_1`. Cycle repeats.

You did **10 physics integrations** but the policy only made **1 decision**.

#### Implications you can feel

- **Actions are zero-order-held.** Your `ctrl` value is constant across all 10 sim substeps within an `env.step`. If you wanted a smoother command profile (interpolating between targets across substeps), you'd build it yourself — MuJoCo doesn't.
- **Per-step kinematic motion is bounded by the action × `control_dt`, not `sim_dt`.** Our ±5° max delta is "per `env.step`," which is one 20 ms control tick — not per 2 ms physics tick.
- **`n_substeps` is a knob.** If you raise the control rate to 100 Hz, `n_substeps` drops to 5, the policy reacts faster but commands are smaller per step. If you lower it to 25 Hz, `n_substeps` rises to 20, each command has more time to compound, and tracking gets harder.
- **`done` (truncated) is on the control clock.** Our `max_steps = 500` means 500 `env.step` calls → 500 × 20 ms = 10 s of sim time → 5 000 `mj_step` calls under the hood.

#### Where this lives in the code

`src/kinesis/envs/panda_track.py`:

- `__init__` computes `n_substeps` from `opt.timestep` and `control_hz`, with a guard if the math doesn't work out.
- `step` runs the action → `ctrl` → substep loop → obs → reward chain described above.

Read the `step` method once before you read anything else in this repo.

### 1.6 Actuators: position vs torque

The Panda model uses **position actuators**:

```xml
<general joint="joint1" gainprm="2000" biasprm="0 -2000 -200" .../>
```

Writing `data.ctrl[i] = θ_desired` sets a target angle; the actuator's gain and bias act like a stiff PD controller pulling the joint toward `θ_desired`. The arm is *stiff* — set ctrl to where you want the joint, and it gets there in a few sim steps.

Compare with **torque actuators** (`<motor>`): `ctrl[i]` is the torque directly applied at the joint, with no built-in stabilisation. The policy must learn to maintain its own balance.

For RL on a manipulator, **always start with position actuators**. The dynamics are easier; PPO learns 5× faster. Move to torque only if you have a specific reason (sim-to-real with compliant control, e.g.).

### 1.7 Reading positions: bodies, sites, geoms

```python
hand_pos = data.body("hand").xpos       # (3,) world-frame position
hand_rot = data.body("hand").xmat       # (9,) world-frame rotation matrix (flat)
hand_q   = data.body("hand").xquat      # (4,) world-frame quaternion (w, x, y, z)
```

`data.body(name)` returns a "view" into MjData's flat arrays — cheap. The same pattern works for `data.geom(name)`, `data.site(name)`, `data.joint(name)`.

The end-effector frame on the Panda is the `hand` body whose origin sits between the fingers. If you wanted a different EE point (e.g. 10 cm past the fingertips), the clean approach is to add a `<site>` in the XML and read `data.site(...).xpos`. We didn't — `hand.xpos` was sufficient.

After mutating `qpos` manually (e.g. on reset), call `mujoco.mj_forward(model, data)` to update derived quantities like `xpos` before reading them. We do this in `reset()`.

### 1.8 Rendering

MuJoCo ships an OpenGL renderer:

```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera=0)   # camera id or -1 for default
frame = renderer.render()               # → numpy array (H, W, 3) uint8
```

This is **offscreen** rendering — no window, no event loop. Perfect for saving videos. We use it in `scripts/eval.py` with `imageio` to write the mp4.

For an **interactive** window (drag the camera with the mouse), use `mujoco.viewer.launch_passive(model, data)` — covered in section 5.

### 1.9 Things that surprise people about MuJoCo

- **Joint limits leak.** They are enforced as soft constraints, so `qpos` can violate them by ~1e-3 rad when an actuator is driving hard into the stop. We loosened a test tolerance because of this — see `tests/test_env.py`.
- **`xpos` is stale after writing `qpos`.** Call `mj_forward` before reading kinematic quantities.
- **`data.ctrl` shape is `(model.nu,)`, not `(7,)`.** For the Panda with the gripper actuator counted, `nu == 8`. We slice `[:7]` and leave `ctrl[7]` untouched.
- **Contact dynamics are *implicit*.** That's why MuJoCo handles stiff contacts (a foot on the ground) without exploding. You don't need to do anything special.

---

## 2. RL fundamentals (just enough)

### 2.1 The MDP — what we're optimising

A reinforcement-learning problem is a **Markov decision process (MDP)**:
- States $s$ (or observations $o$, if partially observed).
- Actions $a$.
- Transition dynamics $p(s' \mid s, a)$ — given by MuJoCo here.
- Reward $r(s, a, s')$ — we *design* this.
- A discount factor $\gamma \in [0, 1)$.

Goal: find a **policy** $\pi(a \mid s)$ that maximises expected discounted return $\mathbb{E}\left[\sum_t \gamma^t r_t\right]$.

In our case:
- $s$ = the 51-D observation (joint state + EE + targets + history).
- $a$ = the 7-D delta-q action.
- $r$ = our hand-designed reward (tracking error + smoothness penalties + in-band bonus).
- $\gamma = 0.99$ — standard choice for continuous control.

### 2.2 Policy gradient in one paragraph

We parameterise $\pi$ as a neural network with parameters $\theta$. The **policy gradient theorem** says that the gradient of the expected return with respect to $\theta$ can be estimated by sampling trajectories from $\pi_\theta$ and computing $\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t$, where $A_t$ is the *advantage* of action $a_t$ over the policy's average at state $s_t$. Intuitively: actions that did better than average get their probability nudged up.

### 2.3 PPO in one paragraph

Vanilla policy gradient is high-variance and can take destructive steps. **Proximal Policy Optimisation (PPO)** stabilises training with two tricks:
1. **Clip the policy update.** Don't let the ratio $\pi_\theta(a)/\pi_{\theta_{\text{old}}}(a)$ move too far from 1, even if the advantage is huge.
2. **Reuse each rollout for multiple epochs.** Cheap, since the clip prevents the policy from drifting too far from where the data was collected.

This is *the* default algorithm for continuous control because it's simple, hard to break, and works on a wide range of tasks. Hence our locked choice.

### 2.4 Why PPO here, not SAC / DDPG / TD3

| Algorithm | Pros | Cons | Verdict for this task |
| --- | --- | --- | --- |
| **PPO** (on-policy) | Stable, simple, parallel-friendly | Sample-hungry | ✅ Sample cost is cheap in sim. |
| **SAC** (off-policy) | Very sample-efficient | Replay-buffer tuning, sensitivity to reward scale | Not needed; we have GPU-free 16-way sim. |
| **DDPG / TD3** | Sample-efficient | Brittle, finicky | Worse SAC. |

The locked design in `CLAUDE.md` says "PPO is enough and you can debug it" — that's the operative rule. Don't reach for an exotic algorithm when the boring one works.

### 2.5 On-policy vs off-policy

- **On-policy** (PPO): you can only learn from data collected by the *current* policy. Once you update, old data is stale.
- **Off-policy** (SAC, DQN): you keep a replay buffer of all past transitions and learn from them long after the policy has moved on.

On-policy is wasteful in terms of samples, but each sample is cheap here (one MuJoCo step), so it doesn't matter. On-policy plays *very* nicely with parallel rollouts because all 16 envs collect data with the same policy.

---

## 3. Gymnasium and SB3

### 3.1 What an env must implement

```python
class MyEnv(gymnasium.Env):
    observation_space: gymnasium.spaces.Space
    action_space: gymnasium.spaces.Space

    def reset(self, *, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
    def close(self): ...
```

That's the entire contract. `terminated=True` means "episode ended due to task outcome" (the robot fell, the puzzle is solved). `truncated=True` means "episode ended due to time budget." We never `terminate`, only `truncate` at 500 steps.

`info` is a free-form dict for diagnostics that don't go through the obs/reward channels — handy for logging.

### 3.2 Spaces

```python
spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)   # bounded continuous
spaces.Discrete(4)                                         # 4 categorical actions
spaces.Dict({...})                                         # composite (rarely used in SB3)
```

Continuous control = `Box`. SB3's PPO assumes the policy outputs a Gaussian over the Box; the policy network's output is the mean, with a learned diagonal log-std.

### 3.3 Wrappers — composable env modification

Gymnasium provides three subclasses to plug logic into the env without modifying it:

- **`ObservationWrapper`** — override `.observation(obs)` to transform every obs. Useful for noise injection, normalisation, masking. Used here for `ObsNoiseWrapper`.
- **`ActionWrapper`** — override `.action(a)` to transform every action before it reaches the env. Useful for action scaling, rescaling.
- **`Wrapper`** (general) — override `step`/`reset` arbitrarily. Used here for `ActionDelayWrapper` because we need state (a queue) across steps.

Wrappers stack:

```python
env = ActionDelayWrapper(ObsNoiseWrapper(PandaTrackEnv(), sigma_m=0.02), delay_steps=2)
```

Each wrapper exposes `.env` for the next layer down; the innermost env is `PandaTrackEnv`. To dig back to a specific layer (e.g. to read raw MuJoCo state), walk the chain — see `_unwrap_to_panda` in `scripts/eval.py`.

### 3.4 VecEnv — parallel rollouts

SB3 needs many parallel envs to feed PPO efficiently:

```python
vec = SubprocVecEnv([lambda: make_env() for _ in range(16)])
obs = vec.reset()                # (16, obs_dim)
obs, r, done, info = vec.step(a) # a: (16, action_dim)
```

`SubprocVecEnv` forks each env into a separate OS process — 16 CPUs busy. `DummyVecEnv` runs them sequentially in one process — slower, but easier to debug. Fall back to `DummyVecEnv` if your env doesn't pickle cleanly (a known macOS pain point with MuJoCo — though we didn't hit it).

PPO is on-policy: it sees `(16, T, ...)` of rollouts, computes advantages, then does several epochs of minibatch updates over them. The 16-way parallelism only affects throughput; the algorithm is identical.

---

## 4. The project, step by step

Each subsection corresponds to a milestone in the original plan (`/Users/harrishayyanar/.claude/plans/polished-weaving-goose.md`).

### 4.1 Home pose (M1) — `scripts/find_home_pose.py`

**Problem.** We need an initial `qpos` for every episode reset. The robot's `home` keyframe in the XML is a fine starting point, but its EE might not be near the trajectory.

**Approach.** Random-search in joint space starting from the `home` keyframe, biased by a small Gaussian noise (σ=0.5 rad). For each candidate, set `data.qpos[:7]`, call `mj_forward`, read `data.body("hand").xpos`, pick the one closest to the target point. 8 000 trials runs in seconds.

**Result.** EE 3.3 cm from `(0.5, 0, 0.4)` — close enough to the trajectory's centre that the arm only has to swing 15 cm to reach the first target.

**Why search instead of solve an IK?** You can; MuJoCo doesn't ship an IK solver but `mink` or `pinocchio` do. Random search is a *cheap one-shot* — we paste the result into the config and never run it again. IK would be overkill.

### 4.2 The Gymnasium env (M2) — `src/kinesis/envs/panda_track.py`

The single most important file in the repo. Walk it top to bottom:

1. `__init__`: load model, build `MjData`, compute `n_substeps`, instantiate the trajectory, read joint limits, define `observation_space` and `action_space`.
2. `_joint_limits`: read `[range]` from each joint via `model.joint(name)`.
3. `_obs`: assemble the 51-D vector. Order matters — wrappers reference fixed offsets.
4. `reset`: zero `MjData`, set `qpos = home + small noise`, zero `qvel`, call `mj_forward`, return initial obs.
5. `step`: clip action to [-1, 1], scale to delta-q, write `ctrl`, run `n_substeps` MuJoCo steps, compute reward, return.

### 4.3 Action design (M2) — why ∆q in [-1, 1]^7

**The choice.** Action ∈ [-1, 1]^7 (network-friendly), mapped to ∆q ∈ ±5° per step (physically sane), added to current joint angles, clipped to joint limits, written to position actuators.

**Why ∆q and not absolute q?** A policy outputting absolute q could try to teleport the joint, which is fine in steady state but devastating during exploration — gradients become useless.

**Why ±5° per step?** At 50 Hz, that's a max joint speed of 2.5 rad/s ≈ 143 deg/s. The Panda's true joint speed limits are several times that, so we're well within safety. Smaller delta → smoother motion but slower tracking; larger → faster but jerkier. ±5° is a known-good compromise.

**Why [-1, 1] and not directly in radians?** The policy network outputs a Gaussian with learned std. Standardising the action range to a fixed box makes that std meaningful across tasks.

### 4.4 Trajectory (M2) — `src/kinesis/utils/trajectory.py`

A pure function — zero state, zero MuJoCo dependence. `CircleTrajectory(center, radius, period)` exposes:

- `target(t)` — current target position.
- `lookahead(t, n, dt)` — stack of `n` future targets spaced `dt` apart.
- `phase_sin_cos(t)` — (sin, cos) of the phase angle.

Why separate from the env? Two reasons:
1. **Testability.** The trajectory has its own test suite (`tests/test_trajectory.py`). No MuJoCo or env machinery to mock.
2. **Swap-ability.** If you later want a figure-8 or a random spline, you write a new trajectory class — no changes to the env.

### 4.5 Observation design (M2)

```
[ q (7), qdot (7), ee_pos (3), target (3), lookahead (12), phase_sin_cos (2), prev_action (7) ]
```

Each piece justified:

- **`q` and `qdot`** — proprioception. The policy must know the robot's state.
- **`ee_pos`** — what we're trying to control. Without it, the policy would have to infer it from `q`, which is theoretically possible but wastes capacity.
- **`target`** — where to be *now*. The task signal.
- **`lookahead`** — where to be *soon*. Critical: a reactive policy (target only, no lookahead) is always behind, especially on fast-moving references. With 4 lookahead samples spaced 0.1 s apart, the policy sees 400 ms into the future.
- **`phase_sin_cos`** — keeps the obs continuous across loop boundaries. A raw angle would wrap from 2π to 0 every period; sin/cos avoids the discontinuity that hurts MLPs.
- **`prev_action`** — lets the policy reason about the action-rate penalty (which depends on `a_t - a_{t-1}`) without needing recurrence.

### 4.6 Reward (M3) — keep it minimal

```python
r = -w_track       * ‖ee - target‖²       # primary objective
    - w_action_rate * ‖a - a_prev‖²        # smoothness in action space
    - w_qdot       * ‖qdot‖²              # smoothness in joint-velocity space
    + w_inband     * 1[‖ee - target‖ < 2 cm]  # shaping bonus
```

Weights `(10, 0.1, 0.001, 0.5)`. Rationale:

- **Squared tracking error** gives a smooth, monotonic gradient toward the target. Linear error (|·|) has zero gradient at 0 and tends to leave the policy with chattering corrections.
- **Action-rate penalty** is essential. Without it, PPO finds policies that achieve low tracking error by violently oscillating around the target — looks great on RMS, looks terrible on a video.
- **Velocity penalty** is small (weight 0.001) — really just there to rule out the worst high-frequency joint chatter.
- **In-band bonus** is the only *shaped* term. It says "once you're close, *stay* close." Without it, the policy tends to oscillate around the target band rather than settling inside it. The threshold (2 cm) is the project's accuracy goal.

The rule from `CLAUDE.md`: **start with 2 reward terms, add only when a failure mode forces it.** Four was the minimum that worked. Five or more terms usually fight each other.

### 4.7 The home-pose tradeoff (M2 → M6)

The most instructive moment of the project.

**First version.** Centered home pose: EE 3.3 cm from the circle *centre* (not the trajectory). Trained 2 M steps. Result: 13 mm rollout RMS in the training callback, but 137 mm max in deterministic eval — a 14 cm startup transient as the arm swung onto the trajectory.

**"Fix".** Solved a new home pose with EE *on* the trajectory at `t=0`. Startup transient gone, but **late-episode tracking degraded to >100 mm.** Cause: the new pose required `joint4 = -1.47 rad`, near its upper limit of `-0.07`. The arm was operating in a low-manipulability region; the policy couldn't keep up with the trajectory at the leftmost extreme.

**Resolution.** Revert to the centered home pose, train for 3 M steps (instead of 2 M), and **report steady-state RMS separately** from full-rollout RMS. The startup transient is honest physics, easily explained, and is excluded from the headline metric by convention.

The lesson: **arms have manipulability margins, and pushing into low-margin regions silently destroys learned policies.** This is the kind of thing you only discover by watching the actual rollout, not by staring at scalar curves.

### 4.8 Uncertainty (M4) — `src/kinesis/envs/wrappers.py`

Two wrappers, both used during training and eval:

- **`ObsNoiseWrapper(σ=2 cm)`** adds Gaussian noise to `ee_pos` *in the observation only*. The targets are clean — the agent has noisy proprioception of its own hand, not noisy knowledge of the goal.
- **`ActionDelayWrapper(k=2)`** queues actions and applies them 2 control steps (40 ms) later. The initial queue is zeros so the first 40 ms is a no-op.

Training with the wrappers from the start, rather than as a post-hoc test, means **robustness is part of the optimization** — not a property we hope for. The ablation table (`results/ablation.md`) shows that the policy is roughly equally good in clean and noisy+delayed conditions.

### 4.9 Training (M6) — `scripts/train.py`

PPO with stable-baselines3, mostly defaults:

```python
PPO(
    "MlpPolicy",
    env=vec,
    learning_rate=3e-4,
    n_steps=2048,        # samples per env before update
    batch_size=64,
    n_epochs=10,         # PPO epochs over each rollout
    gamma=0.99,
    gae_lambda=0.95,
)
```

Callbacks:
- `CheckpointCallback` — saves snapshots every 200 k steps.
- `EvalCallback` — runs the current deterministic policy on a fresh env every 50 k steps, keeps the best.
- Custom `TrackingErrorCallback` — logs `rollout/ee_error_rms_m` to TensorBoard so we can watch the headline metric live.

Training time: 7 min for 3 M steps on CPU, M5 Pro. The 16-way `SubprocVecEnv` was the only optimisation that mattered.

### 4.10 Evaluation (M7) — `scripts/eval.py`

Two critical decisions:

1. **`model.predict(obs, deterministic=True)`** — use the *mean* of the policy distribution, not a sample. This is what you would deploy. Stochastic eval inflates jerk.
2. **Separate startup transient from steady state.** The home pose is 14 cm from `target(t=0)`; the first 0.5 s is an unavoidable swing. Reporting only the full-rollout RMS hides whether the policy actually tracks. We report `RMS` (full), `MAX` (full), `RMS_steady` (t > 1 s), `MAX_steady` (t > 1 s), and `RMS_jerk`.

Artifacts:
- `results/plots/yz_trace.png` — target vs achieved in the y-z plane.
- `results/plots/error_vs_time.png` — error magnitude vs time, with a 1 cm reference line.
- `results/videos/rollout.mp4` — offscreen render at 50 fps.

---

## 5. How to view the simulation

Three options, increasing in usefulness:

### 5.1 Play the saved mp4
Easy. Already on disk.

```bash
open results/videos/rollout.mp4
```

This is the deterministic policy, 10 seconds at 50 fps.

### 5.2 Open the robot in MuJoCo's interactive viewer (no policy)
Useful to *understand the model* — drag the camera around, see joint limits, etc.

```bash
uv run python -m mujoco.viewer --mjcf=assets/mujoco_menagerie/franka_emika_panda/scene.xml
```

You can grab the robot with the mouse and watch passive dynamics. No policy runs.

### 5.3 Run the trained policy in MuJoCo's live viewer (best)
Loads the checkpoint and steps the policy in a draggable interactive window. Use `scripts/play.py`:

```bash
uv run python scripts/play.py
uv run python scripts/play.py --checkpoint checkpoints/best/best_model.zip
```

This is the most insightful view — you can pause, drag the camera, zoom on the gripper, and see exactly how the EE follows the target across the y-z plane.

---

## 6. Where to look next

Curated; opinions live here.

### 6.1 YouTube — what to watch in what order

Linking by channel + lecture/series name rather than URL — URLs change. Search YouTube for the exact title.

**Best starting point for RL:**
- **"Foundations of Deep RL"** — six-lecture series by Pieter Abbeel. ~1 hr each. Goes from MDPs to PPO. The cleanest modern intro. *Channel: Pieter Abbeel.*
- **"RL Course by David Silver"** — 10 lectures, DeepMind / UCL. The classic. Predates deep RL; covers tabular methods, then function approximation. *Channel: David Silver / DeepMind.*

**For deep RL specifically:**
- **"CS285: Deep Reinforcement Learning"** — Sergey Levine, UC Berkeley. The reference graduate course. 20+ lectures. *Channel: RAIL @ Berkeley.*
- **"DeepMind x UCL Reinforcement Learning Course (2021)"** — Hado van Hasselt et al. More recent, more polished, covers DQN through R2D2. *Channel: Google DeepMind.*

**Paper deep-dives:**
- **Yannic Kilcher** — "Proximal Policy Optimization Explained." A ~30 minute walkthrough of the PPO paper. Watch after the Abbeel lectures. *Channel: Yannic Kilcher.*
- **The AI Epiphany (Aleksa Gordić)** — Several PPO and SAC walkthroughs at the implementation level. *Channel: The AI Epiphany.*

**MuJoCo specifically:**
- **"MuJoCo Tutorial"** — DeepMind's official Colab-style intro (a YouTube version exists; also a Colab notebook in the `dm-mujoco` repo). Walks through model loading, `mj_step`, contact dynamics. *Channel: Google DeepMind.*
- **Pascal Klink** — has a clean MuJoCo + RL tutorial series. *Search: "Pascal Klink mujoco".*

**Robotics-RL applied:**
- **Sergey Levine's talks** at robotics conferences (ICRA, CoRL keynotes) on YouTube. Less "course," more "research direction."

### 6.2 Papers — minimum reading list

In order:

1. **Schulman et al., "Proximal Policy Optimization Algorithms" (2017).** The PPO paper. 8 pages. Read it.
2. **Andrychowicz et al., "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study" (2020).** An ablation of every PPO design choice. Worth a careful read once you've trained a few policies.
3. **Tassa et al., "DeepMind Control Suite" (2018).** The reference benchmark suite, all built on MuJoCo. Shows the canonical task design (obs, action, reward) for dozens of continuous-control problems.
4. **Peng et al., "Learning Agile Robotic Locomotion Skills by Imitating Animals" (2020).** Different problem (legged locomotion), but the *recipe* — phase variable, lookahead, reward shaping — is essentially the same as this project.

### 6.3 Books

- **Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.).** Free PDF online. The foundation. Read at least chapters 1–6, 13.
- **Murphy, "Probabilistic Machine Learning: An Introduction"** — for the broader ML context. Optional.

### 6.4 Codebases worth reading

- **stable-baselines3.** You're already using it. Read `sb3/ppo/ppo.py` — it's 200 lines and demystifies the whole algorithm.
- **DeepMind Control Suite (`dm_control`).** The reference for "this is how you design a MuJoCo + RL env." Their envs in `dm_control/suite/` are short and instructive.
- **`mujoco_menagerie`.** Read a few of the XMLs. Compare a manipulator to a quadruped to a humanoid — same primitives, very different scales.

---

## Closing thought

The hardest thing about RL + simulation is *not* the algorithms. PPO is 200 lines. The hard parts are:

1. **Designing the env right.** Obs, action, reward. Most failures are env-design failures, not algorithm failures.
2. **Designing the experiment right.** Deterministic eval, separated transient/steady-state, ablations, video as ground truth.
3. **Diagnosing when it doesn't work.** Watch the trace. Print the rewards. Plot the trajectory in workspace, not just the scalar curves. The home-pose tradeoff in §4.7 was diagnosed by inspecting the per-time-step error pattern — you can't see manipulability collapse on a TensorBoard curve.

Everything else is mechanical.
