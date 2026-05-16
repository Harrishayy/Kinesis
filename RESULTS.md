# Results

All numbers are from deterministic best-checkpoint evaluations under the env's
training distribution unless otherwise noted: 50 Hz control, σ = 2 cm
observation noise on the TCP, 2-step (40 ms) control delay, reset noise on
joint angles, IK-at-reset placing the gripper TCP approximately at
`target(0)`. Episode = 500 steps = 10 s = 2.5 periods of the curve.

The tracked point is the Franka **TCP** (gripper fingertip midpoint, 103.4 mm
out from the wrist body along hand-z). `RMS` and `Max` are the L2 distance
from TCP to target. `Jerk` is the time-RMS of `d³ee/dt³` from the recorded
TCP trace, finite-differenced at the control rate.

---

**Context for the numbers below.** The internal "good" target for this
project (from the project notes) was sub-1 cm RMS on a 15 cm circle at
0.25 Hz. The headline residual policy in §2 clears that target on a
*harder* curve (Viviani, 3-D figure-eight on a sphere) under both
uncertainty sources the brief asks for. The same checkpoint zero-shots
onto the circle at 12.95 mm RMS, which is itself better than every
end-to-end PPO baseline trained natively on the circle (§1, best naive
circle: 23.08 mm). So the "is 6.43 mm good?" question has two
verifiable anchors in this same document: the 1 cm internal target, and
the 23.08 mm naive baseline on a *simpler* curve.

---

## 1. End-to-end PPO baselines (ablations)

A single position-tracking reward and an orientation-alignment penalty;
no analytic feedforward. Five configs vary one factor at a time to
isolate where end-to-end PPO bottoms out before we introduce the
residual decomposition.

| Variant | Trajectory | Steady RMS ↓ | Steady Max ↓ | RMS Jerk ↓ | What changed |
|---|---|---|---|---|---|
| `circle` | circle, 15 cm | 23.08 mm | 63.0 mm | 242 m/s³ | simple baseline trajectory |
| `viviani` | Viviani, R=12 cm | 29.18 mm | 92.0 mm | 243 m/s³ | 3-D figure-eight on a sphere; **harder curve, same compute** |
| `viviani_slow` | Viviani, T=6 s | 16.93 mm | 64.7 mm | 295 m/s³ | slowed period -- hypothesis was *speed-limited*; failed |
| `viviani_4m` | Viviani | 8.85 mm | 17.1 mm | 187 m/s³ | 4M steps instead of 2M; **biggest naive-RL gain** |
| `viviani_v2` | Viviani | 8.40 mm | 15.0 mm | 188 m/s³ | + EE/target Cartesian velocities in obs; lookahead 0.4 → 1.2 s; MLP [64,64] → [256,256] |

**Reads:** the naive `viviani` baseline is worse than the simple `circle` --
unsurprisingly, the curve is harder. Slowing the period (`viviani_slow`)
*hurt* rather than helped, because at fixed step budget the policy sees
fewer trajectory traversals per episode -- a useful negative result.
Doubling training (`viviani_4m`) was the biggest single lever in this
group. The obs / arch upgrades (`viviani_v2`) gave a small additional
gain on max error but didn't move RMS much, signalling we were
**noise-floor-limited**, not architecture-limited.

---

## 2. Residual RL on a 6-DoF analytic feedforward -- the headline

`a_total = clip( a_feedforward(q, target, target_vel) + a_residual(obs), ±1 )`

The feedforward is a damped-least-squares Jacobian-pseudoinverse IK on the
TCP, solving for both position (next-step trajectory displacement +
proportional pull) and orientation (lock hand-z to world −z). The
policy outputs a *residual* on top -- it never has to re-learn kinematics,
only delay/noise/dynamics compensation. Same 4M training budget; same
σ = 2 cm noise + 2-step delay; same `viviani_residual` checkpoint used
for the two zero-shot evals.

| Eval trajectory | Mode | Steady RMS ↓ | Steady Max ↓ | RMS Jerk ↓ |
|---|---|---|---|---|
| **Viviani** | **native (trained)** | **6.43 mm** | **10.98 mm** | **48.8 m/s³** |
| circle | zero-shot | 12.95 mm | 29.9 mm | 51.2 m/s³ |
| figure-8 | zero-shot | 31.45 mm | 94.5 mm | 52.4 m/s³ |

**Reads:**

1. **6.43 mm steady RMS on a deliberately-harder-than-circle curve, under
   σ = 2 cm noise and 2-step delay.** Better than every end-to-end variant
   on the same curve (best non-residual: 8.40 mm).

2. **Zero-shot circle (12.95 mm) is better than the natively-trained
   circle policy in §1 (23.08 mm).** The Viviani-trained residual,
   given the trajectory's own analytic FF at eval time, outperforms a
   circle-specific policy that was actually trained on the curve. The
   residual decomposition is doing the work -- not curve-specific learning.

3. **Jerk dropped 4× across the board** (188–295 → 48–52 m/s³). The FF
   is analytically smooth; the residual is small; total action is
   correspondingly smooth.

---

## 3. Noise-color robustness

Same Viviani-residual policy, evaluated under different observation-noise
spectra. Pink noise is generated via Voss-McCartney (6 octaves) at the
same σ = 2 cm, but with non-zero autocorrelation across the lookahead
window -- a strictly harder filtering problem than IID Gaussian. The
"pink-trained" row trains a separate policy with `noise_color: pink`
during data collection.

| Policy | Eval noise | Steady RMS ↓ | Steady Max ↓ | RMS Jerk ↓ |
|---|---|---|---|---|
| white-trained (`viviani_residual`) | white (training) | 6.43 mm | 10.98 mm | 48.8 m/s³ |
| **white-trained** | **pink** | **6.43 mm** | **11.16 mm** | **41.8 m/s³** |
| pink-trained (`viviani_residual_pink`) | pink | 7.09 mm | 12.48 mm | 40.3 m/s³ |

**Reads:**

- The white-trained policy is **indifferent** to noise color at σ = 2 cm.
  Pink's broadband autocorrelation that was supposed to defeat a fixed-
  lookahead filter doesn't surface at this magnitude -- the FF + 1.2 s
  lookahead horizon already integrates over the noise's correlation
  band. RMS and max are within sample noise of each other.
- Training under pink noise *underperforms slightly* (7.09 vs 6.43 mm)
  on the same eval, at the same compute. The harder data distribution
  doesn't earn its compute back.
- Net: the residual policy is **noise-color robust without retraining** --
  a clean robustness paragraph the submission can claim with one
  controlled experiment, not three.

---

## Reproducibility

```
uv run python scripts/eval.py   --config <name>                            # plots + side-view video
uv run python scripts/tools/render_views.py --config <name>                # multi-view (side/front/bottom/top)
uv run python scripts/train.py  --config <name>                            # retrain (slowest configs ~15 min CPU)
```

Configs in `src/kinesis/configs/{naive,residual}/` (the `--config` resolver
searches both subdirs, so a bare name like `viviani_residual` works).
Checkpoints (best-by-eval) in `checkpoints/<name>/best/best_model.zip`. Plots
in `results/<name>/plots/`, videos in `results/<name>/videos/`.
