# Uncertainty ablation

Deterministic policy from `/Users/harrishayyanar/Documents/Kinesis/checkpoints/best/best_model.zip` evaluated on 400 steps (8.0 s) of circle tracking under four uncertainty conditions.

| Condition | Full RMS (mm) | Full MAX (mm) | Steady RMS (mm) | Steady MAX (mm) | RMS jerk (m/s³) |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 13.93 | 131.13 | 5.20 | 7.52 | 17.5 |
| noise only | 14.17 | 131.11 | 5.73 | 9.68 | 175.5 |
| delay only | 15.73 | 137.35 | 4.15 | 6.93 | 59.6 |
| noise+delay | 15.85 | 137.35 | 4.56 | 9.50 | 162.0 |

Observation noise σ = 2 cm (applied to EE position in obs only). Control delay = 2 steps × 20 ms = 40 ms.
