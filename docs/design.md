# Design Note

This document covers the design of Kinesis: state and action spaces, reward shaping, trajectory representation, evaluation methodology, and design decisions made along the way.

> Filled out incrementally as milestones land. Headings are placeholders.

## State, action, and reward

_TODO — observation vector, action representation (joint-position deltas), reward terms and weights._

## Trajectory representation

_TODO — circle / figure-eight / lissajous / random-waypoint classes; phase variable; lookahead horizon._

## Training

_TODO — PPO setup, vectorization, hyperparameters, training-time budget._

## Evaluation

_TODO — RMS tracking error, max error, jerk, smoothness scores. Episode protocol._

## Uncertainty

_TODO — observation noise, control delay. Ablation results._

## What was tried and discarded

_TODO — design choices that didn't pan out, with rationale._

## Next steps

_TODO — sim-to-real, hardware deployment, orientation tracking._
