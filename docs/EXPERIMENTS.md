# Experiments, Results, and Interpretation

## Controller identities

`PID` in filenames and command-line labels means the repository's model-based baseline. Its implementation is translational PD plus geometric attitude PD with gravity feedforward; no integral accumulator is present.

`PPO (BC+RL)` means a Stable-Baselines3 PPO actor initialized by behavior cloning from PID demonstrations and then fine-tuned on-policy. It must not be described as a successful from-scratch PPO controller.

## Training chronology

### 1. Environment plumbing

A 2048-step smoke run verified that Gymnasium, JAX, Stable-Baselines3, Monitor logging, evaluation callbacks, and checkpoint serialization worked. Smoke performance is not a control result.

### 2. Reward failure and repair

An early reward was a negative tracking cost. The agent found a shorter route to a less-negative episode return: terminate quickly and avoid paying future cost. Falling episode length was the diagnostic signal.

The final environment uses positive bounded rewards for position, velocity, and upright attitude, a small effort penalty, and `-100` for true failure. Time-limit completion is a truncation rather than a failure.

### 3. Pure PPO attempt

Pure PPO was trained for approximately 150,000 hover-curriculum steps. It became partially stabilizing but survived only about five seconds in the ten-second deterministic benchmark. This result is retained as a useful failure case.

### 4. Behavior cloning

The geometric controller generated:

- 160 episodes;
- 300 steps per episode;
- 48,000 observation/action pairs;
- randomized position, velocity, and body rate near `[0, 0, -2]` NED.

The actor mean was trained for 25 epochs with Adam at `8e-4`, minibatches of 512, and mean-squared error against PID residual actions. Final mean demonstration MSE was about `3.4e-5`.

### 5. PPO fine-tuning

The warm-start actor was fine-tuned for 50,000 PPO environment steps. The delivered seed-9 checkpoint completes all evaluation flights but retains a position offset.

## PPO settings

| Setting | Value |
|---|---:|
| Policy | SB3 `MlpPolicy` |
| Hidden layers | 128, 128 |
| Learning rate | 0.0003 |
| Rollout length | 1024 |
| Minibatch size | 256 |
| Discount `gamma` | 0.995 |
| GAE `lambda` | 0.95 |
| Initial log standard deviation | -1.0 |
| PPO clip range | SB3 default 0.2 |
| Value coefficient | SB3 default 0.5 |
| Entropy coefficient | SB3 default 0.0 |
| Maximum gradient norm | SB3 default 0.5 |
| Plant/control step | 0.01 s |
| Episode length | 1000 steps = 10 s |

## Fair comparison protocol

The final JAX comparison holds constant:

- plant equations and parameters;
- full state information;
- target and initial state for a given seed;
- motor order, thrust limit, action update period, and saturation;
- failure conditions and episode duration;
- disturbance timing.

Evaluation seeds are 11, 22, and 33. Each controller is deterministic during evaluation.

Scenarios:

1. `nominal`: nominal mass and no wind;
2. `wind_impulse`: 2.5 m/s north wind from 2.0 to 3.0 s;
3. `mass_plus_20pct`: true plant mass multiplied by 1.2 without controller retuning.

## Metrics

Three-dimensional position RMSE is

```text
RMSE = sqrt(mean(||position - target||^2)).
```

The evaluator also records final error, mean squared normalized motor action, success rate, and mean duration.

## Final JAX results

| Controller | Scenario | RMSE (m) | Final error (m) | Mean squared action | Success |
|---|---|---:|---:|---:|---:|
| Geometric PD (`PID`) | nominal | 0.067327 | 0.0000047 | ~0.668 | 100% |
| PPO (BC+RL) | nominal | 0.213027 | 0.207761 | ~0.668 | 100% |
| Geometric PD (`PID`) | wind impulse | 0.069113 | 0.0000143 | ~0.668 | 100% |
| PPO (BC+RL) | wind impulse | 0.207294 | 0.208231 | ~0.668 | 100% |
| Geometric PD (`PID`) | mass +20% | 0.469251 | 0.490336 | ~0.961 | 100% |
| PPO (BC+RL) | mass +20% | 0.590677 | 0.613621 | ~0.962 | 100% |

Use `results/final_comparison/metrics.csv` as the machine-readable source.

## Interpretation

### Nominal case

The geometric controller removes nearly all final error. PPO completes the flight but converges to an offset equilibrium roughly 0.2 m from the target. Similar motor effort means the bias is not explained by a large effort-saving trade.

### Wind impulse

Both survive. The geometric controller rejects the disturbance and returns to the reference; PPO returns to its biased equilibrium. The learned policy is locally robust to this tested impulse but not reference-exact.

### Mass mismatch

Both develop steady altitude error. The geometric baseline's nominal gravity feedforward is wrong and there is no integral term. PPO saw nominal mass during training and also lacks a guaranteed mechanism for estimating the missing constant force.

### Gazebo position steps

The model-based controller was observed to accept a commanded position change and settle stably after frame, odometry, timing, and rotor-index corrections. PPO hovered stably and moved slightly toward commands but tracked poorly. This matches, rather than contradicts, the JAX result: stable flight and accurate reference tracking are distinct requirements.

## Claims this project supports

- Full 6-DoF direct-motor simulation and control were implemented.
- The geometric controller provides accurate nominal tracking in the JAX plant.
- Pure PPO was insufficient within the tested budget.
- PID behavior cloning plus PPO fine-tuning produced stable ten-second flight in the tested JAX scenarios.
- The learned checkpoint is less accurate than the model-based baseline.
- Independent Gazebo bring-up exposed integration bugs that a self-consistent mathematical simulation did not.

## Claims this project does not support

- PPO outperforms PID/geometric control.
- PPO learned the task entirely from scratch.
- The JAX result transfers quantitatively to Gazebo or a real vehicle.
- The PX4 adapter evaluates the same direct-motor controller.
- Three reset seeds establish a statistically broad robotics benchmark.

## Highest-value next experiment

Retrain PPO for tracking rather than fixed-target recovery:

1. sample a new target each episode and occasionally during an episode;
2. feed only translation-invariant position error, velocity, attitude, and rates, or verify any absolute coordinates are necessary;
3. begin with small steps and grow the curriculum;
4. randomize mass, inertia, drag, motor constant, motor lag, wind, observation noise, and latency;
5. include trajectory velocity/acceleration feedforward targets;
6. evaluate multiple training seeds in both JAX and Gazebo;
7. record integrated absolute error, overshoot, settling time, control variation, and failure rate in addition to RMSE.
