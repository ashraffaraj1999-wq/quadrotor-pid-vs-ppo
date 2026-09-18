# Project Tree and File Responsibilities

This map separates authored source, experiment artifacts, and generated build products. The latter can be deleted and rebuilt; the former define the project.

```text
quadrotor-pid-vs-ppo/
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── configs/
├── src/
├── tests/
├── train_ppo.py
├── pretrain_bc.py
├── evaluate.py
├── render_demo.py
├── checkpoints/
├── results/
├── gazebo/
├── ros2_gazebo_ws/
├── px4_ros2/
└── docs/
```

## Repository root

| File | Purpose |
|---|---|
| `README.md` | GitHub landing page: scope, contracts, headline results, documentation links, and quick-start commands. |
| `.gitignore` | Excludes virtual environments, Python caches, and generated colcon `build/install/log` trees while keeping educational checkpoints and final results. |
| `pyproject.toml` | Python project metadata and tool configuration. It makes `src` importable in a normal Python workflow and declares package identity. |
| `requirements.txt` | Direct Python dependencies for JAX simulation, Gymnasium, Stable-Baselines3/PyTorch, plotting, tests, and animation. |
| `configs/default.yaml` | Human-readable experiment parameter sheet. The current scripts use in-code defaults, so this file documents the intended setup rather than dynamically driving every run. Its target is NED `[0, 0, -2]`. |
| `train_ppo.py` | Creates or resumes an SB3 PPO model, validates the Gymnasium environment, records Monitor logs, runs periodic deterministic evaluation, saves best models, and writes the final checkpoint. |
| `pretrain_bc.py` | Generates PID demonstrations in the JAX plant, converts motor thrusts into PPO residual-action coordinates, and trains the PPO actor mean by supervised MSE before saving a warm-start checkpoint. |
| `evaluate.py` | Executes matched-seed PID and PPO rollouts in nominal, wind-impulse, and +20% mass cases; calculates metrics; writes CSV; and creates the four-panel comparison plot. |
| `render_demo.py` | Reuses deterministic rollout code, saves compressed trajectories, and produces the side-by-side 3D GIF. This is trajectory rendering, not an independent plant. |

## Core Python package: `src/`

| File | Purpose |
|---|---|
| `src/__init__.py` | Marks `src` as a Python package. |
| `src/dynamics_jax.py` | Source of truth for the 13-state JAX plant: parameters, quaternion operations, motor wrench for `[FR, FL, RL, RR]`, translational and rotational derivatives, Euler/RK4 steps, JIT-compiled single step, vmapped batched step, hover command, and initial state. |
| `src/env.py` | Gymnasium wrapper: 19-value observation, residual action mapping, randomized reset, positive living reward, termination/truncation logic, diagnostic `info`, and a minimal RGB-array render. |
| `src/pid.py` | Historical `CascadedPID` class. Implements position/velocity PD, gravity feedforward, desired attitude construction, SO(3) attitude error, rate damping, allocation, and motor saturation. It has no integral state. |

## Tests: `tests/`

| File | Purpose |
|---|---|
| `tests/test_dynamics.py` | Regression tests for hover balance, free fall, equal-thrust torque symmetry, quaternion normalization, and related mechanics invariants. |
| `tests/test_env.py` | Checks action/observation shapes and Gymnasium API compliance so SB3 receives a valid environment contract. |

These tests are intentionally small. They catch frame/sign/allocation regressions more cheaply than a full training run.

## Training artifacts: `checkpoints/`

| Path | Meaning |
|---|---|
| `ppo_6dof_smoke.zip` | Very short plumbing run. Never interpret it as a trained flight controller. |
| `best_seed_8/best_model.zip` | Best periodic checkpoint from the pure PPO seed-8 curriculum; used as the network container/base for behavior cloning. |
| `ppo_bc_warmstart.zip` | Actor after PID behavior cloning and before final PPO fine-tuning. |
| `best_seed_9/best_model.zip` | Best periodic checkpoint during the seed-9 fine-tuning run. |
| `ppo_6dof_seed_9.zip` | Final delivered PPO (BC+RL) checkpoint used by evaluation, animation, and the ROS node. |
| `ppo_6dof_seed_7.zip`, `ppo_6dof_seed_8.zip` | Earlier training endpoints retained for provenance and comparison. |

SB3 `.zip` checkpoints contain policy/value parameters and algorithm metadata. They require compatible Stable-Baselines3 and PyTorch installations.

## Results: `results/`

| Path | Meaning |
|---|---|
| `ppo_seed_*.monitor.csv` | Episode returns and lengths logged during PPO training. |
| `eval_seed_*/evaluations.npz` | Periodic evaluation callback histories. |
| `bc_eval/` and `best_eval/` | Intermediate comparison outputs used while choosing the delivered policy. |
| `final_comparison/metrics.csv` | Final matched-seed quantitative results. |
| `final_comparison/nominal_tracking.png` | Four-panel nominal response plot. |
| `final_comparison/pid_vs_ppo_3d.gif` | 3D animated trajectory comparison. |
| `final_comparison/comparison_trajectories.npz` | Raw PID and PPO states plus `dt`, used by animation/replay tools. |
| root `results/metrics.csv` and `nominal_tracking.png` | Earlier or PID-only evaluation outputs; use `final_comparison/` for the final comparison. |

## Legacy Gazebo replay: `gazebo/`

| File | Purpose |
|---|---|
| `gazebo/comparison.world.sdf` | Lightweight Gazebo world for viewing two replay models. |
| `gazebo/quadrotor_visual/model.sdf` | Kinematic visual model used only for pose replay. |
| `gazebo/quadrotor_visual/model.config` | Gazebo model metadata. |
| `gazebo/replay_trajectories.py` | Loads the stored NPZ trajectories, converts NED/FRD to ENU/FLU, and publishes model poses. |
| `gazebo/README.md` | Replay instructions and an explicit warning that this is not a physics benchmark. |

Do not confuse this folder with `ros2_gazebo_ws`, which applies real simulated motor forces to a dynamic rigid body.

## Live ROS 2/Gazebo workspace: `ros2_gazebo_ws/`

### Entry scripts

| File | Purpose |
|---|---|
| `ros2_gazebo_ws/README.md` | Installation, run, recording, step-command, diagnostic, and WSL graphics guide. |
| `install_lyrical.sh` | Installs the matched ROS/Gazebo packages and Python venv support, recreates an incomplete venv, installs PPO runtime dependencies, runs `rosdep`, and builds with colcon. |
| `run_pid.sh` | Sources ROS, optional venv, and workspace; checks that the build exists; launches Gazebo and the geometric controller. |
| `run_ppo.sh` | Performs the same setup, verifies SB3/Torch and checkpoint, exposes venv site-packages to the ROS console-script interpreter, and launches Gazebo paused while the policy loads. |

### ROS package metadata

| File | Purpose |
|---|---|
| `src/quadrotor_gz/package.xml` | ROS package dependencies and metadata used by ament/rosdep. |
| `src/quadrotor_gz/setup.py` | Installs Python nodes and non-Python resources; declares `controller`, `recorder`, and `performance_overlay` console entry points. |
| `src/quadrotor_gz/setup.cfg` | Places console scripts under the standard ROS package `lib/<package>` path. |
| `src/quadrotor_gz/resource/quadrotor_gz` | Ament resource-index marker. |
| `src/quadrotor_gz/quadrotor_gz/__init__.py` | Python package marker. |

### ROS runtime code

| File | Purpose |
|---|---|
| `quadrotor_gz/controller_node.py` | Central runtime node. Converts Gazebo ENU/FLU odometry and IMU into training NED/FRD state, detects simulation-time rewinds, filters pose-derived velocity, accepts live target parameters, evaluates PID/PPO/hover/idle mode, converts normalized thrust into rotor angular speed, and publishes `Actuators`. |
| `quadrotor_gz/recorder_node.py` | Records odometry for a fixed duration, writes trajectory CSV, and writes RMSE/final/max-error metrics. |
| `quadrotor_gz/performance_overlay_node.py` | Runs a separate Tk/Matplotlib ROS process that follows odometry and controller parameter events; displays rolling N/E/altitude tracking, 3D error, and RMSE; and clears history after a new target or simulator reset. |
| `launch/sim.launch.py` | Finds installed resources, configures Gazebo model search path, launches Gazebo running or paused, launches `ros_gz_bridge`, the selected controller, and optionally the live tracking overlay. |
| `config/bridge.yaml` | Exact bidirectional topic/type mapping for clock, odometry, IMU, and motor-speed actuators. |

### Gazebo plant

| File | Purpose |
|---|---|
| `models/learning_quadrotor/model.sdf` | Dynamic 1.5 kg quadrotor with inertias, collisions, visuals, four revolute rotors in `[FR, FL, RL, RR]` order, motor force/reaction-torque plugins, motor lag, rotor drag, IMU noise, and 100 Hz odometry. |
| `models/learning_quadrotor/model.config` | Gazebo model name/version/SDF metadata. |
| `worlds/quadrotor_world.sdf` | 1 ms physics step, gravity, atmosphere, sensors system, lighting, ground collision, and inclusion of the dynamic model. |

### Generated ROS trees

| Directory | Status |
|---|---|
| `.venv/` | Python environment created by the installer; regenerate rather than commit. |
| `build/` | Colcon intermediate build products. |
| `install/` | Colcon install overlay sourced at runtime. It may contain symlinks into `src` because the build uses `--symlink-install`. |
| `log/` | Colcon build logs and command transcripts. |

Never edit generated copies under `build/` or `install/`; edit `src/quadrotor_gz`, rebuild, and source the overlay.

## PX4 adapter: `px4_ros2/`

| File | Purpose |
|---|---|
| `px4_ros2/quadrotor_bridge/quadrotor_bridge/offboard_adapter.py` | Minimal PX4 ROS 2 offboard node. Publishes proof-of-life/control-mode, position/yaw setpoint, and vehicle commands at 20 Hz; subscribes to PX4 odometry; requests offboard mode and arming after 20 setpoint cycles. |

The adapter is a reference integration component, not a complete standalone ament package in this tree. A real PX4 workspace must include matching `px4_msgs` definitions, package metadata/entry points, Micro XRCE-DDS Agent, PX4 SITL, and normally QGroundControl.

## Documentation: `docs/`

| File | Purpose |
|---|---|
| `ARCHITECTURE_AND_THEORY.md` | Compact theory and relationship among JAX, Gymnasium, PPO, ROS/Gazebo, and PX4. |
| `PROJECT_TREE.md` | This exhaustive source/artifact map. |
| `EXPERIMENTS.md` | Training chronology, evaluation protocol, numerical results, and claim limits. |
| `ROS_GAZEBO_PX4.md` | Integration architecture, transforms, topics, timing, recording, and PX4 boundary. |
| `TROUBLESHOOTING.md` | Symptoms, causes, checks, and fixes discovered during WSL/Gazebo bring-up. |
