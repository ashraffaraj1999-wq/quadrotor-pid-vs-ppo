# Troubleshooting Guide

This guide records the actual failures encountered during WSL 2, ROS 2 Lyrical, Gazebo Jetty, and PPO bring-up. Diagnose from interfaces outward before changing controller gains.

## Virtual environment creation fails with `ensurepip is not available`

### Cause

The Python-version-specific venv package is missing. Ubuntu 26.04 and ROS 2 Lyrical use Python 3.14 in this setup.

### Fix

```bash
sudo apt update
sudo apt install python3-venv python3.14-venv
```

Remove only the incomplete workspace venv and rerun the installer. `install_lyrical.sh` now detects a `.venv` directory without `.venv/bin/activate` and recreates it safely.

## `install/setup.bash: No such file or directory`

### Cause

The earlier installer stopped before `colcon build` completed, so there is no valid install overlay.

### Fix

```bash
cd /path/to/ros2_gazebo_ws
source /opt/ros/lyrical/setup.bash
./install_lyrical.sh
```

Do not manually source `install/setup.bash` until the build finishes successfully. The run scripts now check for the file and report a direct error.

## `AMENT_TRACE_SETUP_FILES: unbound variable`

### Cause

A shell with `set -u` sourced a ROS setup script that references an unset environment variable. ROS setup scripts are not guaranteed to be nounset-safe.

### Fix

The provided scripts use `set -eo pipefail`, not `set -u`. In a custom shell script, temporarily disable nounset while sourcing ROS:

```bash
set +u
source /opt/ros/lyrical/setup.bash
source install/setup.bash
set -u
```

## Gazebo opens but the drone appears motionless

### Possible explanations

- It begins exactly at the hover target, so good control looks stationary.
- PPO mode intentionally starts paused while PyTorch and the checkpoint load.
- The controller process died or the motor bridge is missing.

### Checks

```bash
ros2 node list
ros2 topic hz /quadrotor/odometry
ros2 topic echo /quadrotor/command/motor_speed --once
ros2 param get /quadrotor_controller controller
```

In PPO mode, wait for `controller=ppo` and click Gazebo Play. To make control visible, command a small NED step:

```bash
ros2 param set /quadrotor_controller target_ned "[0.25, 0.0, -2.1]"
```

## `ros2 topic echo` prints nothing when typed in the launch terminal

### Cause

The launch process already owns that foreground terminal. Your text is not executed by a second Bash prompt; it may merely appear among launch output.

### Fix

Open another WSL terminal, source ROS and the workspace there, then run the topic command:

```bash
cd /path/to/ros2_gazebo_ws
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ros2 topic echo /quadrotor/command/motor_speed --once
```

## `A message was lost` while echoing motor commands

### Meaning

The topic is published at 100 Hz and the terminal subscriber can be slower than the producer. Lost debug-display messages do not by themselves mean Gazebo misses every command.

The observed velocities near 639.2 rad/s are physically correct hover speeds:

```text
sqrt((1.5 * 9.80665 / 4) / 9e-6) = 639.226 rad/s.
```

Use `--once` for a snapshot or `ros2 topic hz` for rate diagnostics.

## Drone spirals or flips after reset

Do not begin by lowering gains. Isolate the loop.

### 1. Verify open-loop balance

```bash
ros2 launch quadrotor_gz sim.launch.py controller:=hover
```

Equal commands should keep the vehicle level, with possible slow drift. Immediate roll/yaw suggests a plant, rotor, or bridge problem.

### 2. Check motor order

The required order is `[FR, FL, RL, RR]`. The JAX allocation, ROS output vector, Gazebo `actuator_number`, rotor link position, and turning direction must agree.

The actual root cause of the rapid spiral was a semantic rotor-position mismatch. A roll correction was applied to the wrong physical corners and became positive feedback. The SDF now places indices in the corrected order.

### 3. Check frames and velocity

Gazebo odometry twist is body FLU, not inertial ENU. The controller must rotate it through attitude before converting ENU to NED. The ROS node also uses IMU body rates in FRD.

### 4. Check reset timing

After GUI reset, simulation time can jump backward. The current node detects rewind, clears velocity history, and keeps the control timer on steady time. A log message `simulation time rewind detected; estimator reset` is expected and protective.

## PPO process dies: `No module named stable_baselines3`

### Cause

ROS console scripts may use `/usr/bin/python3`, while Stable-Baselines3 and PyTorch were installed only in the workspace venv.

### Fix

Rerun `install_lyrical.sh`. `run_ppo.sh` verifies imports using the venv and prepends the venv site-packages directory to `PYTHONPATH` so the ROS-generated console script can import them while retaining system ROS packages.

Direct check:

```bash
.venv/bin/python -c 'import stable_baselines3, torch; print("ok")'
```

## PPO launch shows bridge messages but no `controller=ppo` yet

### Cause

PyTorch import and model loading are slow on the `/mnt/c` filesystem and can lag behind Gazebo startup.

### Expected procedure

The PPO launch opens Gazebo paused. Wait for the controller's load message and `controller=ppo`, then press Play. If the process exits, read the traceback above the launch error rather than assuming a physics problem.

## PPO hovers but tracks a commanded step poorly

### Cause

This is primarily a learned-policy limitation, not a bridge failure:

- one fixed training target;
- reset curriculum of only about ±0.5 m horizontal and ±0.2 m vertical;
- no commanded trajectories or target changes during training;
- about 0.2 m nominal bias already visible in JAX evaluation;
- Gazebo adds motor lag, sensor noise, and model mismatch.

The ROS target adapter removes arbitrary absolute-position dependence but cannot invent missing training experience.

The live overlay makes this limitation visible: compare actual and target traces and watch whether the error converges or settles at a nonzero bias. The displayed RMSE resets when a new target is commanded.

### Practical workaround

Use 0.2 to 0.3 m successive waypoints. This is only a workaround. The engineering fix is retraining with randomized targets, trajectories, and dynamics.

## Rebuild rules

Edit only files under `ros2_gazebo_ws/src/quadrotor_gz`. Then:

```bash
cd /path/to/ros2_gazebo_ws
source /opt/ros/lyrical/setup.bash
source .venv/bin/activate
colcon build --symlink-install
source install/setup.bash
```

With `--symlink-install`, pure Python edits are often visible immediately, but rebuild after launch, setup, package metadata, model, world, or config changes. Close an existing launch before starting another one to avoid duplicate topic publishers and simulator-server conflicts.

## Live plot window does not appear

### Checks

```bash
echo "$DISPLAY"
python3 -c 'import tkinter, matplotlib; print("plot dependencies ok")'
ros2 node list | grep performance_overlay
```

WSLg must expose a graphical display, and both Tk and Matplotlib must be available to the ROS console-script interpreter. Rerun `./install_lyrical.sh` to install `python3-matplotlib` and `python3-tk`, rebuild the package, and source `install/setup.bash`. For a server or other headless session, disable the window with `show_overlay:=false` and use the CSV recorder instead.

## Diagnostic checklist

```bash
# Nodes and graph
ros2 node list
ros2 topic list

# State and command rates
ros2 topic hz /quadrotor/odometry
ros2 topic hz /quadrotor/imu
ros2 topic hz /quadrotor/command/motor_speed

# One command sample
ros2 topic echo /quadrotor/command/motor_speed --once

# Active mode and target
ros2 param get /quadrotor_controller controller
ros2 param get /quadrotor_controller target_ned

# Native Gazebo topics
gz topic -l | grep quadrotor
```

Classify the problem before tuning:

1. no node or crashed process;
2. missing/wrong bridge;
3. wrong topic type or command units;
4. frame/sign mismatch;
5. rotor-order/turning-direction mismatch;
6. reset/time-estimator issue;
7. actuator saturation;
8. controller or learned-policy limitation.
