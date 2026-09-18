# ROS 2 Lyrical + Gazebo Jetty full-physics validation

This workspace replaces the earlier pose replay with a second, independent physics plant. Gazebo integrates the vehicle rigid body and applies force and reaction torque at four dynamic rotor links. The ROS controller receives simulated odometry, transforms Gazebo ENU/FLU data to the NED/FRD convention used during training, and publishes physical motor angular velocities.

## What is modeled

- 1.5 kg rigid body with the project inertia and 0.23 m arm length
- four revolute rotor links
- quadratic rotor thrust and reaction torque
- asymmetric 12.5/25 ms motor spin-up/spin-down lag
- rotor drag and rolling moment
- ground collision, gravity, atmosphere, 1 ms physics step
- noisy 200 Hz IMU and 100 Hz ground-truth odometry
- ROS/Gazebo simulation clock

The controller runs at 100 Hz. Motor-index order is front-right, front-left, rear-left, rear-right, matching the allocation used to train PPO. The model starts at ENU `[0, 0, 2]`, which is NED `[0, 0, -2]`. Its ENU yaw is +90 degrees so body-forward points north; after the ENU/FLU to NED/FRD transformation this is NED yaw zero.

Both controller launch scripts also open a compact, always-on-top tracking dashboard over the Gazebo window. Plotting runs in a separate ROS process, so repainting the dashboard cannot delay the 100 Hz motor-control loop.

## Install in WSL 2

Open an Ubuntu shell and copy or access this workspace through `/mnt/c`. From `ros2_gazebo_ws`:

```bash
source /opt/ros/lyrical/setup.bash
chmod +x install_lyrical.sh run_pid.sh run_ppo.sh
./install_lyrical.sh
```

ROS 2 Lyrical's default simulator pairing is Gazebo Jetty. The installer deliberately uses `ros-lyrical-ros-gz` instead of mixing Gazebo repositories.

## Run PID

```bash
cd /path/to/ros2_gazebo_ws
./run_pid.sh
```

The live dashboard opens automatically and shows:

- north, east, and altitude actual-versus-target traces;
- instantaneous three-dimensional position error;
- running position RMSE for the current target/reset segment;
- the active controller, target, and NED error vector.

The default history window is 20 seconds. Gazebo reset-time rewind clears the plot, and changing `target_ned` begins a fresh tracking segment. The window is placed at the upper-right and marked always-on-top so it can sit over the Gazebo GUI. It remains a separate process/window rather than a compiled Gazebo C++ plugin.

To record a 10 second trial, open a second sourced WSL terminal:

```bash
source /opt/ros/lyrical/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 run quadrotor_gz recorder --ros-args \
  -p output:=$PWD/results/gazebo_pid.csv -p duration:=10.0
```

## Run PPO

Pass the Linux/WSL path of the trained checkpoint:

```bash
./run_ppo.sh "$(realpath ../checkpoints/ppo_6dof_seed_9.zip)"
```

PPO startup opens Gazebo paused because importing PyTorch and loading the policy is slower than starting physics. Wait until the terminal prints `controller=ppo`, then click Gazebo's Play button. This prevents the vehicle from falling before the first policy command.

The ROS adapter translates a commanded target into the fixed absolute coordinate system used during PPO training while preserving the real position error. This allows translated hover points without changing the learned policy. The training reset curriculum was only about +/-0.5 m horizontally and +/-0.2 m vertically, so test 0.2 to 0.3 m steps before attempting a one-metre step. Larger paths should be sent as successive waypoints or addressed by retraining with randomized targets and trajectories.

Record it using a different filename:

```bash
ros2 run quadrotor_gz recorder --ros-args \
  -p output:=$PWD/results/gazebo_ppo.csv -p duration:=10.0
```

Each recording produces a trajectory CSV and a `_metrics.csv` file containing position RMSE, final error, and maximum error.

## Command a visible position step

The model initially starts exactly at its hover target, so a healthy controller appears stationary. While Gazebo is running, use another sourced terminal to change the target in NED coordinates:

```bash
ros2 param set /quadrotor_controller target_ned "[1.0, 0.0, -2.5]"
```

This commands one metre north and half a metre higher. Return to the original hover point with:

```bash
ros2 param set /quadrotor_controller target_ned "[0.0, 0.0, -2.0]"
```

## Overlay controls

The overlay is enabled by default for PID and PPO. To run without it:

```bash
ros2 launch quadrotor_gz sim.launch.py controller:=pid show_overlay:=false
```

To change the rolling history duration:

```bash
ros2 launch quadrotor_gz sim.launch.py \
  controller:=pid overlay_window_seconds:=30.0
```

It can also be started manually against an existing simulation:

```bash
ros2 run quadrotor_gz performance_overlay --ros-args \
  -p controller:=pid -p window_seconds:=20.0
```

If the dashboard does not appear, confirm that WSLg is displaying Linux GUI windows and that `python3-matplotlib` and `python3-tk` are installed. Rerunning `./install_lyrical.sh` installs both packages.

## Useful checks

```bash
ros2 topic hz /quadrotor/odometry
ros2 topic echo /quadrotor/command/motor_speed --once
gz topic -l | grep quadrotor
```

### Open-loop plant isolation

The `hover` controller publishes four exactly equal hover speeds and applies no feedback:

```bash
ros2 launch quadrotor_gz sim.launch.py controller:=hover
```

If this mode remains level but drifts slowly, the Gazebo rotor plant is balanced and any rapid PID spiral is in the feedback/frame path. If it rapidly rolls or yaws, inspect the four native Gazebo motor-speed topics and rotor configuration before changing PID gains.

At hover, all four commanded motor velocities should be approximately 639 rad/s. If the vehicle immediately yaws while all commands are equal, stop the run and verify that the installed Jetty motor plugin accepts `actuator_number`; do not tune the controller around a rotor-indexing problem.

## Architectural boundary

This package tests the direct-motor PID and PPO against Gazebo physics. It does not silently substitute PX4's attitude controller. A separate PX4 SITL experiment should use PX4-compatible actuator or high-level offboard interfaces and is a different comparison because PX4 then owns estimation and inner-loop stabilization.

## WSL graphics

Windows 11 WSLg normally displays the Gazebo GUI directly. For a headless run, change the launch `gz sim` command to include `-s` and record data through ROS topics.

For a headless run, also pass `show_overlay:=false`; Tk/Matplotlib requires a graphical display.
