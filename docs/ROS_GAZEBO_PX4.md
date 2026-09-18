# ROS 2, Gazebo, and PX4 Integration Guide

## Three boundaries to keep separate

### JAX direct-motor experiment

The controller reads the exact simulated state and commands four normalized rotor thrusts. The same compact plant is used for training and the headline comparison.

### ROS 2/Gazebo direct-motor experiment

Gazebo owns the rigid-body integration, rotor force plugins, collision, sensors, and simulation clock. A ROS 2 node reads odometry/IMU and publishes four rotor angular speeds. This is the independent validation path for the same controller interface.

### PX4 offboard experiment

The ROS node sends NED position/yaw setpoints to PX4. PX4 owns state estimation, position/velocity loops, attitude/rate loops, safety state machine, and actuator allocation. This is useful autopilot integration, but it is not a direct motor-level PID-versus-PPO comparison.

## ROS 2/Gazebo architecture

```text
Gazebo Jetty
  dynamic model + rotors + IMU + odometry
       | GZ messages                         ^ GZ Actuators
       v                                     |
ros_gz_bridge parameter_bridge               |
       | ROS Odometry / Imu                  | ROS Actuators
       v                                     |
quadrotor_controller -------------------------+
       |
       +-- live parameter target_ned

quadrotor_flight_recorder subscribes to odometry and writes CSV metrics.
quadrotor_performance_overlay subscribes to odometry and parameter events and
renders the rolling tracking dashboard in a separate GUI process.
```

The bridge mappings are:

| Topic | Direction | ROS type | Gazebo type |
|---|---|---|---|
| `/clock` | GZ to ROS | `rosgraph_msgs/msg/Clock` | `gz.msgs.Clock` |
| `/quadrotor/odometry` | GZ to ROS | `nav_msgs/msg/Odometry` | `gz.msgs.Odometry` |
| `/quadrotor/imu` | GZ to ROS | `sensor_msgs/msg/Imu` | `gz.msgs.IMU` |
| `/quadrotor/command/motor_speed` | ROS to GZ | `actuator_msgs/msg/Actuators` | `gz.msgs.Actuators` |

## Gazebo plant parameters

- total mass: 1.50 kg (`1.46` kg body plus four `0.01` kg rotors);
- inertia close to `[0.029, 0.029, 0.055]` kg m²;
- arm length 0.23 m;
- 1 ms physics step and real-time factor 1;
- four revolute rotor links in `[FR, FL, RL, RR]` order;
- `motorConstant = 9e-6`, so `thrust = motorConstant * omega²`;
- maximum rotor speed 1000 rad/s, yielding 9 N maximum thrust;
- 12.5 ms spin-up and 25 ms spin-down time constants;
- rotor drag and rolling moment;
- noisy 200 Hz IMU and 100 Hz ground-truth odometry;
- ground collision, atmosphere, gravity, and lighting.

Hover speed is

```text
omega_hover = sqrt((m g / 4) / motorConstant) = 639.226 rad/s.
```

The received values near 639.2 rad/s therefore confirmed that the command bridge and hover conversion were working.

## Frame conversion

Gazebo uses ENU world coordinates and FLU body coordinates. Training uses NED world and FRD body coordinates.

```text
C_NED_ENU = [[0, 1,  0],
             [1, 0,  0],
             [0, 0, -1]]

C_FLU_FRD = diag(1, -1, -1)
R_NED_FRD = C_NED_ENU R_ENU_FLU C_FLU_FRD
```

Gazebo's 3D odometry twist is reported in the child/body FLU frame. To recover inertial NED velocity:

```text
v_NED = C_NED_ENU R_ENU_FLU v_FLU.
```

Confusing body and inertial velocity creates cross-axis damping and can destabilize a controller even if every gain is otherwise correct.

## Reset-safe state handling

A Gazebo GUI reset can move simulation time backward. The odometry publisher also warns that rewind is not intrinsically supported. The controller protects itself in two ways:

1. it detects non-increasing timestamps or gaps larger than 0.2 s, clears the pose-derived velocity estimator, and rejects invalid pose samples;
2. the control timer uses ROS steady wall time rather than simulation time, so a backward `/clock` jump does not suspend motor publication.

Body rates come from the IMU when available. Position-derived velocity is low-pass filtered as `0.35 * raw + 0.65 * previous` between valid samples.

## Motor conversion and the spiral bug

The controllers output normalized thrust `u_i`. The ROS node performs:

```text
force_i = 9.0 * u_i
omega_i = sqrt(force_i / 9e-6)
omega_i = clip(omega_i, 0, 1000).
```

The JAX plant and controller originally shared the same allocation algebra, so an incorrect human-readable motor label did not change their closed loop: the same index meant the same algebraic rotor on both sides. Gazebo gave each index a physical position. When that physical order disagreed, a corrective roll command strengthened the roll instead of opposing it. Correcting the SDF to `[FR, FL, RL, RR]` removed this positive-feedback spiral.

This is a general lesson: self-consistency tests are necessary but not sufficient. Integration requires semantic tests that connect each index to a physical actuator.

## Controller modes

| Mode | Behavior |
|---|---|
| `pid` | Evaluates the cascaded geometric controller at 100 Hz. |
| `ppo` | Loads SB3/PyTorch checkpoint and evaluates deterministic actor mean. |
| `hover` | Publishes four equal nominal hover thrusts; useful for plant isolation. |
| `idle` | Publishes zero rotor force. |

PPO starts Gazebo paused because importing PyTorch and loading the checkpoint is slower than starting the physics server. Wait for `controller=ppo`, then click Play.

## Live target command

The target is a dynamic ROS parameter in NED metres:

```bash
ros2 param set /quadrotor_controller target_ned "[0.25, 0.0, -2.1]"
```

The PPO adapter preserves the true error but translates absolute position into the coordinate system used at training:

```text
policy_position = training_target + current_position - commanded_target.
```

This supplies translation invariance for the fixed-target network input. It does not make large commands in distribution. Use small increments with the current checkpoint.

## Live tracking-performance overlay

`performance_overlay_node.py` converts each Gazebo ENU position sample to NED and maintains a bounded 20-second history. Its four panels show north, east, and positive-up altitude against their references, followed by total 3D position error and running RMSE. A status line displays the current target and component-wise NED error.

The overlay subscribes to `/parameter_events`, so changing the controller's `target_ned` updates the reference without creating a second command interface. A target change starts a fresh segment so the displayed RMSE measures the new command rather than mixing two different references. A non-increasing odometry timestamp indicates Gazebo reset/rewind and also clears the history.

Matplotlib/Tk rendering runs in a separate ROS process from the controller. Therefore an expensive redraw cannot block motor publication. The Tk window uses the desktop's always-on-top hint and an upper-right geometry, making it an overlay over the Gazebo window while avoiding the extra compiler and ABI dependencies of a native Gazebo/Qt plugin.

The launch arguments are:

| Argument | Default | Purpose |
|---|---:|---|
| `show_overlay` | `true` | Start or suppress the GUI process. |
| `overlay_window_seconds` | `20.0` | Rolling plot duration in seconds. |

Run headless with `show_overlay:=false`. The installer supplies `python3-matplotlib` and `python3-tk` for WSLg.

## Recording a matched trial

In a second sourced terminal:

```bash
source /opt/ros/lyrical/setup.bash
source .venv/bin/activate
source install/setup.bash

ros2 run quadrotor_gz recorder --ros-args \
  -p output:=$PWD/results/gazebo_pid.csv \
  -p duration:=10.0 \
  -p target_enu:="[0.0, 0.0, 2.0]"
```

The recorder writes the full trajectory and a sibling `_metrics.csv` file. Use a separate name for PPO. For a fair trial, reset to the same initial condition, use the same target timing, and begin recording at the same phase of the experiment.

## PX4 ROS 2 path

PX4 v1.14 and newer uses uXRCE-DDS for ROS 2 integration. The data path is:

```text
ROS 2 px4_msgs <-> Micro XRCE-DDS Agent <-> PX4 uXRCE-DDS client <-> uORB.
```

The `offboard_adapter.py` node publishes:

- `/fmu/in/offboard_control_mode`;
- `/fmu/in/trajectory_setpoint`;
- `/fmu/in/vehicle_command`.

It subscribes to `/fmu/out/vehicle_odometry`. At 20 Hz it publishes position-mode proof-of-life and the setpoint `[0, 0, -2]` NED with yaw zero. After 20 cycles it requests offboard mode and arming.

PX4 and the ROS workspace must use compatible `px4_msgs` definitions. With newer PX4 message versioning a translation node may be used, but a matched branch remains the clearest learning setup. Start with PX4 SITL, Micro XRCE-DDS Agent, and QGroundControl. PX4 offboard control has real safety consequences; do not test the minimal adapter on hardware without a complete arming/failsafe/takeover design.

## If the goal is a fair PX4-level comparison

Choose one common interface for both controllers. For example:

- both generate trajectory setpoints while PX4 controls the inner loops; or
- both generate attitude/thrust setpoints while PX4 controls rate/allocation; or
- both generate actuator commands in a controlled SITL-only experiment.

Do not compare direct-motor PPO against PX4 position control and attribute the result only to “PPO versus PID”; the plant, estimator, controller stack, and interface would all differ.
