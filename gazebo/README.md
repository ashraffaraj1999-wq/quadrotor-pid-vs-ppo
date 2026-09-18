# Gazebo Harmonic trajectory replay

This replays the already-evaluated PID and PPO trajectories for visual comparison; Gazebo is not used as the benchmark plant. On Ubuntu with ROS 2 and `ros_gz_interfaces` installed:

```bash
export GZ_SIM_RESOURCE_PATH=$PWD/gazebo:$GZ_SIM_RESOURCE_PATH
gz sim -r gazebo/comparison.world.sdf
# In a sourced ROS 2 environment, from the repository root:
python gazebo/replay_trajectories.py
```

The replay converts NED/FRD state to Gazebo ENU/FLU explicitly. For a true PX4 dynamics demo, use PX4 SITL with Gazebo and send high-level offboard setpoints through the separate ROS 2 adapter; do not treat this kinematic replay as a second physics validation.
