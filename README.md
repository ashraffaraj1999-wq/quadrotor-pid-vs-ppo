# Full-Dynamics Quadrotor Control Lab

An educational, end-to-end comparison of a model-based geometric controller and a learned PPO policy on a nonlinear six-degree-of-freedom quadrotor.

The repository contains two deliberately different simulation layers:

1. a compact JAX rigid-body model used for fast training, controlled experiments, plots, and animation; and
2. an independent ROS 2 Lyrical + Gazebo Jetty plant with dynamic rotors, motor lag, collisions, sensors, frame conversion, and live motor commands.

It also contains a separate PX4 ROS 2 offboard adapter. That adapter sends high-level position/yaw setpoints and leaves estimation and inner-loop stabilization to PX4; it is not secretly used by either direct-motor controller.

## What the project demonstrates

- A 13-state, full 6-DoF rigid-body plant in NED/FRD coordinates.
- Quaternion attitude kinematics and RK4 integration at 100 Hz.
- A cascaded position/velocity and geometric attitude PD controller. The class keeps the historical name `CascadedPID`, but the implemented integral gain is zero and there is no integral state.
- A Gymnasium environment with a 19-value observation and hover-centered residual actions.
- Stable-Baselines3 PPO, reward-design failure analysis, PID behavior cloning, and PPO fine-tuning.
- Fair, matched-seed evaluation under nominal flight, a wind impulse, and +20% mass.
- A side-by-side plot and animated 3D PID/PPO replay.
- A live ROS/Gazebo simulation with an always-on-top real-time tracking dashboard, plus a separate PX4 offboard integration path.

## Coordinate and actuator contract

| Item | Contract |
|---|---|
| Training world frame | NED: north, east, down |
| Training body frame | FRD: forward, right, down |
| Gazebo frames | ENU world and FLU body, converted explicitly in the controller node |
| State | `[p_NED(3), v_NED(3), q_body_to_NED(wxyz,4), omega_FRD(3)]` |
| Physical motor order | `[front-right, front-left, rear-left, rear-right]` or `[FR, FL, RL, RR]` |
| Plant action | Four normalized rotor thrusts in `[0, 1]` |
| PPO action | Four residuals in `[-1, 1]`; `u_motor = clip(u_hover + 0.35 a, 0, 1)` |

The motor order is a safety-critical semantic contract. The JAX mixer and the Gazebo rotor links must agree. An earlier label mismatch was masked in the self-consistent JAX plant but caused positive feedback in Gazebo until the SDF rotor indices were corrected.

## Measured JAX comparison

Each row is the average of matched evaluation seeds 11, 22, and 33. Both controllers receive identical initial states, targets, actuator limits, timing, and disturbances.

| Scenario | Controller | Position RMSE (m) | Final error (m) | Success |
|---|---|---:|---:|---:|
| Nominal | Geometric PD (`PID`) | 0.067 | 0.000005 | 100% |
| Nominal | PPO (BC+RL) | 0.213 | 0.208 | 100% |
| Wind impulse | Geometric PD (`PID`) | 0.069 | 0.000014 | 100% |
| Wind impulse | PPO (BC+RL) | 0.207 | 0.208 | 100% |
| Mass +20% | Geometric PD (`PID`) | 0.469 | 0.490 | 100% |
| Mass +20% | PPO (BC+RL) | 0.591 | 0.614 | 100% |

The honest conclusion is that PPO became a stable flight controller, but it did not beat the model-based baseline. The learned policy retains roughly 0.2 m nominal bias and tracks commanded Gazebo steps poorly because training used one fixed target and only small reset offsets.

## Documentation map

- [`docs/ARCHITECTURE_AND_THEORY.md`](docs/ARCHITECTURE_AND_THEORY.md): mathematical and software architecture overview.
- [`docs/PROJECT_TREE.md`](docs/PROJECT_TREE.md): purpose of every authored source file and generated artifact category.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): training history, fair comparison protocol, results, and interpretation.
- [`docs/ROS_GAZEBO_PX4.md`](docs/ROS_GAZEBO_PX4.md): live simulator, topics, transforms, reset handling, PPO limitations, and PX4 boundary.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md): the exact WSL/Gazebo issues encountered and how to diagnose them.

## Windows: train and evaluate the JAX plant

From this repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q

python evaluate.py
python train_ppo.py --smoke

# Reproduce the successful imitation-plus-RL path.
python train_ppo.py --timesteps 150000 --seed 8
python pretrain_bc.py --base checkpoints/best_seed_8/best_model.zip --epochs 25 --seed 8
python train_ppo.py --timesteps 50000 --seed 9 --resume checkpoints/ppo_bc_warmstart.zip

python evaluate.py --ppo checkpoints/ppo_6dof_seed_9.zip `
  --ppo-label "PPO (BC+RL)" --out results/final_comparison
python render_demo.py --ppo checkpoints/ppo_6dof_seed_9.zip
```

The smoke run validates software plumbing only. It is not long enough to train a useful controller.

## WSL 2: run the live ROS 2/Gazebo plant

```bash
cd ros2_gazebo_ws
source /opt/ros/lyrical/setup.bash
chmod +x install_lyrical.sh run_pid.sh run_ppo.sh
./install_lyrical.sh

# Model-based controller
./run_pid.sh

# Learned policy; wait for controller=ppo, then click Gazebo Play.
./run_ppo.sh "$(realpath ../checkpoints/ppo_6dof_seed_9.zip)"
```

Both commands automatically open a rolling dashboard over Gazebo with north/east/altitude tracking, instantaneous 3D error, and running RMSE. Disable it for a headless run with `show_overlay:=false` when launching `sim.launch.py` directly.

Command a small live position step from another sourced WSL terminal:

```bash
ros2 param set /quadrotor_controller target_ned "[0.25, 0.0, -2.1]"
```

Start with 0.2 to 0.3 m increments for PPO. One-metre steps are outside its training curriculum. The geometric controller can accept larger steps, subject to motor saturation and the simple reference generator.

## Main result artifacts

- `results/final_comparison/metrics.csv`: numerical comparison.
- `results/final_comparison/nominal_tracking.png`: PID/PPO response plot.
- `results/final_comparison/pid_vs_ppo_3d.gif`: side-by-side animation.
- `results/final_comparison/comparison_trajectories.npz`: raw trajectory archive.

## Scope and safety

This is a simulation and learning project, not flight-certified software. The JAX model omits high-fidelity propeller aerodynamics, battery dynamics, structural flexibility, estimator error, and many hardware failure modes. The Gazebo plant adds motor dynamics, collision, and sensors but remains a simulation. Run PX4 offboard code in SITL first, retain a manual takeover path, and never infer real-aircraft safety from these results.

## License

MIT, see [LICENSE](LICENSE).
