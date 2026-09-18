# Architecture and Theory Guide

## System view

```text
                       FAST RESEARCH LOOP
target -> Gymnasium -> PID or SB3 PPO -> normalized rotor thrusts
              |                               |
              +-------- JAX 6-DoF plant <-----+
                           |
                    metrics / plot / GIF

                    INDEPENDENT VALIDATION LOOP
Gazebo odometry + IMU -> ROS 2 controller -> motor angular speeds
          ^                                      |
          +--- ros_gz bridge <- Gazebo physics <-+

                     AUTOPILOT INTEGRATION LOOP
ROS 2 offboard adapter -> PX4 position setpoint -> PX4 estimator/controllers
          ^                                             |
          +--------- VehicleOdometry via uXRCE-DDS <----+
```

The loops are related but not interchangeable. JAX provides a fast plant for learning and controlled evaluation. Gazebo is a second physics implementation that reveals frame, actuator, timing, and sensor errors. PX4 offboard mode is an autopilot interface in which PX4 supplies most of the control stack.

## State, frames, and action

The JAX state is

```text
x = [position_NED(3), velocity_NED(3), quaternion_body_to_NED_wxyz(4), body_rates_FRD(3)]
```

NED means north-east-down. FRD means forward-right-down. Therefore a hover point two metres above the origin is `[0, 0, -2]`.

The physical motor vector is `[FR, FL, RL, RR]`. If each rotor produces force `f_i`, the X-frame allocation is

```text
T     =  f_FR + f_FL + f_RL + f_RR
tau_x =  a(-f_FR + f_FL + f_RL - f_RR)
tau_y =  a( f_FR + f_FL - f_RL - f_RR)
tau_z = ct( f_FR - f_FL + f_RL - f_RR)
```

where `a = arm/sqrt(2)` and `ct` is the yaw reaction-moment coefficient.

## Full rigid-body plant

Translation follows Newton's second law:

```text
p_dot = v
m v_dot = R(q) [0, 0, -sum(f_i)] + f_drag + m[0, 0, g]
f_drag = -D(v - wind)
```

Rotation follows Euler's rigid-body equation:

```text
J omega_dot = tau - omega x (J omega)
q_dot = 0.5 q tensor_product [0, omega]
```

The state is advanced with fixed-step RK4 at 0.01 s and the quaternion is renormalized after each step.

### What “full dynamics” means here

It means all three translational and all three rotational degrees of freedom are coupled through attitude and four actuators. It does not mean high-fidelity rotor aerodynamics. The JAX plant omits blade flapping, induced flow, vortex-ring state, ground effect, battery sag, flexible modes, sensor delay, and motor lag. Gazebo adds several of these integration effects, especially motor lag, collision, rotor drag, and noisy sensors, but it also remains an approximation.

## Why JAX is used

JAX supplies a NumPy-like numerical language plus program transformations:

- `jax.jit` compiles the pure RK4 step with XLA after the first trace. Repeated simulation steps then avoid Python-level work.
- `jax.vmap` creates a batched version of the same step without manually rewriting the equations for a batch dimension.
- JAX arrays make the plant compatible with automatic differentiation, but this project does not use plant gradients to train PPO. Stable-Baselines3 trains its neural networks with PyTorch and model-free sampled trajectories.
- The functional style makes state transition explicit: `next_state = step(state, action, params, dt, wind)`. This is easier to test and vectorize than a simulator with hidden mutable state.

JAX compilation has a one-time tracing/compile cost. Arguments that change Python control flow or array shapes can trigger recompilation. The fixed `dt` is declared static so the compiler sees a stable program.

## Why Gymnasium is used

Gymnasium is an interface contract between the plant and a reinforcement-learning algorithm. `Quadrotor6DoFEnv` defines:

- `reset(seed=...) -> (observation, info)`;
- `step(action) -> (observation, reward, terminated, truncated, info)`;
- observation and action spaces;
- random initial conditions, target, reward, and failure rules.

The 19-value observation is the 13-state vector plus position error and negative velocity:

```text
o = [x(13), target - position(3), -velocity(3)]
```

Velocity appears twice. The state preserves a complete, reusable physical representation; the appended error terms expose task-relevant features directly to the neural network.

`terminated` means a physical/safety failure such as leaving the workspace, hitting the ground boundary, or extreme tilt. `truncated` means the 1000-step time limit ended a valid episode. The distinction matters because value learning should normally bootstrap through time-limit truncation but not through terminal failure.

## Geometric baseline

The baseline is a cascade:

1. Position/velocity PD produces commanded acceleration.
2. Gravity feedforward converts it into desired total force.
3. Desired force plus desired yaw constructs a desired rotation matrix.
4. A coordinate-free attitude error on SO(3) produces commanded torque.
5. A 4x4 allocation matrix converts collective thrust and torque into four rotor forces.

There is no integral state. The historical class name `CascadedPID` is retained so old commands and checkpoints remain understandable; the accurate control-theory name is cascaded translational PD plus geometric attitude PD with gravity feedforward.

## PPO and behavior cloning

PPO is on-policy actor-critic reinforcement learning. The actor represents a Gaussian action distribution; the critic estimates state value. Rollouts produce temporal-difference residuals, generalized advantage estimates, and return targets. PPO reuses each rollout for several minibatch epochs while clipping the new/old action-probability ratio around one.

The policy action is a residual around model-based hover:

```text
u_hover = mg / (4 f_max) ~= 0.409
u_motor = clip(u_hover + 0.35 a_policy, 0, 1)
```

This encodes only gravity balance; it does not encode attitude or position feedback.

Pure PPO did not reliably complete the ten-second benchmark within the tested budget. The final policy was therefore warm-started from 48,000 PID observation/action pairs. Behavior cloning minimized actor-mean MSE for 25 epochs, after which PPO fine-tuned the policy for 50,000 environment steps. The label `PPO (BC+RL)` is essential because the stabilizing prior came from the model-based controller.

## Reward engineering lesson

The first negative per-step tracking cost created an unintended incentive: crashing early stopped future costs. A terminal penalty alone was not enough when the accumulated cost of surviving exceeded the one-time penalty. The final reward uses positive bounded terms for position, velocity, and upright attitude, a small effort penalty, and a large terminal failure penalty. Surviving near the target is therefore always preferable to deliberately terminating.

## Why the Gazebo PPO is stable but tracks poorly

The checkpoint learned one absolute target and a small reset distribution: approximately ±0.5 m horizontally and ±0.2 m vertically. The ROS node translates absolute position into the training coordinate system while preserving true error, preventing an arbitrary world coordinate from confusing the policy. That adapter does not create missing large-step experience. Large target changes remain out of distribution, and the learned equilibrium already has about 0.2 m bias in the original JAX plant.

The correct remedy is retraining with randomized targets and trajectory commands, not simply increasing gains after deployment. A useful next curriculum would sample target offsets, ramp references, waypoint sequences, mass, drag, wind, sensor noise, delay, and motor lag.
