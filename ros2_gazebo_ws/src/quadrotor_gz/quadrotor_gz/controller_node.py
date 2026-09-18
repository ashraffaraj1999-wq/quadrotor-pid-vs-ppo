"""Run the project PID or trained PPO against Gazebo's independent physics plant.

Gazebo reports ENU/FLU odometry. The original training environment uses NED/FRD,
so all state data are transformed before either controller is evaluated.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from actuator_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from sensor_msgs.msg import Imu


C_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
C_FLU_FRD = np.diag([1.0, -1.0, -1.0])  # maps FRD coordinates into FLU


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Scalar-first unit quaternion to direction-cosine matrix."""
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Numerically stable direction-cosine matrix to scalar-first quaternion."""
    q = np.empty(4)
    trace = np.trace(r)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q[:] = [0.25*s, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
                (r[1, 0]-r[0, 1])/s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = 2.0 * math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12))
            q[:] = [(r[2, 1]-r[1, 2])/s, 0.25*s, (r[0, 1]+r[1, 0])/s,
                    (r[0, 2]+r[2, 0])/s]
        elif i == 1:
            s = 2.0 * math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12))
            q[:] = [(r[0, 2]-r[2, 0])/s, (r[0, 1]+r[1, 0])/s, 0.25*s,
                    (r[1, 2]+r[2, 1])/s]
        else:
            s = 2.0 * math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12))
            q[:] = [(r[1, 0]-r[0, 1])/s, (r[0, 2]+r[2, 0])/s,
                    (r[1, 2]+r[2, 1])/s, 0.25*s]
    q /= max(np.linalg.norm(q), 1e-12)
    return q if q[0] >= 0 else -q


def odom_to_training_state(msg: Odometry) -> np.ndarray:
    p_enu = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                      msg.pose.pose.position.z])
    # Gazebo OdometryPublisher reports 3D twist in the child/body FLU frame.
    v_flu = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                      msg.twist.twist.linear.z])
    q_enu_flu = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                          msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
    r_enu_flu = quat_to_rot(q_enu_flu)
    r_ned_frd = C_NED_ENU @ r_enu_flu @ C_FLU_FRD
    q_ned_frd = rot_to_quat(r_ned_frd)
    omega_flu = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                          msg.twist.twist.angular.z])
    omega_frd = C_FLU_FRD @ omega_flu
    v_ned = C_NED_ENU @ (r_enu_flu @ v_flu)
    return np.r_[C_NED_ENU @ p_enu, v_ned, q_ned_frd, omega_frd]


class CascadedGeometricController:
    def __init__(self):
        self.mass, self.gravity = 1.5, 9.80665
        self.arm, self.max_thrust, self.yaw_coeff = 0.23, 9.0, 0.018
        self.kp_pos = np.array([2.2, 2.2, 4.0])
        self.kd_pos = np.array([2.8, 2.8, 3.2])
        self.kp_att = np.array([7.5, 7.5, 3.0])
        self.kd_att = np.array([0.34, 0.34, 0.22])

    def command(self, state: np.ndarray, target_ned: np.ndarray) -> np.ndarray:
        pos, vel, q, rates = state[:3], state[3:6], state[6:10], state[10:13]
        acc_cmd = self.kp_pos * (target_ned - pos) - self.kd_pos * vel
        force_n = self.mass * (acc_cmd - np.array([0.0, 0.0, self.gravity]))
        thrust = np.linalg.norm(force_n)
        b3 = -force_n / max(thrust, 1e-9)
        b2 = np.cross(b3, np.array([1.0, 0.0, 0.0]))
        b2 /= max(np.linalg.norm(b2), 1e-9)
        rd = np.column_stack((np.cross(b2, b3), b2, b3))
        r = quat_to_rot(q)
        skew = 0.5 * (rd.T @ r - r.T @ rd)
        e_r = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        tau = -self.kp_att * e_r - self.kd_att * rates
        a = self.arm / math.sqrt(2.0)
        # Motor indices are FR, FL, RL, RR. This order must match the SDF links
        # because PPO learned the same allocation in the JAX plant.
        allocation = np.array([
            [1, 1, 1, 1], [-a, a, a, -a], [a, a, -a, -a],
            [self.yaw_coeff, -self.yaw_coeff, self.yaw_coeff, -self.yaw_coeff],
        ])
        forces = np.linalg.solve(allocation, np.r_[thrust, tau])
        return np.clip(forces / self.max_thrust, 0.0, 1.0)


class QuadrotorController(Node):
    def __init__(self):
        super().__init__('quadrotor_controller')
        self.declare_parameter('controller', 'pid')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('target_ned', [0.0, 0.0, -2.0])
        self.declare_parameter('ppo_training_target_ned', [0.0, 0.0, -2.0])
        self.declare_parameter('control_rate_hz', 100.0)
        self.declare_parameter('motor_constant', 9e-6)
        self.declare_parameter('max_motor_speed', 1000.0)
        self.mode = self.get_parameter('controller').value.lower()
        self.target = np.asarray(self.get_parameter('target_ned').value, dtype=float)
        self.ppo_training_target = np.asarray(
            self.get_parameter('ppo_training_target_ned').value, dtype=float)
        self.motor_constant = float(self.get_parameter('motor_constant').value)
        self.max_motor_speed = float(self.get_parameter('max_motor_speed').value)
        self.state = None
        self._last_stamp_ns = None
        self._last_position_enu = None
        self._velocity_ned = np.zeros(3)
        self._imu_rates_frd = None
        self.pid = CascadedGeometricController()
        self.policy = self._load_policy() if self.mode == 'ppo' else None
        self.pub = self.create_publisher(Actuators, '/quadrotor/command/motor_speed', 10)
        self.create_subscription(Odometry, '/quadrotor/odometry', self._on_odom, 20)
        self.create_subscription(Imu, '/quadrotor/imu', self._on_imu, 50)
        self.add_on_set_parameters_callback(self._on_parameters)
        rate = float(self.get_parameter('control_rate_hz').value)
        # A steady wall clock keeps control alive if Gazebo's /clock jumps backward
        # during a GUI world reset.
        self._control_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0 / rate, self._control, clock=self._control_clock)
        self.get_logger().info(f'controller={self.mode}, target_ned={self.target.tolist()}')

    def _load_policy(self):
        checkpoint = Path(str(self.get_parameter('checkpoint').value)).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'PPO checkpoint does not exist: {checkpoint}')
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError('PPO mode needs stable-baselines3 and torch in the WSL Python environment') from exc
        self.get_logger().info(f'loading PPO policy from {checkpoint}')
        return PPO.load(str(checkpoint), device='cpu')

    def _on_odom(self, msg: Odometry):
        position_enu = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                                 msg.pose.pose.position.z])
        quaternion = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                               msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        if not np.all(np.isfinite(position_enu)) or np.linalg.norm(quaternion) < 0.5:
            self.get_logger().warning('discarding invalid odometry sample')
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        reset_or_gap = (self._last_stamp_ns is None or stamp_ns <= self._last_stamp_ns or
                        stamp_ns - self._last_stamp_ns > 200_000_000)
        if reset_or_gap:
            if self._last_stamp_ns is not None and stamp_ns <= self._last_stamp_ns:
                self.get_logger().warning('simulation time rewind detected; estimator reset')
            self._velocity_ned = np.zeros(3)
        else:
            dt = (stamp_ns - self._last_stamp_ns) * 1e-9
            raw_velocity_ned = C_NED_ENU @ ((position_enu - self._last_position_enu) / dt)
            self._velocity_ned = 0.35 * raw_velocity_ned + 0.65 * self._velocity_ned

        state = odom_to_training_state(msg)
        state[3:6] = self._velocity_ned
        if self._imu_rates_frd is not None:
            state[10:13] = self._imu_rates_frd
        elif reset_or_gap:
            state[10:13] = 0.0
        self.state = state
        self._last_stamp_ns = stamp_ns
        self._last_position_enu = position_enu

    def _on_imu(self, msg: Imu):
        omega_flu = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                              msg.angular_velocity.z])
        if np.all(np.isfinite(omega_flu)):
            self._imu_rates_frd = C_FLU_FRD @ omega_flu

    def _on_parameters(self, parameters):
        """Allow position steps to be commanded while Gazebo is running."""
        for parameter in parameters:
            if parameter.name == 'target_ned':
                candidate = np.asarray(parameter.value, dtype=float)
                if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                    return SetParametersResult(
                        successful=False, reason='target_ned must contain three finite numbers')
                self.target = candidate
                self.get_logger().info(f'new target_ned={self.target.tolist()}')
        return SetParametersResult(successful=True)

    def _normalized_forces(self) -> np.ndarray:
        if self.mode == 'hover':
            hover = 1.5 * 9.80665 / (4.0 * 9.0)
            return np.full(4, hover)
        if self.mode == 'pid':
            return self.pid.command(self.state, self.target)
        if self.mode == 'ppo':
            # Training used one fixed absolute target. Translate position into that
            # coordinate system so a commanded target change does not create an
            # irrelevant out-of-distribution absolute-position input. The true
            # target error, attitude, velocity, and body rates are unchanged.
            policy_state = self.state.copy()
            policy_state[:3] = (
                self.ppo_training_target + self.state[:3] - self.target)
            obs = np.r_[policy_state, self.target - self.state[:3],
                        -self.state[3:6]].astype(np.float32)
            residual, _ = self.policy.predict(obs, deterministic=True)
            hover = 1.5 * 9.80665 / (4.0 * 9.0)
            return np.clip(hover + 0.35 * np.asarray(residual, dtype=float), 0.0, 1.0)
        return np.zeros(4)

    def _control(self):
        if self.state is None:
            return
        normalized = self._normalized_forces()
        forces = 9.0 * normalized
        omega = np.sqrt(np.maximum(forces, 0.0) / self.motor_constant)
        omega = np.clip(omega, 0.0, self.max_motor_speed)
        msg = Actuators()
        if hasattr(msg, 'header'):
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = omega.tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
