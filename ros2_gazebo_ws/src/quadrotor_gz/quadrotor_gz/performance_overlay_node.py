"""Always-on-top live tracking plots for the ROS 2 / Gazebo flight test.

The overlay deliberately runs in its own process.  Rendering a graph must never
delay the 100 Hz control loop.  Gazebo odometry is converted from ENU to the
NED convention used by the controller before tracking errors are calculated.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterEvent
from rclpy.node import Node
from rclpy.qos import qos_profile_parameter_events


class TrackingPerformanceOverlay(Node):
    """Subscribe to the live trial and render a rolling performance dashboard."""

    def __init__(self):
        super().__init__('quadrotor_performance_overlay')
        self.declare_parameter('controller', 'pid')
        self.declare_parameter('target_ned', [0.0, 0.0, -2.0])
        self.declare_parameter('window_seconds', 20.0)
        self.declare_parameter('refresh_hz', 10.0)
        self.declare_parameter('always_on_top', True)
        self.declare_parameter('window_geometry', '820x620-20+60')

        self.controller = str(self.get_parameter('controller').value).upper()
        self.target_ned = np.asarray(
            self.get_parameter('target_ned').value, dtype=float)
        self.window_seconds = max(
            float(self.get_parameter('window_seconds').value), 2.0)
        self.refresh_hz = float(np.clip(
            self.get_parameter('refresh_hz').value, 1.0, 30.0))

        # Odometry is nominally 100 Hz.  The extra margin handles short bursts
        # without allowing an indefinitely growing GUI data structure.
        max_samples = int(math.ceil(self.window_seconds * 150.0))
        self.times = deque(maxlen=max_samples)
        self.positions_ned = deque(maxlen=max_samples)
        self.targets_ned = deque(maxlen=max_samples)
        self.error_norms = deque(maxlen=max_samples)
        self._segment_start_stamp = None
        self._last_stamp = None
        self._closed = False

        self.create_subscription(
            Odometry, '/quadrotor/odometry', self._on_odometry, 50)
        self.create_subscription(
            ParameterEvent, '/parameter_events', self._on_parameter_event,
            qos_profile_parameter_events)

        self._create_window()
        self.get_logger().info(
            f'live {self.controller} tracking overlay started; '
            f'target_ned={self.target_ned.tolist()}')

    def _create_window(self):
        """Create the Tk / Matplotlib window after ROS parameters are ready."""
        import matplotlib

        matplotlib.use('TkAgg')
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.root = tk.Tk()
        self.root.title(f'Quadrotor {self.controller} | Live tracking')
        self.root.geometry(str(self.get_parameter('window_geometry').value))
        if bool(self.get_parameter('always_on_top').value):
            self.root.attributes('-topmost', True)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        self.figure = Figure(figsize=(8.2, 6.2), dpi=100, facecolor='#101820')
        self.axes = self.figure.subplots(2, 2)
        self.figure.subplots_adjust(
            left=0.09, right=0.98, bottom=0.11, top=0.86,
            hspace=0.34, wspace=0.25)
        self.title = self.figure.suptitle(
            f'{self.controller} | waiting for Gazebo odometry',
            color='white', fontsize=13, fontweight='bold')
        self.status = self.figure.text(
            0.5, 0.025, 'Target and error metrics will appear when physics runs.',
            ha='center', color='#d7e3f4', fontsize=9)

        position_specs = (
            ('North tracking', 'North (m)'),
            ('East tracking', 'East (m)'),
            ('Altitude tracking', 'Altitude (m)'),
        )
        self.position_lines = []
        for axis, (title, ylabel) in zip(self.axes.flat[:3], position_specs):
            actual, = axis.plot([], [], color='#32a8ff', linewidth=2.0,
                                label='actual')
            target, = axis.plot([], [], color='#ffb000', linewidth=1.7,
                                linestyle='--', label='target')
            self.position_lines.append((actual, target))
            self._style_axis(axis, title, ylabel)
            axis.legend(loc='best', fontsize=8, facecolor='#172330',
                        edgecolor='#52677d', labelcolor='white')

        error_axis = self.axes.flat[3]
        self.error_line, = error_axis.plot(
            [], [], color='#ff5964', linewidth=2.0, label=r'$\|e_p\|$')
        self.rmse_line, = error_axis.plot(
            [], [], color='#66e08a', linewidth=1.8, label='running RMSE')
        self._style_axis(error_axis, '3D position error', 'Error (m)')
        error_axis.legend(loc='best', fontsize=8, facecolor='#172330',
                          edgecolor='#52677d', labelcolor='white')
        for axis in self.axes.flat:
            axis.set_xlabel('Time since target/reset (s)', color='#d7e3f4')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _style_axis(axis, title: str, ylabel: str):
        axis.set_facecolor('#172330')
        axis.set_title(title, color='white', fontsize=10, fontweight='bold')
        axis.set_ylabel(ylabel, color='#d7e3f4')
        axis.tick_params(colors='#d7e3f4', labelsize=8)
        for spine in axis.spines.values():
            spine.set_color('#52677d')
        axis.grid(True, color='#52677d', alpha=0.28, linewidth=0.7)

    def _on_odometry(self, msg: Odometry):
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        p = msg.pose.pose.position
        position_enu = np.array([p.x, p.y, p.z], dtype=float)
        if not np.all(np.isfinite(position_enu)):
            return

        # A GUI reset rewinds Gazebo time.  Begin a fresh plot segment instead
        # of drawing a line backward across the previous flight.
        if self._last_stamp is not None and stamp <= self._last_stamp:
            self._clear_history(stamp)
        if self._segment_start_stamp is None:
            self._segment_start_stamp = stamp
        self._last_stamp = stamp

        position_ned = np.array(
            [position_enu[1], position_enu[0], -position_enu[2]])
        error_norm = float(np.linalg.norm(self.target_ned - position_ned))
        self.times.append(stamp - self._segment_start_stamp)
        self.positions_ned.append(position_ned)
        self.targets_ned.append(self.target_ned.copy())
        self.error_norms.append(error_norm)

        cutoff = self.times[-1] - self.window_seconds
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.positions_ned.popleft()
            self.targets_ned.popleft()
            self.error_norms.popleft()

    def _on_parameter_event(self, event: ParameterEvent):
        """Follow live ``target_ned`` changes made on the controller node."""
        if event.node.rsplit('/', 1)[-1] != 'quadrotor_controller':
            return
        for parameter in (*event.new_parameters, *event.changed_parameters):
            if parameter.name != 'target_ned':
                continue
            candidate = np.asarray(
                parameter.value.double_array_value, dtype=float)
            if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                if not np.allclose(candidate, self.target_ned):
                    self.target_ned = candidate
                    self._clear_history(self._last_stamp)
                    self.get_logger().info(
                        f'overlay target_ned={self.target_ned.tolist()}')

    def _clear_history(self, stamp=None):
        self.times.clear()
        self.positions_ned.clear()
        self.targets_ned.clear()
        self.error_norms.clear()
        self._segment_start_stamp = stamp
        self._last_stamp = stamp

    def _redraw(self):
        if not self.times:
            return
        times = np.asarray(self.times)
        positions = np.asarray(self.positions_ned)
        targets = np.asarray(self.targets_ned)
        errors = np.asarray(self.error_norms)
        running_rmse = np.sqrt(
            np.cumsum(errors * errors) / np.arange(1, len(errors) + 1))

        # North and east are already NED components.  Human-readable altitude
        # is positive upward, so it is the negative of Down.
        actual_series = (positions[:, 0], positions[:, 1], -positions[:, 2])
        target_series = (targets[:, 0], targets[:, 1], -targets[:, 2])
        for axis, lines, actual, target in zip(
                self.axes.flat[:3], self.position_lines,
                actual_series, target_series):
            lines[0].set_data(times, actual)
            lines[1].set_data(times, target)
            self._set_limits(axis, times, np.r_[actual, target])

        self.error_line.set_data(times, errors)
        self.rmse_line.set_data(times, running_rmse)
        self._set_limits(self.axes.flat[3], times, np.r_[errors, running_rmse],
                         lower_zero=True)

        error_ned = targets[-1] - positions[-1]
        self.title.set_text(
            f'{self.controller} live tracking | '
            f'error {errors[-1]:.3f} m | RMSE {running_rmse[-1]:.3f} m')
        self.status.set_text(
            f'Target NED [{targets[-1, 0]:+.2f}, {targets[-1, 1]:+.2f}, '
            f'{targets[-1, 2]:+.2f}] m    '
            f'Error NED [{error_ned[0]:+.3f}, {error_ned[1]:+.3f}, '
            f'{error_ned[2]:+.3f}] m')
        self.canvas.draw_idle()

    def _set_limits(self, axis, times, values, lower_zero=False):
        end = max(float(times[-1]), 1.0)
        axis.set_xlim(max(0.0, end - self.window_seconds),
                      max(self.window_seconds, end))
        finite = values[np.isfinite(values)]
        if not len(finite):
            return
        low, high = float(np.min(finite)), float(np.max(finite))
        margin = max(0.08 * (high - low), 0.03)
        axis.set_ylim(0.0 if lower_zero else low - margin, high + margin)

    def run(self):
        """Cooperate with Tk without blocking ROS subscription callbacks."""
        ros_poll_ms = 5
        redraw_ms = int(round(1000.0 / self.refresh_hz))

        def poll_ros():
            if self._closed or not rclpy.ok():
                return
            rclpy.spin_once(self, timeout_sec=0.0)
            self.root.after(ros_poll_ms, poll_ros)

        def redraw():
            if self._closed:
                return
            self._redraw()
            self.root.after(redraw_ms, redraw)

        self.root.after(0, poll_ros)
        self.root.after(redraw_ms, redraw)
        self.root.mainloop()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.root.quit()
        self.root.destroy()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrackingPerformanceOverlay()
        node.run()
    except (ImportError, RuntimeError) as exc:
        if node is not None:
            node.get_logger().error(
                f'cannot create plotting window: {exc}. '
                'Install python3-matplotlib and python3-tk, and run under WSLg.')
        else:
            raise
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
