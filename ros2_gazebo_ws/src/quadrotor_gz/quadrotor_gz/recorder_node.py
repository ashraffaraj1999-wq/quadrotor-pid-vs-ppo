"""Record Gazebo odometry and write tracking metrics for matched trials."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class FlightRecorder(Node):
    def __init__(self):
        super().__init__('quadrotor_flight_recorder')
        self.declare_parameter('output', 'gazebo_trial.csv')
        self.declare_parameter('duration', 10.0)
        self.declare_parameter('target_enu', [0.0, 0.0, 2.0])
        self.output = Path(str(self.get_parameter('output').value)).expanduser()
        self.duration = float(self.get_parameter('duration').value)
        self.target = np.asarray(self.get_parameter('target_enu').value, dtype=float)
        self.t0 = None
        self.rows = []
        self.done = False
        self.create_subscription(Odometry, '/quadrotor/odometry', self._odom, 50)

    def _odom(self, msg: Odometry):
        if self.done:
            return
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.t0 is None:
            self.t0 = stamp
        t = stamp - self.t0
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        w = msg.twist.twist.angular
        self.rows.append([t, p.x, p.y, p.z, v.x, v.y, v.z, q.w, q.x, q.y, q.z, w.x, w.y, w.z])
        if t >= self.duration:
            self._write()

    def _write(self):
        if self.done or not self.rows:
            return
        self.done = True
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['time', 'x_enu', 'y_enu', 'z_enu', 'vx', 'vy', 'vz',
                             'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz'])
            writer.writerows(self.rows)
        position = np.asarray(self.rows)[:, 1:4]
        error = np.linalg.norm(position - self.target, axis=1)
        metrics_path = self.output.with_name(self.output.stem + '_metrics.csv')
        with metrics_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['samples', 'duration_s', 'position_rmse_m', 'final_error_m', 'max_error_m'])
            writer.writerow([len(error), self.rows[-1][0], np.sqrt(np.mean(error**2)),
                             error[-1], np.max(error)])
        self.get_logger().info(f'wrote {self.output} and {metrics_path}')

    def destroy_node(self):
        self._write()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FlightRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
