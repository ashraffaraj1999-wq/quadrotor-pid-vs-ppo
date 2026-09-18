"""Cascaded position/attitude controller with geometric attitude error."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass
class CascadedPID:
    mass: float = 1.5
    gravity: float = 9.80665
    inertia: np.ndarray = field(default_factory=lambda: np.array([0.029, 0.029, 0.055]))
    arm: float = 0.23
    max_thrust: float = 9.0
    yaw_coeff: float = 0.018
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([2.2, 2.2, 4.0]))
    kd_pos: np.ndarray = field(default_factory=lambda: np.array([2.8, 2.8, 3.2]))
    kp_att: np.ndarray = field(default_factory=lambda: np.array([7.5, 7.5, 3.0]))
    kd_att: np.ndarray = field(default_factory=lambda: np.array([0.34, 0.34, 0.22]))

    def command(self, state: np.ndarray, target_ned: np.ndarray, yaw: float = 0.0) -> np.ndarray:
        pos, vel, q, rates = state[:3], state[3:6], state[6:10], state[10:13]
        # Required total force in NED; gravity is +z in NED.
        acc_cmd = self.kp_pos * (target_ned - pos) - self.kd_pos * vel
        force_n = self.mass * (acc_cmd - np.array([0., 0., self.gravity]))
        thrust = np.linalg.norm(force_n)
        b3 = -force_n / max(thrust, 1e-9)  # desired body-down axis
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
        b2 = np.cross(b3, heading); b2 /= max(np.linalg.norm(b2), 1e-9)
        b1 = np.cross(b2, b3)
        rd = np.column_stack((b1, b2, b3))
        r = quat_to_rot(q)
        skew = 0.5 * (rd.T @ r - r.T @ rd)
        e_r = np.array([skew[2,1], skew[0,2], skew[1,0]])
        tau = -self.kp_att * e_r - self.kd_att * rates
        a = self.arm / np.sqrt(2.0)
        allocation = np.array([[1,1,1,1],[-a,a,a,-a],[a,a,-a,-a],
                               [self.yaw_coeff,-self.yaw_coeff,self.yaw_coeff,-self.yaw_coeff]])
        rotor_forces = np.linalg.solve(allocation, np.r_[thrust, tau])
        return np.clip(rotor_forces / self.max_thrust, 0.0, 1.0)

