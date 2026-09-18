"""Differentiable 6-DoF rigid-body quadrotor dynamics in JAX.

State is [position_NED(3), velocity_NED(3), quaternion_wxyz(4), body_rates(3)].
The body frame is FRD (x forward, y right, z down); NED z is positive down.
Actions are four normalized rotor commands in [0, 1].
"""
from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp


class QuadParams(NamedTuple):
    mass: jax.Array
    inertia: jax.Array
    arm: jax.Array
    max_thrust: jax.Array
    yaw_coeff: jax.Array
    linear_drag: jax.Array
    gravity: jax.Array


DEFAULT_PARAMS = QuadParams(
    mass=jnp.asarray(1.5), inertia=jnp.asarray([0.029, 0.029, 0.055]),
    arm=jnp.asarray(0.23), max_thrust=jnp.asarray(9.0),
    yaw_coeff=jnp.asarray(0.018), linear_drag=jnp.asarray([0.12, 0.12, 0.20]),
    gravity=jnp.asarray(9.80665),
)


def quat_normalize(q: jax.Array) -> jax.Array:
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-9)


def quat_to_rotation(q: jax.Array) -> jax.Array:
    """Body-to-NED rotation matrix for a wxyz quaternion."""
    w, x, y, z = quat_normalize(q)
    return jnp.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quat_derivative(q: jax.Array, omega: jax.Array) -> jax.Array:
    w, x, y, z = q
    p, r, s = omega
    return 0.5 * jnp.asarray([
        -x*p-y*r-z*s, w*p+y*s-z*r, w*r-x*s+z*p, w*s+x*r-y*p
    ])


def motor_wrench(action: jax.Array, p: QuadParams) -> tuple[jax.Array, jax.Array]:
    """X-configuration allocation: motor indices FR, FL, RL, RR."""
    f = jnp.clip(action, 0.0, 1.0) * p.max_thrust
    a = p.arm / jnp.sqrt(2.0)
    thrust_body = jnp.asarray([0.0, 0.0, -jnp.sum(f)])
    tau = jnp.asarray([
        a * (-f[0] + f[1] + f[2] - f[3]),
        a * ( f[0] + f[1] - f[2] - f[3]),
        p.yaw_coeff * (f[0] - f[1] + f[2] - f[3]),
    ])
    return thrust_body, tau


def derivatives(state: jax.Array, action: jax.Array, p: QuadParams,
                wind_ned: jax.Array) -> jax.Array:
    pos, vel, q, omega = state[:3], state[3:6], state[6:10], state[10:13]
    rotation = quat_to_rotation(q)
    thrust_b, torque_b = motor_wrench(action, p)
    relative_air = vel - wind_ned
    force_drag_n = -p.linear_drag * relative_air
    acceleration = rotation @ thrust_b / p.mass + force_drag_n / p.mass + jnp.asarray([0., 0., p.gravity])
    omega_dot = (torque_b - jnp.cross(omega, p.inertia * omega)) / p.inertia
    return jnp.concatenate([vel, acceleration, quat_derivative(q, omega), omega_dot])


def step_euler(state: jax.Array, action: jax.Array, p: QuadParams,
               dt: float, wind_ned: jax.Array) -> jax.Array:
    nxt = state + dt * derivatives(state, action, p, wind_ned)
    return nxt.at[6:10].set(quat_normalize(nxt[6:10]))


def step_rk4(state: jax.Array, action: jax.Array, p: QuadParams,
             dt: float, wind_ned: jax.Array) -> jax.Array:
    f = lambda x: derivatives(x, action, p, wind_ned)
    k1 = f(state); k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2); k4 = f(state + dt * k3)
    nxt = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return nxt.at[6:10].set(quat_normalize(nxt[6:10]))


step_jax = jax.jit(step_rk4, static_argnames=("dt",))
batched_step_jax = jax.jit(jax.vmap(step_rk4, in_axes=(0, 0, None, None, 0)), static_argnames=("dt",))


def hover_action(p: QuadParams = DEFAULT_PARAMS) -> jax.Array:
    return jnp.ones(4) * (p.mass * p.gravity / (4.0 * p.max_thrust))


def initial_state(position=(0.0, 0.0, 0.0)) -> jax.Array:
    return jnp.asarray([*position, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
