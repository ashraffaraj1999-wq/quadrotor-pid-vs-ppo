import numpy as np
import jax.numpy as jnp
from src.dynamics_jax import DEFAULT_PARAMS, hover_action, initial_state, step_jax, motor_wrench

def test_hover_has_negligible_vertical_acceleration():
    s=np.asarray(initial_state()); n=np.asarray(step_jax(jnp.asarray(s),hover_action(),DEFAULT_PARAMS,.001,jnp.zeros(3)))
    assert abs(n[5]) < 1e-4
def test_free_fall_accelerates_down_ned():
    s=initial_state(); n=step_jax(s,jnp.zeros(4),DEFAULT_PARAMS,.01,jnp.zeros(3)); assert float(n[5]) > 0
def test_equal_motors_have_zero_torque():
    _,tau=motor_wrench(jnp.ones(4)*.4,DEFAULT_PARAMS); np.testing.assert_allclose(tau,0,atol=1e-7)
def test_quaternion_stays_normalized():
    s=initial_state().at[10].set(1.0); n=step_jax(s,hover_action(),DEFAULT_PARAMS,.01,jnp.zeros(3)); assert abs(float(jnp.linalg.norm(n[6:10]))-1)<1e-6

