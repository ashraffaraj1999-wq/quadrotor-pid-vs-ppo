"""Gymnasium wrapper around the JAX 6-DoF dynamics."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jax.numpy as jnp
from .dynamics_jax import DEFAULT_PARAMS, initial_state, step_jax


class Quadrotor6DoFEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, dt=0.01, max_steps=1000, target=(0., 0., -2.), randomize=True):
        super().__init__(); self.dt=dt; self.max_steps=max_steps
        self.target=np.asarray(target, np.float32); self.randomize=randomize
        # Symmetric residual policy action. Zero means model-based hover; this avoids
        # asking PPO to discover gravity compensation before it can learn stabilization.
        self.action_space=spaces.Box(-1.,1.,shape=(4,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(19,),dtype=np.float32)
        self.params=DEFAULT_PARAMS; self.state=np.array(initial_state(),dtype=np.float32,copy=True); self.steps=0

    def _obs(self):
        return np.r_[self.state, self.target-self.state[:3], -self.state[3:6]].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.steps=0
        self.state=np.array(initial_state(self.target),dtype=np.float32,copy=True)
        if self.randomize:
            self.state[:3]=self.target+self.np_random.uniform([-0.5,-0.5,-0.2],[0.5,0.5,0.2])
            self.state[3:6]=self.np_random.normal(0,0.10,3)
            self.state[10:13]=self.np_random.normal(0,0.05,3)
        return self._obs(), {}

    def step(self, action):
        wind=np.zeros(3,np.float32)
        hover=float(self.params.mass*self.params.gravity/(4.0*self.params.max_thrust))
        motor_action=np.clip(hover+0.35*np.asarray(action),0.,1.)
        self.state=np.asarray(step_jax(jnp.asarray(self.state),jnp.asarray(motor_action),self.params,self.dt,jnp.asarray(wind)),dtype=np.float32)
        self.steps+=1
        ep=np.linalg.norm(self.target-self.state[:3]); ev=np.linalg.norm(self.state[3:6])
        tilt=1.0-(self.state[6]**2+self.state[9]**2-self.state[7]**2-self.state[8]**2)
        effort=np.mean(np.square(motor_action-0.41))
        # Positive bounded living reward: surviving near the reference always beats
        # deliberately terminating to avoid future costs.
        reward=float(1.50*np.exp(-0.70*ep*ep) + 0.25*np.exp(-0.50*ev*ev)
                     + 0.25*np.exp(-4.0*tilt*tilt) - 0.02*effort)
        terminated=bool(np.linalg.norm(self.state[:3])>8 or self.state[2]>1.0 or abs(self.state[6])<0.17)
        truncated=self.steps>=self.max_steps
        # Without a terminal penalty PPO can improve return by crashing early and
        # avoiding future tracking costs. This makes failure strictly unattractive.
        if terminated:
            reward -= 100.0
        return self._obs(), reward, terminated, truncated, {"position_error":float(ep),"control_effort":float(effort)}

    def render(self):
        img=np.full((256,256,3),245,np.uint8); x=int(128+20*self.state[1]); z=int(220+20*self.state[2])
        if 4<=x<252 and 4<=z<252: img[z-4:z+5,x-4:x+5]=[24,110,180]
        return img
