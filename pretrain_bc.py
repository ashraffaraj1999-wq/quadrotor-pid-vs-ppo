"""Warm-start an SB3 PPO actor from PID demonstrations, then save for RL fine-tuning."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import torch
from stable_baselines3 import PPO
from src.dynamics_jax import DEFAULT_PARAMS, initial_state, step_jax
from src.pid import CascadedPID

def dataset(episodes: int, steps: int, seed: int):
    rng=np.random.default_rng(seed); target=np.array([0.,0.,-2.]); pid=CascadedPID(); obs=[]; act=[]
    hover=float(DEFAULT_PARAMS.mass*DEFAULT_PARAMS.gravity/(4*DEFAULT_PARAMS.max_thrust))
    for _ in range(episodes):
        state=np.asarray(initial_state(tuple(target+rng.uniform([-.5,-.5,-.2],[.5,.5,.2]))),float)
        state[3:6]=rng.normal(0,.10,3); state[10:13]=rng.normal(0,.05,3)
        for _ in range(steps):
            motor=pid.command(state,target); policy_action=np.clip((motor-hover)/.35,-1.,1.)
            obs.append(np.r_[state,target-state[:3],-state[3:6]].astype(np.float32)); act.append(policy_action.astype(np.float32))
            state=np.asarray(step_jax(jnp.asarray(state),jnp.asarray(motor),DEFAULT_PARAMS,.01,jnp.zeros(3)))
    return np.asarray(obs),np.asarray(act)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('checkpoints/ppo_bc_warmstart')); ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--seed',type=int,default=8); a=ap.parse_args()
    x,y=dataset(episodes=160,steps=300,seed=a.seed); model=PPO.load(a.base); device=model.device
    opt=torch.optim.Adam(model.policy.parameters(),lr=8e-4); rng=np.random.default_rng(a.seed)
    for epoch in range(a.epochs):
        order=rng.permutation(len(x)); losses=[]
        for start in range(0,len(x),512):
            idx=order[start:start+512]; xb=torch.as_tensor(x[idx],device=device); yb=torch.as_tensor(y[idx],device=device)
            mean=model.policy.get_distribution(xb).distribution.mean
            loss=torch.nn.functional.mse_loss(mean,yb); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f'epoch={epoch+1} mse={np.mean(losses):.6f}')
    a.out.parent.mkdir(parents=True,exist_ok=True); model.save(a.out); print(a.out.with_suffix('.zip'))
if __name__=='__main__': main()
