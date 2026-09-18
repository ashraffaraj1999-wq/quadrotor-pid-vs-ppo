from __future__ import annotations
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from src.env import Quadrotor6DoFEnv

def main():
    p=argparse.ArgumentParser(); p.add_argument('--timesteps',type=int,default=200_000); p.add_argument('--seed',type=int,default=7); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',type=Path); a=p.parse_args()
    env=Quadrotor6DoFEnv(); check_env(env, warn=True)
    Path('results').mkdir(exist_ok=True)
    env=Monitor(env, filename=f'results/ppo_seed_{a.seed}')
    out=Path('checkpoints'); out.mkdir(exist_ok=True)
    if a.resume:
        model=PPO.load(a.resume,env=env)
    else:
        model=PPO('MlpPolicy',env,learning_rate=3e-4,n_steps=1024,batch_size=256,gamma=0.995,gae_lambda=0.95,
                  policy_kwargs={'net_arch':[128,128],'log_std_init':-1.0},seed=a.seed,verbose=1)
    eval_env=Monitor(Quadrotor6DoFEnv(randomize=True))
    callback=EvalCallback(eval_env,best_model_save_path=f'checkpoints/best_seed_{a.seed}',log_path=f'results/eval_seed_{a.seed}',eval_freq=10_000,n_eval_episodes=8,deterministic=True)
    model.learn(total_timesteps=2048 if a.smoke else a.timesteps,progress_bar=False,callback=callback,reset_num_timesteps=not bool(a.resume))
    name='ppo_6dof_smoke' if a.smoke else f'ppo_6dof_seed_{a.seed}'
    model.save(out/name); print(out/f'{name}.zip')
if __name__=='__main__': main()
