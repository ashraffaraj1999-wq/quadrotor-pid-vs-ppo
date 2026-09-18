"""Render a side-by-side 3D PID/PPO flight animation and export replay data."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
from evaluate import rollout
from src.pid import CascadedPID, quat_to_rot
from src.dynamics_jax import DEFAULT_PARAMS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ppo',type=Path,default=Path('checkpoints/ppo_6dof_seed_9.zip')); ap.add_argument('--out',type=Path,default=Path('results/final_comparison')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    model=PPO.load(a.ppo); hover=float(DEFAULT_PARAMS.mass*DEFAULT_PARAMS.gravity/(4*DEFAULT_PARAMS.max_thrust))
    ppo=lambda s: np.clip(hover+.35*model.predict(np.r_[s,[0,0,-2]-s[:3],-s[3:6]].astype(np.float32),deterministic=True)[0],0,1)
    runs={'PID':rollout(CascadedPID(),seed=11),'PPO (BC+RL)':rollout(ppo,seed=11)}
    np.savez_compressed(a.out/'comparison_trajectories.npz',pid=runs['PID']['state'],ppo=runs['PPO (BC+RL)']['state'],dt=.01)
    fig=plt.figure(figsize=(10,5)); axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    colors={'PID':'#1676b8','PPO (BC+RL)':'#e07a24'}; artists=[]
    for ax,name in zip(axes,runs):
        ax.set(xlim=(-.7,.7),ylim=(-.7,.7),zlim=(1.5,2.35),xlabel='East (m)',ylabel='North (m)',zlabel='Altitude (m)',title=name)
        ax.scatter([0],[0],[2],marker='*',s=100,c='green',label='target'); ax.legend(loc='upper right')
        trail,=ax.plot([],[],[],lw=2,c=colors[name]); arm1,=ax.plot([],[],[],lw=4,c=colors[name]); arm2,=ax.plot([],[],[],lw=4,c=colors[name]); label=ax.text2D(.03,.94,'',transform=ax.transAxes)
        artists.append((trail,arm1,arm2,label))
    indices=np.arange(0,min(len(x['state']) for x in runs.values()),5)
    def update(frame):
        k=indices[frame]
        for (name,r),bundle in zip(runs.items(),artists):
            s=r['state']; pos=s[k,:3]; enu=np.array([pos[1],pos[0],-pos[2]]); hist=s[:k+1,:3]
            trail,arm1,arm2,label=bundle; trail.set_data(hist[:,1],hist[:,0]); trail.set_3d_properties(-hist[:,2])
            R=quat_to_rot(s[k,6:10]); arms=np.array([[.22,.22,0],[.22,-.22,0],[-.22,-.22,0],[-.22,.22,0]])
            pts=(R@arms.T).T+pos; pe=np.c_[pts[:,1],pts[:,0],-pts[:,2]]
            arm1.set_data(pe[[0,2],0],pe[[0,2],1]); arm1.set_3d_properties(pe[[0,2],2]); arm2.set_data(pe[[1,3],0],pe[[1,3],1]); arm2.set_3d_properties(pe[[1,3],2])
            err=np.linalg.norm(pos-np.array([0,0,-2])); label.set_text(f't={k*.01:4.1f}s  error={err:.3f}m')
        return [x for b in artists for x in b]
    ani=FuncAnimation(fig,update,frames=len(indices),interval=50,blit=False); ani.save(a.out/'pid_vs_ppo_3d.gif',writer=PillowWriter(fps=20)); plt.close(fig)
    print(a.out/'pid_vs_ppo_3d.gif')
if __name__=='__main__': main()
