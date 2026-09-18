"""Deterministic PID/PPO evaluation and plot generation."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.dynamics_jax import DEFAULT_PARAMS, initial_state, step_jax
from src.pid import CascadedPID

def rollout(controller, seed=11, mass_scale=1.0, wind=(0.,0.,0.), seconds=10.):
    rng=np.random.default_rng(seed); dt=.01; n=int(seconds/dt); target=np.array([0.,0.,-2.])
    p0=target+rng.uniform([-.4,-.4,-.15],[.4,.4,.15])
    state=np.asarray(initial_state(tuple(p0)),float); state[3:6]=rng.normal(0,.05,3); state[10:13]=rng.normal(0,.03,3)
    states=[]; actions=[]; failed=False
    params=DEFAULT_PARAMS._replace(mass=DEFAULT_PARAMS.mass*mass_scale)
    for k in range(n):
        action=controller.command(state,target) if hasattr(controller,'command') else controller(state)
        gust=np.asarray(wind) if 2.0<k*dt<3.0 else np.zeros(3)
        state=np.asarray(step_jax(jnp.asarray(state),jnp.asarray(action),params,dt,jnp.asarray(gust)))
        states.append(state.copy()); actions.append(np.asarray(action))
        failed=bool(np.linalg.norm(state[:3])>8 or state[2]>1.0 or abs(state[6])<0.17)
        if failed: break
    s=np.asarray(states); a=np.asarray(actions); err=np.linalg.norm(s[:,:3]-target,axis=1)
    return {'state':s,'action':a,'rmse':float(np.sqrt(np.mean(err**2))),
            'final_error':float(err[-1]),'effort':float(np.mean(np.sum(a*a,axis=1))),
            'failed':failed,'duration_s':len(s)*dt}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ppo',type=Path); ap.add_argument('--ppo-label',default='PPO'); ap.add_argument('--out',type=Path,default=Path('results')); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    controllers={'PID':CascadedPID()}
    if args.ppo:
        from stable_baselines3 import PPO
        m=PPO.load(args.ppo); hover=float(DEFAULT_PARAMS.mass*DEFAULT_PARAMS.gravity/(4*DEFAULT_PARAMS.max_thrust))
        controllers[args.ppo_label]=lambda s: np.clip(hover+0.35*m.predict(np.r_[s,[0,0,-2]-s[:3],-s[3:6]].astype(np.float32),deterministic=True)[0],0.,1.)
    scenarios={'nominal':(1.,(0,0,0)),'wind_impulse':(1.,(2.5,0,0)),'mass_plus_20pct':(1.2,(0,0,0))}
    rows=[]; nominal={}
    for cname,c in controllers.items():
        for sname,(ms,w) in scenarios.items():
            metrics=[]
            for seed in (11,22,33):
                r=rollout(c,seed,ms,w); metrics.append(r)
            rows.append({'controller':cname,'scenario':sname,'position_rmse_m':np.mean([x['rmse'] for x in metrics]),'final_error_m':np.mean([x['final_error'] for x in metrics]),'mean_squared_action':np.mean([x['effort'] for x in metrics]),'success_rate':np.mean([not x['failed'] for x in metrics]),'mean_duration_s':np.mean([x['duration_s'] for x in metrics])})
            if sname=='nominal': nominal[cname]=metrics[0]
    with open(args.out/'metrics.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]); wr.writeheader(); wr.writerows(rows)
    fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for name,r in nominal.items():
        t=np.arange(len(r['state']))*.01
        ax[0,0].plot(t,r['state'][:,0],label=name); ax[0,1].plot(t,-r['state'][:,2],label=name)
        ax[1,0].plot(r['state'][:,1],r['state'][:,0],label=name); ax[1,1].plot(t,np.linalg.norm(r['state'][:,10:13],axis=1),label=name)
    ax[0,0].set(title='North position',ylabel='m'); ax[0,1].set(title='Altitude',ylabel='m'); ax[1,0].set(title='Horizontal path',xlabel='East (m)',ylabel='North (m)'); ax[1,1].set(title='Body-rate norm',xlabel='time (s)',ylabel='rad/s')
    for x in ax.flat: x.grid(alpha=.25); x.legend();
    fig.savefig(args.out/'nominal_tracking.png',dpi=180); plt.close(fig)
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
