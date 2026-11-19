"""
File: power_coupled_diffuse_robustness.py
Author: Xin Lu
Date: 2026-11-18
Description: Matched-sample diffuse-field model-mismatch experiment.
"""
from __future__ import annotations
import argparse,csv,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from nearfield_profile_ml import steering_nearfield, NearfieldProfileProblem
from run_power_contrast import POS,FREQ_HZ,P_MEAN,define_noises
from run_snapshot_scan import THETA,RANGE,SEED,NS
from werner_jansson_wsf import wsf_fit
N_SNAP=768
EPS=(0.,.05,.15,.30)
C=343.0

def diffuse_kernel():
    ds=np.linalg.norm(POS[:,None,:]-POS[None,:,:],axis=-1)
    k=2*np.pi*FREQ_HZ/C
    # np.sinc(x) = sin(pi*x)/(pi*x)
    R=np.sinc(k*ds/np.pi)
    return R/np.trace(R).real

def worker(job):
    name,rep,levels,maxiter=job
    N=define_noises()[name]
    seed=SEED+(0 if name=='A' else 1000000)+rep
    rng=np.random.default_rng(seed)
    # Same normal draws as frozen 0.75-contrast scan: entire 3072 sample vectors per state
    Z=[(rng.standard_normal((5,NS[-1]))+1j*rng.standard_normal((5,NS[-1])))/np.sqrt(2) for _ in range(2)]
    a,_=steering_nearfield(np.deg2rad(THETA),RANGE,POS,FREQ_HZ)
    A=np.outer(a,a.conj())
    Rd=diffuse_kernel()
    p=[P_MEAN-.75,P_MEAN+.75]
    out=[]
    for eps in levels:
        S=[]
        for pt,z in zip(p,Z):
            cov=N+pt*(A+eps*Rd)
            L=np.linalg.cholesky(cov)
            X=L@z[:,:N_SNAP]
            S.append(X@X.conj().T/N_SNAP)
        fit=NearfieldProfileProblem(np.asarray(S),np.array([N_SNAP,N_SNAP]),POS,FREQ_HZ).estimate_multistart(maxiter=maxiter)
        wfit=wsf_fit(S[0],S[1])
        for method,values in [('Profile ML',dict(theta_hat_deg=fit['theta_deg'],range_hat_m=fit['range_m'],success=int(fit['success']),
                    objective=fit['profile_nll'])),('WJ-WSF',dict(theta_hat_deg=wfit['theta_hat_deg'],range_hat_m=wfit['range_hat_m'],success=wfit['success'],objective=wfit['score']))]:
            out.append(dict(noise=name,replicate=rep,seed=seed,delta=.75,n_per_state=N_SNAP,
             eps=eps,method=method,theta_true_deg=THETA,range_true_m=RANGE,
             theta_hat_deg=values['theta_hat_deg'],range_hat_m=values['range_hat_m'],
             success=values['success'],objective=values['objective'],
             range_boundary=int(values['range_hat_m']<=.4001 or values['range_hat_m']>=2.9999)))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=120)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--maxiter',type=int,default=120)
    p.add_argument('--levels',type=str,default='0.05,0.15,0.30')
    p.add_argument('--output',type=str,default=str(ROOT/'reproduced/mismatch/mismatch_trials.csv'))
    args=p.parse_args();levels=[float(s) for s in args.levels.split(',')]
    tasks=[(name,rep,levels,args.maxiter) for name in ('A','B') for rep in range(args.replicates)]
    st=time.time();rows=[]
    if args.workers==1:
        for k,j in enumerate(tasks,1):
            rows.extend(worker(j))
            if k%20==0:print('done',k,'/',len(tasks),'time',round(time.time()-st,1),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k,r in enumerate(pool.map(worker,tasks,chunksize=1),1):
                rows.extend(r)
                if k%20==0:print('done',k,'/',len(tasks),'time',round(time.time()-st,1),flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('COMPLETE',len(rows),'fits',round(time.time()-st,1),'sec',output,flush=True)
if __name__=='__main__':main()
