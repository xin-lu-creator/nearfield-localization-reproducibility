"""
File: run_multistate_high_snapshots.py
Author: Xin Lu
Date: 2026-11-18
Description: Large-sample three-state experiment using exact complex-Wishart sampling.
"""
from __future__ import annotations
import os,sys,argparse,csv,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src'),str(R/'experiments')]
from nearfield_profile_ml import NearfieldProfileProblem,steering_nearfield,efficient_information
from run_power_contrast import POS,FREQ_HZ,define_noises
TH,RR=21.7,.93
P=(.85,1.60,2.35)
SEED=20261110500

def sample_wishart(rng,n,m):
    """Sample (Z Z^H)/n exactly in law, Z_ij independent CN(0,1)."""
    B=np.zeros((m,m),complex)
    for i in range(m):
        B[i,i]=np.sqrt(rng.gamma(n-i))
        for j in range(i):B[i,j]=(rng.standard_normal()+1j*rng.standard_normal())/np.sqrt(2)
    return B@B.conj().T/n

def one(job):
    noise,rep,n,maxiter=job
    seed=SEED+(0 if noise=='A' else 1000000)+rep
    rng=np.random.default_rng(seed)
    N=define_noises()[noise];a,_=steering_nearfield(np.deg2rad(TH),RR,POS,FREQ_HZ)
    aa=np.outer(a,a.conj());stats=[]
    for p in P:
        L=np.linalg.cholesky(N+p*aa)
        stats.append(L@sample_wishart(rng,n,len(a))@L.conj().T)
    fit=NearfieldProfileProblem(np.array(stats),np.repeat(n,3),POS,FREQ_HZ).estimate_multistart(maxiter=maxiter)
    return dict(noise=noise,replicate=rep,seed=seed,T=3,n_per_state=n,n_total=3*n,
        theta_true_deg=TH,range_true_m=RR,theta_hat_deg=float(fit['theta_deg']),range_hat_m=float(fit['range_m']),
        profile_nll=float(fit['profile_nll']),starts_tried=int(fit['starts_tried']),
        converged_starts=int(fit['converged_starts']),optimizer_success=int(fit['success']))

def save(path,rows):
    if not rows:return
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--replicates',type=int,default=4000)
    ap.add_argument('--jobs',type=int,default=4)
    ap.add_argument('--noise',choices=['A','B','both'],default='A')
    ap.add_argument('--n-per-state',type=int,default=16384)
    ap.add_argument('--bootstrap',type=int,default=4000)
    ap.add_argument('--maxiter',type=int,default=120)
    a=ap.parse_args()
    out=R/'reproduced'/'high_snapshot';out.mkdir(exist_ok=True,parents=True)
    noises=['A','B'] if a.noise=='both' else [a.noise]
    jobs=[(noise,i,a.n_per_state,a.maxiter) for noise in noises for i in range(a.replicates)]
    rows=[];tic=time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for i,v in enumerate(pool.map(one,jobs,chunksize=5),1):
            rows.append(v)
            if i%100==0 or i==len(jobs):
                save(out/'high_snapshot_trials.csv',rows)
                print(f'[HIGH] {i}/{len(jobs)}; elapsed={time.time()-tic:.1f}s',flush=True)
    rows.sort(key=lambda r:(r['noise'],r['replicate']));save(out/'high_snapshot_trials.csv',rows)
    summary=[]
    for noise in noises:
        vals=[v for v in rows if v['noise']==noise];N=define_noises()[noise]
        F=efficient_information(np.deg2rad(TH),RR,POS,FREQ_HZ,N,P,[a.n_per_state]*3);C=np.linalg.inv(F)
        E=np.array([[v['theta_hat_deg']-TH,v['range_hat_m']-RR] for v in vals])
        X=np.array([(E[:,0]*np.pi/180)**2/C[0,0], E[:,1]**2/C[1,1]])
        rng=np.random.default_rng(2026111098+('A'!=noise));ix=rng.integers(0,len(vals),(a.bootstrap,len(vals)))
        ci=np.quantile(X[:,ix].mean(axis=2),[.025,.975],axis=1)
        summary.append(dict(noise=noise,T=3,n_per_state=a.n_per_state,n_total=3*a.n_per_state,replicates=len(vals),
          angle_ratio=float(X[0].mean()),range_ratio=float(X[1].mean()),
          angle_ci_low=float(ci[0,0]),angle_ci_high=float(ci[1,0]),
          range_ci_low=float(ci[0,1]),range_ci_high=float(ci[1,1]),
          empirical_correlation=float(np.corrcoef(E.T)[0,1]),
          theoretical_correlation=float(C[0,1]/np.sqrt(C[0,0]*C[1,1])),
          range_rmse_m=float(np.sqrt(np.mean(E[:,1]**2))),
          bearing_rmse_deg=float(np.sqrt(np.mean(E[:,0]**2))),
          optimizer_success_fraction=float(np.mean([v['optimizer_success'] for v in vals]))))
    save(out/'high_snapshot_summary.csv',summary)
    (out/'high_snapshot_config.json').write_text(json.dumps(dict(seed_base=SEED,noise=a.noise,
      replicates_per_noise=a.replicates,n_per_state=a.n_per_state,powers=P,
      theta_true_deg=TH,range_true_m=RR,positions_m=POS.tolist(),frequency_hz=FREQ_HZ,
      simulation='EXACT circular complex Gaussian sample covariance via complex Wishart Bartlett factor',
      optimizer='four-start L-BFGS-B local Gaussian profile, no oracle start',
      bootstrap_replicates=a.bootstrap),indent=2),encoding='utf8')
    print('[SUMMARY]',json.dumps(summary));print('[DONE]',len(rows),flush=True)
if __name__=='__main__':main()
