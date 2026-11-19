"""
File: werner_jansson_wsf.py
Author: Xin Lu
Date: 2026-11-18
Description: Single-source Werner–Jansson whitened weighted subspace fitting benchmark.
"""
from __future__ import annotations
import argparse, csv, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular, cho_solve
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from nearfield_profile_ml import steering_nearfield, efficient_information
from run_power_contrast import POS,FREQ_HZ,P_MEAN,define_noises
from run_snapshot_scan import THETA,RANGE,NS,SEED
THETA_GRID=np.deg2rad(np.linspace(-65,65,27))
RANGE_GRID=np.array([.45,.6,.8,1.,1.3,1.8,2.5,3.])
GRID_THETA,GRID_R=np.meshgrid(THETA_GRID,RANGE_GRID,indexing='ij')
GRID_COORDS=np.stack([GRID_THETA.ravel(),GRID_R.ravel()],axis=1)
GRID_A=np.asarray([steering_nearfield(th,r,POS,FREQ_HZ)[0] for th,r in GRID_COORDS])
BOUNDS=[(-np.deg2rad(70),np.deg2rad(70)),(.4,3.)]


def wsf_fit(S_low, S_high, starts=4):
    """Return one-source whitened WSF nearfield estimate and diagnostics."""
    m=S_low.shape[0]
    L=np.linalg.cholesky((S_low+S_low.conj().T)/2)
    C=solve_triangular(L,S_high,lower=True,check_finite=False)
    C=solve_triangular(L,C.conj().T,lower=True,check_finite=False).conj().T
    eig,U=np.linalg.eigh((C+C.conj().T)/2)
    e=U[:,-1]
    c=solve_triangular(L.conj().T,e,lower=False,check_finite=False)
    K=cho_solve((L,True),np.eye(m,dtype=complex),check_finite=False)

    def score_and_jac(y):
        a,D=steering_nearfield(float(y[0]),float(y[1]),POS,FREQ_HZ)
        z=np.vdot(c,a)
        dz=np.conj(c)@D
        ka=K@a
        g=float(np.vdot(a,ka).real)
        dg=2*np.real(D.conj().T@ka)
        score=abs(z)**2/g
        ds=(2*np.real(np.conj(z)*dz)/g-(abs(z)**2/g**2)*dg)
        return -float(score), -ds.astype(float)

    z=GRID_A@np.conj(c)
    g=np.einsum('gm,mn,gn->g',GRID_A.conj(),K,GRID_A,optimize=True).real
    scores=np.abs(z)**2/g
    ids=np.argsort(scores)[::-1]
    trials=[]
    for i in ids:
        p=GRID_COORDS[i]
        if all(abs(p[0]-q[0])>.025 or abs(p[1]-q[1])>.085 for q in trials):trials.append(p)
        if len(trials)>=starts:break
    best=None
    for p in trials:
        res=minimize(score_and_jac, np.asarray(p),jac=True,bounds=BOUNDS,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-10,'gtol':1e-7,'maxls':40})
        if best is None or res.fun<best.fun: best=res
    return dict(theta_hat_deg=float(np.rad2deg(best.x[0])),range_hat_m=float(best.x[1]),
         score=float(-best.fun), success=int(best.success), iterations=int(best.nit),
         lambda1=float(eig[-1]),lambda2=float(eig[-2]),
         theta_boundary=int(abs(np.rad2deg(best.x[0]))>69.99),
         range_boundary=int(best.x[1]<=.4001 or best.x[1]>=2.9999))


def worker(job):
    name,rep=job
    N=define_noises()[name]
    seed=SEED+(0 if name=='A' else 1000000)+rep
    rng=np.random.default_rng(seed)
    z=[(rng.standard_normal((5,NS[-1]))+1j*rng.standard_normal((5,NS[-1])))/np.sqrt(2) for _ in range(2)]
    a,_=steering_nearfield(np.deg2rad(THETA),RANGE,POS,FREQ_HZ)
    A=np.outer(a,a.conj())
    delta=.75
    samples=[np.linalg.cholesky(N+p*A)@zz for p,zz in zip((P_MEAN-delta,P_MEAN+delta),z)]
    rows=[]
    for n in NS:
        Ss=[x[:,:n]@x[:,:n].conj().T/n for x in samples]
        fit=wsf_fit(Ss[0],Ss[1])
        rows.append(dict(noise=name,replicate=rep,seed=seed,delta=delta,n_per_state=n,
           theta_true_deg=THETA,range_true_m=RANGE,**fit))
    return rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=1000)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--start',type=int,default=0)
    p.add_argument('--output',type=str,default=str(ROOT/'reproduced/wsf/wsf_trials.csv'));args=p.parse_args()
    jobs=[(name,rep) for name in ('A','B') for rep in range(args.start,args.start+args.replicates)]
    t0=time.time();rows=[]
    if args.workers==1:
        result=map(worker,jobs)
        for k,r in enumerate(result,1):
            rows.extend(r)
            if k%100==0:print('done',k,'/',len(jobs),'time',round(time.time()-t0,1),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k,r in enumerate(pool.map(worker,jobs,chunksize=8),1):
                rows.extend(r)
                if k%100==0:print('done',k,'/',len(jobs),'time',round(time.time()-t0,1),flush=True)
    save_csv(Path(args.output),rows)
    print('COMPLETE',len(rows),'fits',round(time.time()-t0,1),'sec',args.output,flush=True)
if __name__=='__main__':main()
