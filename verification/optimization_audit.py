"""
File: optimization_audit.py
Author: Xin Lu
Date: 2026-11-18
Description: Check the weak-contrast multistart likelihood diagnostic.
"""
from __future__ import annotations
import argparse,csv,sys,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'experiments')]
from nearfield_profile_ml import NearfieldProfileProblem,steering_nearfield
from run_power_contrast import POS,TH,RANGE_M,FREQ_HZ,BASE_SEED,P_MEAN,N_PER_STATE,define_noises

def read(name):
 with (R/'data'/name).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def check():
 rows=read('optimization_audit_trials.csv');summ=read('optimization_audit_summary.csv');frozen=read('power_contrast_trials.csv')
 assert len(rows)==len(frozen)==1200 and len(summ)==10
 keys=lambda z:(z['noise'],int(z['replicate']),float(z['delta']))
 archive={keys(r):r for r in frozen};assert len(archive)==1200
 counts=collections.Counter()
 for r in rows:
  k=keys(r);base=archive[k]
  assert abs(float(r['theta_base_deg'])-float(base['theta_hat_deg']))<1e-9
  assert abs(float(r['range_base_m'])-float(base['range_hat_m']))<1e-9
  assert abs(float(r['nll_base'])-float(base['nll']))<1e-7
  flag=(float(r['nll_base'])-float(r['nll_dense'])>1e-3)
  assert flag==bool(int(r['better_nll_found']))
  counts[(r['noise'],float(r['delta']))]+=int(flag)
 assert counts['A',.35]==4 and counts['B',.35]==6
 assert sum(v for (noise,d),v in counts.items() if d>=.75)==0
 print('[PASS] frozen audit rows=1200; delta=.35: A=4, B=6 better NLL; delta>=.75: 0')

def rerun(noise,rep,delta):
 N=define_noises()[noise];a,_=steering_nearfield(TH,RANGE_M,POS,FREQ_HZ)
 seed=BASE_SEED+(0 if noise=='A' else 1000000)+rep
 rng=np.random.default_rng(seed);n=N_PER_STATE
 zs=[(rng.standard_normal((5,n))+1j*rng.standard_normal((5,n)))/np.sqrt(2) for _ in range(2)]
 A=np.outer(a,a.conj());states=[]
 for p,Z in zip([P_MEAN-delta,P_MEAN+delta],zs):
  X=np.linalg.cholesky(N+p*A)@Z;states.append(X@X.conj().T/n)
 pb=NearfieldProfileProblem(np.array(states),np.array([n,n]),POS,FREQ_HZ)
 base=pb.estimate_multistart(maxiter=120)
 # 29 starts = one truth-based numerical AUDIT start plus blind grid.
 pts=[(TH,RANGE_M)]+[(float(th),float(ra)) for th in np.deg2rad([-60,-40,-20,0,20,40,60]) for ra in [.5,.8,1.3,2.2]]
 dense=[pb.estimate(th,ra,maxiter=3000) for th,ra in pts]
 best=min(dense,key=lambda z:z['profile_nll'])
 print(dict(noise=noise,replicate=rep,delta=delta,seed=seed,
            nll_multistart=base['profile_nll'],nll_denser=best['profile_nll'],
            improvement=base['profile_nll']-best['profile_nll']))
 return base,best
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rerun',action='store_true')
 p.add_argument('--noise',choices=['A','B'],default='A');p.add_argument('--replicate',type=int,default=0)
 p.add_argument('--delta',type=float,default=.35);a=p.parse_args()
 check()
 if a.rerun:rerun(a.noise,a.replicate,a.delta)
