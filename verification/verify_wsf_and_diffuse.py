"""
File: verify_wsf_and_diffuse.py
Author: Xin Lu
Date: 2026-11-18
Description: Check WSF whitening and diffuse-field covariance calculations.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.linalg import solve_triangular,cho_solve
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'src'))
from run_power_contrast import POS,FREQ_HZ,P_MEAN,define_noises
from run_snapshot_scan import THETA,RANGE
from nearfield_profile_ml import steering_nearfield
from werner_jansson_wsf import wsf_fit
from power_coupled_diffuse_robustness import diffuse_kernel

rng=np.random.default_rng(2026111799)
err=[];grad_err=[]
for i in range(100):
 Z=rng.standard_normal((5,5))+1j*rng.standard_normal((5,5))
 Q=Z@Z.conj().T/5+np.eye(5)
 J=rng.standard_normal((5,5))+1j*rng.standard_normal((5,5))
 S=J@J.conj().T/5+Q
 L=np.linalg.cholesky(Q)
 X=solve_triangular(L,S,lower=True)
 C=solve_triangular(L,X.conj().T,lower=True).conj().T
 eigen,U=np.linalg.eigh(C)
 e=U[:,-1]
 c=solve_triangular(L.conj().T,e,lower=False)
 K=cho_solve((L,True),np.eye(5))
 th=rng.uniform(-.6,.6);r=rng.uniform(.5,2.5)
 a,D=steering_nearfield(th,r,POS,FREQ_HZ)
 y=solve_triangular(L,a,lower=True)
 direct=abs(np.vdot(e,y))**2/np.vdot(y,y).real
 fast=abs(np.vdot(c,a))**2/np.vdot(a,K@a).real
 err.append(abs(fast-direct))
 z=np.vdot(c,a);dz=np.conj(c)@D;ka=K@a;g=np.vdot(a,ka).real
 dgd=2*np.real(D.conj().T@ka)
 jac=2*np.real(np.conj(z)*dz)/g-(abs(z)**2/g**2)*dgd
 def sc(t,rr):
  a2,_=steering_nearfield(t,rr,POS,FREQ_HZ)
  return float(abs(np.vdot(c,a2))**2/np.vdot(a2,K@a2).real)
 steps=np.array([1e-6,1e-6]); fd=np.array([(sc(th+h,r)-sc(th-h,r))/(2*h) if j==0 else (sc(th,r+h)-sc(th,r-h))/(2*h) for j,h in enumerate(steps)])
 grad_err.append(np.linalg.norm(jac-fd))

pop=[]
a,_=steering_nearfield(np.deg2rad(THETA),RANGE,POS,FREQ_HZ)
for noise,N in define_noises().items():
 S0=N+(P_MEAN-.75)*np.outer(a,a.conj())
 S1=N+(P_MEAN+.75)*np.outer(a,a.conj())
 est=wsf_fit(S0,S1)
 pop.append(dict(noise=noise,angle_error_deg=float(est['theta_hat_deg']-THETA),range_error_m=float(est['range_hat_m']-RANGE)))

Rd=diffuse_kernel();evals=np.linalg.eigvalsh(Rd)
assert evals.min()>-1e-12
assert np.max(np.abs(np.diag(Rd)-.2))<1e-12
assert max(err)<1e-12
assert max(grad_err)<1e-7
assert max(abs(v['range_error_m']) for v in pop)<1e-3
ans=dict(whitened_projection_identity_max_abs_error=max(err),jacobian_central_difference_max_norm=max(grad_err),
 diffuse_min_eigenvalue=float(evals.min()),diffuse_trace=float(np.trace(Rd).real),
 population_wsf_recovery=pop,checks=100)
p=ROOT/'reproduced/verification_new_experiments.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ans,indent=2))
print(json.dumps(ans,indent=2))
