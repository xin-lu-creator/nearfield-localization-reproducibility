"""
File: werner_jansson_equivalence.py
Author: Xin Lu
Date: 2026-11-18
Description: Verify the two-state equivalence to the Werner–Jansson information bound.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nearfield_profile_ml import steering_nearfield,efficient_information,full_information_schur

def wj_single_source(theta,ran,pos,freq,Q,delta,n0,n1):
    """Whitened theorem-1 specialization: alpha=n1/n0, lambda=1+delta*s.
    I_WJ=2n1 delta^2*s/(lambda+alpha)*Re(D^H B_Q D), B_Q=Q^-1/2 Pi_perp Q^-1/2.
    This is the inverse of WJ's single-signal CRB.
    """
    a,D=steering_nearfield(theta,ran,pos,freq)
    d,U=np.linalg.eigh(Q);Z=(U*(d**-.5))@U.conj().T
    abar=Z@a;s=np.vdot(abar,abar).real
    u=abar/np.sqrt(s);proj=np.eye(len(a))-np.outer(u,u.conj())
    B=Z@proj@Z
    alpha=n1/n0;lam=1+delta*s
    return 2*n1*delta**2*s/(lam+alpha)*np.real(D.conj().T@B@D)

def test(n_cases=200,seed=20261110941):
    rng=np.random.default_rng(seed);errors=[]
    for i in range(n_cases):
        m=4+(i%2)
        xs=np.linspace(-.22,.22,m)+rng.normal(0,.006,m)
        pos=np.c_[xs,np.zeros(m),np.zeros(m)]
        ang=float(np.deg2rad(rng.uniform(-35,35)));ran=float(rng.uniform(.7,1.8));freq=1250.
        H=rng.normal(size=(m,m))+1j*rng.normal(size=(m,m))
        N=.5*np.eye(m)+H@H.conj().T/m
        p0=0. if i%3==0 else float(rng.uniform(.05,1.5))
        delta=float(rng.uniform(.3,1.8));p1=p0+delta
        n0=int(rng.integers(100,1300));n1=int(rng.integers(100,1300))
        a,_=steering_nearfield(ang,ran,pos,freq)
        Q=N+p0*np.outer(a,a.conj())
        WJ=wj_single_source(ang,ran,pos,freq,Q,delta,n0,n1)
        EF=efficient_information(ang,ran,pos,freq,N,[p0,p1],[n0,n1])
        # Independent nuisance projection. p0=0 is treated as the interior
        # analytic extension, not as a claim of a standard boundary CRB.
        full=full_information_schur(ang,ran,pos,freq,N,[p0,p1],[n0,n1])
        denom=max(1.,np.linalg.norm(EF))
        errors.append((np.linalg.norm(WJ-EF)/denom,np.linalg.norm(full-EF)/denom))
    e=np.array(errors)
    print(f'WJ vs gauge-invariant EFIM: {n_cases} cases, max rel={e[:,0].max():.3e}')
    print(f'full nuisance Fisher vs EFIM: max rel={e[:,1].max():.3e}')
    assert e.max()<1e-8
    return e
if __name__=='__main__':test()
