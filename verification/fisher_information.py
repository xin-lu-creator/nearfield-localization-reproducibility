"""
File: fisher_information.py
Author: Xin Lu
Date: 2026-11-18
Description: Independent full nuisance-Fisher projection verification.
"""
import csv
from pathlib import Path
import numpy as np
rng=np.random.default_rng(2026110617)
O=Path(__file__).resolve().parents[1]/'reproduced'/'verification'
O.mkdir(parents=True,exist_ok=True)

def h_basis(m):
    b=[]
    for i in range(m):
        a=np.zeros((m,m),complex); a[i,i]=1;b.append(a)
    for i in range(m):
        for j in range(i+1,m):
            a=np.zeros((m,m),complex);a[i,j]=a[j,i]=1;b.append(a)
            a=np.zeros((m,m),complex);a[i,j]=1j;a[j,i]=-1j;b.append(a)
    assert len(b)==m*m
    return b

def exact_information(a,D,N,p,n):
    W=np.linalg.inv(N);s=np.vdot(a,W@a).real
    w=n/(1+s*p)
    p_bar=np.dot(w,p)/sum(w)
    V=np.dot(w,(p-p_bar)**2)
    B=W-np.outer(W@a,(W@a).conj())/s
    G=2*s*np.real(D.conj().T@B@D)
    return V*G,V,G

def fish_projection(a,D,N,p,n):
    M=len(a);T=len(p)
    b=h_basis(M)
    invs=[np.linalg.inv(N+pt*np.outer(a,a.conj())) for pt in p]
    aa=np.outer(a,a.conj())
    d=[np.outer(D[:,i],a.conj())+np.outer(a,D[:,i].conj()) for i in range(D.shape[1])]
    # Direct Fisher inner product in the full manifold
    ders=[]
    for dd in d:ders.append([pt*dd for pt in p])
    for bb in b:ders.append([bb]*T)
    for j in range(T):ders.append([aa if i==j else np.zeros_like(aa) for i in range(T)])
    q=len(d)
    F=np.empty((len(ders),len(ders)))
    for i in range(len(ders)):
        for j in range(i+1):
            v=sum(nt*np.trace(W@hi@W@hj).real for nt,W,hi,hj in zip(n,invs,ders[i],ders[j]))
            F[i,j]=F[j,i]=v
    A=F[:q,:q];B=F[:q,q:];C=F[q:,q:]
    I=A-B@np.linalg.pinv(C,hermitian=True,rcond=1e-11)@B.T
    return (I+I.T)/2, np.linalg.matrix_rank(C,tol=1e-8),len(C)

def main():
    rows=[]
    for k in range(60):
        m=int(rng.integers(2,6)); T=int(rng.integers(2,6)); q=2
        z=(rng.normal(size=(m,m))+1j*rng.normal(size=(m,m)))/np.sqrt(2)
        N=z@z.conj().T + np.eye(m)*(.4+float(rng.random()))
        a=(rng.normal(size=m)+1j*rng.normal(size=m))/np.sqrt(2)
        D=(rng.normal(size=(m,q))+1j*rng.normal(size=(m,q)))/np.sqrt(2)
        p=rng.uniform(.3,2.4,T)
        if k%15==0:p[:]=1.2
        n=rng.integers(12,201,T).astype(float)
        exact,V,G=exact_information(a,D,N,p,n)
        projected,rank,dim=fish_projection(a,D,N,p,n)
        delta=np.max(np.abs(exact-projected))
        rel=delta/max(1,np.linalg.norm(exact))
        c=float(rng.uniform(-.15,.15));shift=N+c*np.outer(a,a.conj());
        if np.linalg.eigvalsh(shift).min()<=0: c=0.;shift=N
        if (p-c).min()<=0: c=0.;shift=N
        shifted,V2,G2=exact_information(a,D,shift,p-c,n)
        gauge=np.max(np.abs(exact-shifted))
        rows.append(dict(seed=2026110617,case=k,sensors=m,states=T,info_delta=delta,rel_delta=rel,gauge_delta=gauge,nuisance_rank=rank,nuisance_dim=dim,weighted_power_variation=V))
    worst=max(rows,key=lambda x:x['rel_delta'])
    print('CASES',len(rows),'MAX_ABS',max(x['info_delta'] for x in rows),'MAX_REL',worst['rel_delta'],'MAX_GAUGE',max(x['gauge_delta'] for x in rows))
    print('RANK_DEFICIENCY_ALL',all(x['nuisance_dim']-x['nuisance_rank']>=1 for x in rows))
    assert max(x['rel_delta'] for x in rows)<1e-8
    assert max(x['gauge_delta'] for x in rows)<1e-8
    with (O/'INDEPENDENT_EFIM_CHECKS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()

