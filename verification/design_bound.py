"""
File: design_bound.py
Author: Xin Lu
Date: 2026-11-18
Description: Verify the two-extreme-state information-allocation bound.
"""
import numpy as np

def check(seed=20261110,configurations=1000):
    rng=np.random.default_rng(seed);worst=0.
    for _ in range(configurations):
        t=int(rng.integers(2,9));P=np.exp(rng.uniform(-2,2,t));n=rng.integers(1,500,t)
        num=sum(n);w=n/num
        A=np.dot(w,P);H=1/np.dot(w,1/P)
        Gamma=sum(n/P*(P-H)**2)
        limit=num*(np.sqrt(P.max())-np.sqrt(P.min()))**2
        err=abs(Gamma-num*(A-H))/max(1.,abs(Gamma))
        assert err<1e-12,err
        assert Gamma<=limit+1e-11*max(1.,limit)
        worst=max(worst,err)
    print(f'[PASS] {configurations} state allocations; max identity relative error {worst:.3e}')
if __name__=='__main__':check()
