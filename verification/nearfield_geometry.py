"""
File: nearfield_geometry.py
Author: Xin Lu
Date: 2026-11-18
Description: Verify the near-field inverse-fourth-power range-information coefficient.
"""
from __future__ import annotations
import sys
from pathlib import Path
import csv
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nearfield_profile_ml import steering_nearfield, geom_information

FREQ = 1250.
WAVE_SPEED = 343.
THETA = np.deg2rad(20.)
POS = np.array([[x,0,0] for x in [-.24,-.12,0.,.12,.24]],float)

def noise(name):
    m=len(POS);ij=np.abs(np.arange(m)[:,None]-np.arange(m)[None,:])
    if name=='A':
        tau,c0,c1,amp,ang,ran=.48,.44,.37,1.65,-45.,1.5
    else:
        tau,c0,c1,amp,ang,ran=.86,.30,.65,1.60,-34.,1.42
    bg,_=steering_nearfield(np.deg2rad(ang),ran,POS,FREQ)
    return c0*np.eye(m)+c1*tau**ij+amp*np.outer(bg,bg.conj()),bg

def run():
    rows=[]
    for name in ('A','B'):
        N,bg=noise(name)
        assert np.linalg.eigvalsh(N).min()>0
        W=np.linalg.inv(N)
        x=POS[:,0];k=2*np.pi*FREQ/WAVE_SPEED
        ainf=np.exp(1j*k*x*np.sin(THETA))/np.sqrt(len(x))
        v=(1j*k*x*np.cos(THETA))*ainf
        w=(-x*np.sin(THETA)+.5j*k*x*x*np.cos(THETA)**2)*ainf
        sinf=np.vdot(ainf,W@ainf).real
        Binf=W-np.outer(W@ainf,ainf.conj()@W)/sinf
        uv=np.vdot(v,Binf@v).real
        asym=2*sinf*(np.vdot(w,Binf@w).real-np.vdot(v,Binf@w).real**2/uv)
        assert asym>0
        for r in (.95,1.5,3.,6.,10.,20.):
            a,D=steering_nearfield(THETA,r,POS,FREQ)
            h=1e-3*r
            a2p=steering_nearfield(THETA,r+2*h,POS,FREQ)[0]
            ap=steering_nearfield(THETA,r+h,POS,FREQ)[0]
            am=steering_nearfield(THETA,r-h,POS,FREQ)[0]
            a2m=steering_nearfield(THETA,r-2*h,POS,FREQ)[0]
            dnum=(-a2p+8*ap-8*am+a2m)/(12*h)
            derivative_rel=np.linalg.norm(dnum-D[:,1])/max(np.linalg.norm(D[:,1]),1e-14)
            assert derivative_rel<2e-7, (name,r,derivative_rel)
            G=geom_information(THETA,r,POS,FREQ,N)
            conditional=G[1,1]-G[0,1]**2/G[0,0]
            W=np.linalg.inv(N);s=np.vdot(a,W@a).real
            B=W-np.outer(W@a, a.conj()@W)/s
            ba=np.linalg.norm(B@a);bb=np.linalg.norm(B@bg)
            assert conditional>0 and ba<1e-12 and bb>1e-3
            rows.append(dict(noise=name,range_m=r,geometry_Gr_given_theta=conditional,
                             geometry_Gr_given_theta_times_r4=conditional*r**4,
                             raw_geometry_Grr=G[1,1],asymptotic_C=asym,derivative_rel_error=derivative_rel,
                             norm_Ba=ba,norm_Bbg=bb))
    print('Noise A/B, r=0.95 m',[(x['noise'],x['geometry_Gr_given_theta']) for x in rows if x['range_m']==.95])
    for name in ('A','B'):
        a=[x for x in rows if x['noise']==name and x['range_m']==10.][0]
        b=[x for x in rows if x['noise']==name and x['range_m']==20.][0]
        assert abs(a['geometry_Gr_given_theta_times_r4']/b['geometry_Gr_given_theta_times_r4']-1)<.01
        assert abs(b['geometry_Gr_given_theta_times_r4']/b['asymptotic_C']-1)<.005
        print('Asymptotic C',name,b['asymptotic_C'])
    out=ROOT/'reproduced'/'nearfield_geometry.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as file:
        w=csv.DictWriter(file,list(rows[0]));w.writeheader();w.writerows(rows)
    print('[PASS] saved',out,'with',len(rows),'physical geometry points')
if __name__=='__main__':run()
