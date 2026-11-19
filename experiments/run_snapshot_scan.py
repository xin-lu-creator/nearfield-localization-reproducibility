"""
File: run_snapshot_scan.py
Author: Xin Lu
Date: 2026-11-18
Description: Paired snapshot-count simulation for the profile-likelihood estimator.
"""
import sys,csv,json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nearfield_profile_ml import NearfieldProfileProblem,steering_nearfield,efficient_information
from run_power_contrast import POS,FREQ_HZ,P_MEAN,define_noises
THETA=21.7;RANGE=.93;NS=(192,768,3072);DELTAS=(.75,1.5);SEED=2026110609

def one(job):
    name,rep,maxiter=job;N=define_noises()[name];seed=SEED+(0 if name=='A' else 1000000)+rep
    rng=np.random.default_rng(seed)
    z=[(rng.standard_normal((5,NS[-1]))+1j*rng.standard_normal((5,NS[-1])))/np.sqrt(2) for _ in range(2)]
    a,_=steering_nearfield(np.deg2rad(THETA),RANGE,POS,FREQ_HZ);aa=np.outer(a,a.conj());out=[]
    for delta in DELTAS:
        samples=[np.linalg.cholesky(N+p*aa)@zz for p,zz in zip((P_MEAN-delta,P_MEAN+delta),z)]
        for n in NS:
            S=np.array([x[:,:n]@x[:,:n].conj().T/n for x in samples]);err=''
            try:
                fit=NearfieldProfileProblem(S,np.array([n,n]),POS,FREQ_HZ).estimate_multistart(maxiter=maxiter)
                th=float(fit['theta_deg']);rr=float(fit['range_m'])
                vals=(th,rr,float(fit['profile_nll']),int(fit['success']),int(fit['starts_tried']),int(fit['converged_starts']))
            except Exception as ex:
                vals=(np.nan,np.nan,np.inf,0,0,0);err=repr(ex)
            th,rr,nll,ok,starts,conv=vals
            out.append(dict(noise=name,replicate=rep,seed=seed,delta=delta,n_per_state=n,
                theta_true_deg=THETA,range_true_m=RANGE,theta_hat_deg=th,range_hat_m=rr,
                nll=nll,optimizer_success=ok,n_starts=starts,converged_starts=conv,
                theta_boundary=int(abs(th)>=69.99),range_boundary=int(rr<=.4001 or rr>=2.9999),error=err))
    return out

def save(path,data):
    if not data:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

def summary(rows):
    out=[]
    for name in ('A','B'):
      for delta in DELTAS:
       for n in NS:
        v=[r for r in rows if r['noise']==name and r['delta']==delta and r['n_per_state']==n]
        if not v:continue
        th=np.asarray([x['theta_hat_deg'] for x in v]);rr=np.asarray([x['range_hat_m'] for x in v]);keep=np.isfinite(th)&np.isfinite(rr)
        tt=th[keep]-THETA;r=rr[keep]-RANGE
        efim=efficient_information(np.deg2rad(THETA),RANGE,POS,FREQ_HZ,define_noises()[name],(P_MEAN-delta,P_MEAN+delta),(n,n))
        crb=np.linalg.inv(efim)
        crt=float(crb[0,0]*(180/np.pi)**2);crr=float(crb[1,1])
        out.append(dict(noise=name,delta=delta,n_per_state=n,trials=len(v),valid=int(keep.sum()),
            angle_mse=float(np.mean(tt**2)),range_mse=float(np.mean(r**2)),
            angle_crb=crt,range_crb=crr,angle_mse_over_crb=float(np.mean(tt**2)/crt),
            range_mse_over_crb=float(np.mean(r**2)/crr),angle_bias=float(np.mean(tt)),range_bias=float(np.mean(r)),
            angle_outliers_gt10=int(np.sum(abs(tt)>10)),angle_boundary_hits=sum(z['theta_boundary'] for z in v),
            range_boundary_hits=sum(z['range_boundary'] for z in v)))
    return out


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--replicates',type=int,default=1000);arg.add_argument('--jobs',type=int,default=4);arg.add_argument('--maxiter',type=int,default=120);x=arg.parse_args()
    folder=ROOT/'reproduced'/'snapshot_scan';folder.mkdir(parents=True,exist_ok=True)
    jobs=[(name,i,x.maxiter) for name in ('A','B') for i in range(x.replicates)]
    all_rows=[]
    with ProcessPoolExecutor(max_workers=x.jobs) as pool:
      for i,r in enumerate(pool.map(one,jobs,chunksize=1),1):
        all_rows.extend(r)
        if i%24==0 or i==len(jobs):
            save(folder/'snapshot_scan_trials.csv',all_rows)
            print(f'[PROGRESS] {i}/{len(jobs)} groups; {len(all_rows)} fits',flush=True)
    all_rows.sort(key=lambda z:(z['noise'],z['delta'],z['n_per_state'],z['replicate']))
    save(folder/'snapshot_scan_trials.csv',all_rows);s=summary(all_rows);save(folder/'snapshot_scan_summary.csv',s)
    (folder/'snapshot_scan_config.json').write_text(json.dumps(dict(seed=SEED,theta_true_deg=THETA,range_true_m=RANGE,n_values=NS,contrasts=DELTAS,
       replicates=x.replicates,maxiter=x.maxiter,noise=['A','B'],angle_search_deg=[-70,70],range_search_m=[.4,3.],
       angle_pilot_grid_deg='-60:4:60; truth off grid',range_pilot_grid_m=[.55,.75,1.,1.5,2.2],
       pairing='same random draws across n and delta per noise/replicate',nuisance='unknown shared full Hermitian positive-definite covariance'),indent=2),encoding='utf-8')
    assert len(all_rows)==2*2*3*x.replicates
    for a in s:print(a['noise'],a['delta'],a['n_per_state'],'MSE/CRB',round(a['angle_mse_over_crb'],3),round(a['range_mse_over_crb'],3),'range hits',a['range_boundary_hits'])
    print('[PASS] all',len(all_rows),'fits completed')
if __name__=='__main__':main()
