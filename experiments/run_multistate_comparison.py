"""
File: run_multistate_comparison.py
Author: Xin Lu
Date: 2026-11-18
Description: Paired two-state and three-state snapshot allocation experiment.
"""
from __future__ import annotations
import argparse,csv,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nearfield_profile_ml import (NearfieldProfileProblem, steering_nearfield,
       efficient_information, full_information_schur)
from run_power_contrast import POS,FREQ_HZ,define_noises
TH=21.7; R=0.93; SEED=2026111010
POWERS={'three_state':(0.85,1.60,2.35),'two_extremes':(0.85,2.35)}
COUNTS={'three_state':(1024,1024,1024),'two_extremes':(1536,1536)}

def crb_table():
  out=[]; a,_=steering_nearfield(np.deg2rad(TH),R,POS,FREQ_HZ)
  for noise,N in define_noises().items():
    W=np.linalg.inv(N);s=float(np.vdot(a,W@a).real)
    for design,p in POWERS.items():
      n=np.array(COUNTS[design]);p=np.array(p);ome=n/(1+s*p)
      pbar=np.average(p,weights=ome);V=float(np.sum(ome*(p-pbar)**2))
      fi=efficient_information(np.deg2rad(TH),R,POS,FREQ_HZ,N,p,n)
      crb=np.linalg.inv(fi)
      P=p+1/s; weight=n/sum(n)
      A=np.dot(weight,P);H=1/np.dot(weight,1/P)
      gamma=sum(n)*(A-H)
      assert abs(V*s-gamma)<1e-9
      out.append(dict(noise=noise,design=design,T=len(p),n_total=int(sum(n)),powers=json.dumps(p.tolist()),counts=json.dumps(n.tolist()),
      V=V,gamma=gamma,arithmetic_mean_P=A,harmonic_mean_P=H,
      upper_bound_gamma=sum(n)*(np.sqrt(P.max())-np.sqrt(P.min()))**2,
      angle_crb_deg2=float(crb[0,0])*(180/np.pi)**2,range_crb_m2=float(crb[1,1])))
  return out

def one(job):
  noise_id,rep,maxiter=job;N=define_noises()[noise_id]
  seed=SEED+(0 if noise_id=='A' else 1000000)+rep
  rng=np.random.default_rng(seed)
  a,_=steering_nearfield(np.deg2rad(TH),R,POS,FREQ_HZ);aa=np.outer(a,a.conj())
  # Draw 1536 complex-Gaussian snapshots at each extreme, plus a separate mid block.
  zs={k:(rng.standard_normal((5,1536))+1j*rng.standard_normal((5,1536)))/np.sqrt(2) for k in ('low','high')}
  zs['middle']=(rng.standard_normal((5,1024))+1j*rng.standard_normal((5,1024)))/np.sqrt(2)
  state_samples={
    'low':np.linalg.cholesky(N+.85*aa)@zs['low'],
    'middle':np.linalg.cholesky(N+1.60*aa)@zs['middle'],
    'high':np.linalg.cholesky(N+2.35*aa)@zs['high']}
  out=[]
  for design,keys,counts in [('three_state',('low','middle','high'),(1024,1024,1024)),
                             ('two_extremes',('low','high'),(1536,1536))]:
    st=np.array([state_samples[k][:,:n]@state_samples[k][:,:n].conj().T/n for k,n in zip(keys,counts)])
    try:
      fit=NearfieldProfileProblem(st,np.asarray(counts),POS,FREQ_HZ).estimate_multistart(maxiter=maxiter)
      out.append(dict(noise=noise_id,replicate=rep,seed=seed,design=design,T=len(counts),n_total=3072,
         theta_true_deg=TH,range_true_m=R,theta_hat_deg=float(fit['theta_deg']),range_hat_m=float(fit['range_m']),
         profile_nll=float(fit['profile_nll']),converged_starts=int(fit['converged_starts']),
         starts_tried=int(fit['starts_tried']),optimizer_success=int(fit['success']),exception=''))
    except Exception as ex:
      out.append(dict(noise=noise_id,replicate=rep,seed=seed,design=design,T=len(counts),n_total=3072,
         theta_true_deg=TH,range_true_m=R,theta_hat_deg=np.nan,range_hat_m=np.nan,
         profile_nll=np.inf,converged_starts=0,starts_tried=0,optimizer_success=0,exception=repr(ex)))
  return out

def write_csv(path,rows):
  if not rows:return
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summary(rows,crb,bootstrap=4000):
  refs={(r['noise'],r['design']):r for r in crb};out=[]
  for noise in ('A','B'):
    for design in ('three_state','two_extremes'):
      vals=sorted((r for r in rows if r['noise']==noise and r['design']==design),key=lambda x:x['replicate'])
      if not vals:continue
      ref=refs[noise,design]
      e_th=np.array([r['theta_hat_deg']-TH for r in vals]);e_r=np.array([r['range_hat_m']-R for r in vals]);keep=np.isfinite(e_th)&np.isfinite(e_r)
      n_valid=keep.sum();t2=e_th[keep]**2/ref['angle_crb_deg2'];r2=e_r[keep]**2/ref['range_crb_m2']
      res=dict(noise=noise,design=design,trials=len(vals),valid=int(n_valid),n_total=3072,
      angle_mse_crb=float(t2.mean()) if n_valid else np.nan,range_mse_crb=float(r2.mean()) if n_valid else np.nan,
      angle_rmse_deg=float(np.sqrt(np.mean(e_th[keep]**2))) if n_valid else np.nan,
      range_rmse_m=float(np.sqrt(np.mean(e_r[keep]**2))) if n_valid else np.nan,
      theta_gt10deg=int(np.sum(np.abs(e_th[keep])>10)),range_boundary_hits=int(np.sum((np.array([r['range_hat_m'] for r in vals])<.4001)|(np.array([r['range_hat_m'] for r in vals])>2.9999))),
      optimizer_success_fraction=float(np.mean([r['optimizer_success'] for r in vals])))
      if n_valid>=10:
        rng=np.random.default_rng(20261110 + (0 if noise=='A' else 101)+(0 if design=='three_state' else 37))
        idx=rng.integers(0,n_valid,(bootstrap,n_valid))
        for key,a in [('angle',t2),('range',r2)]:
          ci=np.quantile(a[idx].mean(axis=1),[.025,.975]);res[key+'_boot95_low']=float(ci[0]);res[key+'_boot95_high']=float(ci[1]);
      out.append(res)
  return out

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--replicates',type=int,default=1000);parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--maxiter',type=int,default=120);parser.add_argument('--bootstrap',type=int,default=4000);a=parser.parse_args()
  dest=ROOT/'reproduced'/'multistate';dest.mkdir(parents=True,exist_ok=True)
  ref=crb_table();write_csv(dest/'theoretical_crb.csv',ref)
  for noise,N in define_noises().items():
    efim=efficient_information(np.deg2rad(TH),R,POS,FREQ_HZ,N,POWERS['three_state'],COUNTS['three_state'])
    fi=full_information_schur(np.deg2rad(TH),R,POS,FREQ_HZ,N,POWERS['three_state'],COUNTS['three_state'])
    err=np.linalg.norm(fi-efim)/max(1.,np.linalg.norm(efim));assert err<1e-8,err
    print('[FISHER]',noise,'three_state relative diff',err,flush=True)
  jobs=[(noise,i,a.maxiter) for noise in ('A','B') for i in range(a.replicates)]
  rows=[];tic=time.time()
  with ProcessPoolExecutor(max_workers=a.jobs) as pool:
    for i,result in enumerate(pool.map(one,jobs,chunksize=3),1):
      rows.extend(result)
      if i%100==0 or i==len(jobs):
        write_csv(dest/'multistate_trials.csv',rows)
        print(f'[PROGRESS] paired replicates {i}/{len(jobs)}, fits {len(rows)}; {time.time()-tic:.1f}s',flush=True)
  rows.sort(key=lambda x:(x['noise'],x['design'],x['replicate']))
  write_csv(dest/'multistate_trials.csv',rows);s=summary(rows,ref,a.bootstrap);write_csv(dest/'multistate_summary.csv',s)
  cfg=dict(seed_base=SEED,theta_true_deg=TH,range_true_m=R,frequency_hz=FREQ_HZ,sound_speed_m_s=343,
   positions_m=POS.tolist(),power_designs=POWERS,counts=COUNTS,replicates_per_noise=a.replicates,
   optimizer='four-start L-BFGS-B local profile likelihood',optimizer_maxiter=a.maxiter,
   noise='Noise A and Noise B, shared Hermitian positive-definite colored covariance',
   pairing='extreme-state draws shared across two- and three-state designs; middle state independently drawn',
   bootstrap_replicates=a.bootstrap,bootstrap_method='percentile resampling by paired Monte Carlo replicate, applied to each metric independently')
  (dest/'experiment_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
  for x in s:print('[SUMMARY]',x,flush=True)
  print('[DONE]',len(rows),'fits',flush=True)
if __name__=='__main__':main()
