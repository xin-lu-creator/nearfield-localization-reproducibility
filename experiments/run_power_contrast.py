"""
File: run_power_contrast.py
Author: Xin Lu
Date: 2026-11-18
Description: Power-contrast simulation for near-field bearing and range estimation.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
sys.path.insert(0, str(REPO / 'src'))
from nearfield_profile_ml import (
    NearfieldProfileProblem, steering_nearfield, noise_covariance,
    efficient_information, full_information_schur,
)

POS = np.array([[-0.24,0,0],[-0.12,0,0],[0,0,0],[0.12,0,0],[0.24,0,0]], float)
TH = np.deg2rad(20.0)
RANGE_M = 0.95
FREQ_HZ = 1250.0
P_MEAN = 1.6
N_PER_STATE = 192
DELTAS = (0.0, 0.35, 0.75, 1.15, 1.50)
BASE_SEED = 2026110606


def define_noises():
    """Two fixed positive-definite colored backgrounds; neither depends on delta."""
    N_a = noise_covariance(POS, freq_hz=FREQ_HZ)
    b, _ = steering_nearfield(np.deg2rad(-34), 1.42, POS, FREQ_HZ)
    d = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    N_b = 0.30 * np.eye(5) + 0.65 * 0.86**d + 1.60 * np.outer(b, b.conj())
    for name, N in [('A', N_a), ('B', N_b)]:
        assert np.linalg.eigvalsh(N).min() > 0, name
    return {'A': N_a, 'B': N_b}


def run_one(task):
    """Hold standard-normal draws fixed across all contrasts for one replication."""
    noise_id, replicate, maxiter = task
    noises = define_noises()
    N = noises[noise_id]
    a, _ = steering_nearfield(TH, RANGE_M, POS, FREQ_HZ)
    A = np.outer(a, a.conj())
    # independent across replicates and noise types, paired within each series
    noise_code = 0 if noise_id == 'A' else 1
    seed = BASE_SEED + noise_code * 1000000 + replicate
    rng = np.random.default_rng(seed)
    zs = [(rng.standard_normal((5, N_PER_STATE)) +
           1j * rng.standard_normal((5, N_PER_STATE))) / np.sqrt(2)
          for _ in range(2)]
    results = []
    for delta in DELTAS:
        powers = [P_MEAN - delta, P_MEAN + delta]
        cov = [N + p * A for p in powers]
        st = []
        for S, Z in zip(cov, zs):
            L = np.linalg.cholesky(S)
            X = L @ Z
            st.append(X @ X.conj().T / N_PER_STATE)
        prob = NearfieldProfileProblem(np.asarray(st),np.asarray([N_PER_STATE]*2),POS,FREQ_HZ)
        try:
            fit = prob.estimate_multistart(maxiter=maxiter)
            out = {
                'theta_hat_deg': fit['theta_deg'],
                'range_hat_m': fit['range_m'],
                'nll': fit['profile_nll'],
                'optimizer_success': int(fit['success']),
                'n_starts': fit['starts_tried'],
                'converged_starts': fit['converged_starts'],
                'iterations': fit['nit'],
            }
        except Exception as e:
            out = {
                'theta_hat_deg': float('nan'), 'range_hat_m': float('nan'),
                'nll': float('inf'), 'optimizer_success': 0,
                'n_starts': 0, 'converged_starts': 0,
                'iterations': 0, 'exception': repr(e),
            }
        results.append({
            'noise': noise_id, 'replicate': replicate, 'seed': seed,
            'delta': delta, 'p0': powers[0], 'p1': powers[1],
            'n_per_state': N_PER_STATE, 'theta_true_deg': np.rad2deg(TH),
            'range_true_m': RANGE_M, **out,
        })
    return results


def crb_table():
    """Oracle model bounds for comparison only; no truth enters optimization."""
    rows = []
    for name, N in define_noises().items():
        a, _ = steering_nearfield(TH, RANGE_M, POS, FREQ_HZ)
        s = np.vdot(a, np.linalg.solve(N, a)).real
        for delta in DELTAS:
            p = [P_MEAN - delta, P_MEAN + delta]
            counts = [N_PER_STATE, N_PER_STATE]
            w = np.asarray(counts, float) / (1 + s * np.asarray(p))
            V = float(w[0] * w[1] / sum(w) * (p[1] - p[0])**2)
            efim = efficient_information(TH, RANGE_M, POS, FREQ_HZ,N,p,counts)
            if delta == 0:
                std_theta, std_r = np.inf, np.inf
            else:
                # Interdependent (theta, r): invert the whole 2-by-2 EFIM.
                cov_par = np.linalg.inv(efim)
                std_theta = np.rad2deg(np.sqrt(cov_par[0,0]))
                std_r = np.sqrt(cov_par[1,1])
            rows.append({'noise':name,'delta':delta,'p0':p[0],'p1':p[1],
                         'mean_power':P_MEAN, 's':s,'V':V,
                         'efim_min_eigenvalue':float(np.linalg.eigvalsh(efim).min()) if delta>0 else 0.,
                         'crb_theta_std_deg':std_theta,
                         'crb_range_std_m':std_r,
                         'N_min_eigenvalue':np.linalg.eigvalsh(N).min(),
                         'N_max_eigenvalue':np.linalg.eigvalsh(N).max()})
    return pd.DataFrame(rows)


def aggregate(raw, bounds, bootstrap_seed=919, n_boot=1000):
    rng = np.random.default_rng(bootstrap_seed)
    rows=[]
    for (name, delta), grp in raw.groupby(['noise','delta'], sort=True):
        grp = grp.sort_values('replicate')
        theta_err = grp.theta_hat_deg.to_numpy(float) - 20.
        range_err = grp.range_hat_m.to_numpy(float) - RANGE_M
        valid = np.isfinite(theta_err) & np.isfinite(range_err)
        n_total = len(grp)
        n_valid = valid.sum()
        if n_valid:
            t = theta_err[valid];rr = range_err[valid]
            theta_rmse = np.sqrt(np.mean(t*t));range_rmse=np.sqrt(np.mean(rr*rr))
            ix = rng.integers(0,len(t),size=(n_boot,len(t)))
            bt = np.sqrt(np.mean(t[ix]**2,axis=1))
            br = np.sqrt(np.mean(rr[ix]**2,axis=1))
            ct = np.quantile(bt,[.025,.975]);cr=np.quantile(br,[.025,.975])
            t_med = np.median(np.abs(t));r_med=np.median(np.abs(rr))
            failures = sum(np.abs(t)>10.)
        else:
            theta_rmse=range_rmse=t_med=r_med=np.nan
            ct=cr=[np.nan,np.nan];failures=0
        ref=bounds[(bounds.noise==name)&(bounds.delta==delta)].iloc[0]
        rows.append({'noise':name,'delta':delta,'trials':n_total,'valid_trials':int(n_valid),
                     'angle_rmse_deg':theta_rmse,'angle_RMSE_boot95_low':ct[0],
                     'angle_RMSE_boot95_high':ct[1],
                     'range_rmse_m':range_rmse,'range_RMSE_boot95_low':cr[0],
                     'range_RMSE_boot95_high':cr[1],
                     'angle_abs_median_deg':t_med,'range_abs_median_m':r_med,
                     'angle_outliers_gt10deg':int(failures),
                     'angle_outlier_fraction':failures/max(n_total,1),
                     'at_least_three_starts_converged_fraction':float(np.mean(grp.converged_starts.to_numpy()>=3)),
                     'crb_theta_std_deg':ref.crb_theta_std_deg,
                     'crb_range_std_m':ref.crb_range_std_m,'V':ref.V})
    return pd.DataFrame(rows)




def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--replicates',type=int,default=120)
    ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--maxiter',type=int,default=120)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.smoke:args.replicates=2
    (REPO/'reproduced'/'power_contrast').mkdir(parents=True,exist_ok=True)
    jobs=[(name,i,args.maxiter) for name in ['A','B'] for i in range(args.replicates)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for j, res in enumerate(pool.map(run_one,jobs,chunksize=1),1):
            rows.extend(res)
            if j%20==0 or j==len(jobs):
                print(f'[PROGRESS] {j}/{len(jobs)} paired replicate blocks; {len(rows)} fits',flush=True)
    raw=pd.DataFrame(rows).sort_values(['noise','delta','replicate']).reset_index(drop=True)
    bounds=crb_table()
    summary=aggregate(raw,bounds)
    prefix='SMOKE_' if args.smoke else ''
    raw.to_csv(REPO/'reproduced'/'power_contrast'/f'{prefix}POWER_CONTRAST_TRIALS.csv',index=False)
    bounds.to_csv(REPO/'reproduced'/'power_contrast'/f'{prefix}POWER_CONTRAST_CRB.csv',index=False)
    summary.to_csv(REPO/'reproduced'/'power_contrast'/f'{prefix}POWER_CONTRAST_SUMMARY.csv',index=False)
    config={'release_date':'2026-11-18','simulation':'proper_complex_gaussian',
            'paired_common_random_numbers':True,'seed':BASE_SEED,
            'n_per_state':N_PER_STATE,'mean_power':P_MEAN,'deltas':DELTAS,
            'source_angle_deg':20,'source_range_m':RANGE_M,
            'freq_hz':FREQ_HZ,'sound_speed_m_per_s':343,
            'sensor_positions_m':POS.tolist(),'maxiter':args.maxiter,
            'replicates_per_noise_and_delta':args.replicates,
            'n_noise_structures':2,'initialization':'truth-blind fixed 4-start with contrast pilot',
            'note':'delta=0 not identifiable; infinite CRB. No real-data validation.'}
    (REPO/'reproduced'/'power_contrast'/f'{prefix}CONFIG.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4g}'))
    assert len(raw)==2*len(DELTAS)*args.replicates
    assert np.allclose(raw.p0+raw.p1,2*P_MEAN)
    print('[PASS] fixed-mean paired experiment complete',flush=True)

if __name__=='__main__':main()
