"""
File: analyze_snapshot_scan.py
Author: Xin Lu
Date: 2026-11-18
Description: MSE, CRB, and bootstrap statistics for the snapshot-count experiment.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from run_snapshot_scan import POS,FREQ_HZ,P_MEAN,THETA,RANGE,NS,DELTAS,SEED,define_noises
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nearfield_profile_ml import efficient_information
DATA=ROOT/'data'
OUT=ROOT/'reproduced'/'analysis'
REPS=1000
B=4000


def derive():
    trials=pd.read_csv(DATA/'snapshot_scan_trials.csv')
    assert len(trials)==REPS*2*len(DELTAS)*len(NS),(len(trials),'missing trials')
    assert not trials.duplicated(subset=['noise','replicate','delta','n_per_state']).any()
    assert trials.theta_hat_deg.notna().all() and trials.range_hat_m.notna().all()
    assert np.isfinite(trials.nll).all()
    for name in ('A','B'):
        r=trials[trials.noise==name]
        assert set(r.replicate)==set(range(REPS))
        assert r.groupby('replicate').size().eq(6).all()
        assert all((r.groupby('replicate').seed.nunique()==1))
        assert all(r.seed==SEED+(0 if name=='A' else 1000000)+r.replicate)
    rng=np.random.default_rng(2026110611)
    boots=rng.integers(0,REPS,size=(B,REPS),dtype=np.int32)
    rows=[]
    contrasts=[]
    for name in ('A','B'):
        dat=trials[trials.noise==name]
        for delta in DELTAS:
            E={}
            for n in NS:
                sub=dat[(dat.delta==delta)&(dat.n_per_state==n)].sort_values('replicate')
                assert len(sub)==REPS and np.array_equal(sub.replicate.to_numpy(),np.arange(REPS))
                errth=sub.theta_hat_deg.to_numpy(float)-THETA
                errr=sub.range_hat_m.to_numpy(float)-RANGE
                efim=efficient_information(np.deg2rad(THETA),RANGE,POS,FREQ_HZ,define_noises()[name],(P_MEAN-delta,P_MEAN+delta),(n,n))
                crb=np.linalg.inv(efim)
                crbt=float(crb[0,0]*(180/np.pi)**2)
                crbr=float(crb[1,1])
                thsq=errth**2/crbt
                rsq=errr**2/crbr
                E[n]={'th':thsq,'r':rsq}
                bt=thsq[boots].mean(axis=1)
                br=rsq[boots].mean(axis=1)
                rows.append(dict(noise=name,delta=delta,n_per_state=n,trials=REPS,valid=REPS,
                    angle_mse=float(np.mean(errth**2)),range_mse=float(np.mean(errr**2)),
                    angle_crb=crbt,range_crb=crbr,
                    angle_mse_over_crb=float(np.mean(thsq)),range_mse_over_crb=float(np.mean(rsq)),
                    angle_ratio_boot95_low=float(np.quantile(bt,.025)),angle_ratio_boot95_high=float(np.quantile(bt,.975)),
                    range_ratio_boot95_low=float(np.quantile(br,.025)),range_ratio_boot95_high=float(np.quantile(br,.975)),
                    angle_bias=float(np.mean(errth)),range_bias=float(np.mean(errr)),
                    angle_outliers_gt10=int(np.sum(abs(errth)>10)),
                    angle_boundary_hits=int(sub.theta_boundary.sum()),range_boundary_hits=int(sub.range_boundary.sum()),
                    optimization_success_fraction=float(sub.optimizer_success.mean()),
                    converged_starts_mean=float(sub.converged_starts.mean())))
            for measure in ('th','r'):
                diff=E[192][measure]-E[3072][measure]
                bd=diff[boots].mean(axis=1)
                contrasts.append(dict(noise=name,delta=delta,measure='angle' if measure=='th' else 'range',
                      ratio_improvement=float(diff.mean()),ci_low=float(np.quantile(bd,.025)),
                      ci_high=float(np.quantile(bd,.975)),replicates=REPS))
    out=pd.DataFrame(rows).sort_values(['noise','delta','n_per_state'])
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/'snapshot_scan_summary.csv',index=False,float_format='%.12g')
    pd.DataFrame(contrasts).to_csv(OUT/'snapshot_scan_paired_comparisons.csv',index=False,float_format='%.12g')

    print(out[['noise','delta','n_per_state','angle_mse_over_crb','range_mse_over_crb']].to_string(index=False))
    print('[PASS] paired bootstrap and summary regenerated in', OUT)
    return out


if __name__ == '__main__':
    derive()
