"""
File: analyze_followup.py
Author: Xin Lu
Date: 2026-11-18
Description: Paired bootstrap analysis for WSF comparison and diffuse mismatch.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
ROOT=Path(__file__).resolve().parents[1]
UP=ROOT
sys.path.insert(0,str(UP/'experiments'))
from figure_style import configure,new_axes,save_panel,style_for,legend
DATA=UP/'data';OUT=ROOT/'reproduced/followup';OUT.mkdir(parents=True,exist_ok=True)
BOOT=4000

def bci(values,inds):return np.percentile(np.mean(values[inds],axis=1),[2.5,97.5])

def wsf_summary():
    w=pd.read_csv(DATA/'wsf_trials.csv')
    m=pd.read_csv(DATA/'snapshot_scan_trials.csv')
    assert len(w)==6000 and len(w[['noise','replicate','n_per_state']].drop_duplicates())==6000
    assert w.replicate.min()==0 and w.replicate.max()==999
    m=m[np.isclose(m.delta,.75)]
    merge=w.merge(m,on=['noise','replicate','seed','delta','n_per_state'],validate='one_to_one',suffixes=('_wsf','_ml'))
    assert len(merge)==6000
    frozen=pd.read_csv(DATA/'snapshot_scan_summary.csv')
    shared_boot=np.random.default_rng(2026110611).integers(0,1000,size=(BOOT,1000),dtype=np.int32)
    entries=[];paired=[]
    for noise in ('A','B'):
      for n in (192,768,3072):
        sub=merge.query('noise==@noise and n_per_state==@n').sort_values('replicate')
        assert len(sub)==1000 and np.array_equal(sub.replicate.to_numpy(),np.arange(1000))
        model=frozen.query('noise==@noise and delta==0.75 and n_per_state==@n').iloc[0]
        errs={}
        for method, suffix in [('Profile ML','ml'),('WJ-WSF','wsf')]:
            se=(sub[f'range_hat_m_{suffix}'].to_numpy()-.93)**2
            ae=(sub[f'theta_hat_deg_{suffix}'].to_numpy()-21.7)**2
            errs[method]=se
            bs=bci(se,shared_boot)/model.range_crb
            br=float(np.mean(se)/model.range_crb)
            ai=bci(ae,shared_boot)/model.angle_crb
            ar=float(np.mean(ae)/model.angle_crb)
            entries.append(dict(noise=noise,delta=.75,n_per_state=n,method=method,replicates=1000,
             range_mse_over_crb=br,range_ci95_low=bs[0],range_ci95_high=bs[1],
             bearing_mse_over_crb=ar,bearing_ci95_low=ai[0],bearing_ci95_high=ai[1],
             range_rmse_m=np.sqrt(np.mean(se)),bearing_rmse_deg=np.sqrt(np.mean(ae)),
             range_bias_m=np.mean(sub[f'range_hat_m_{suffix}'].to_numpy()-.93),
             optimizer_success_rate=np.mean(sub['success'] if method=='WJ-WSF' else sub['optimizer_success'])))
            if method=='Profile ML':
                assert abs(br-model.range_mse_over_crb)<1e-9
                assert np.max(abs(bs-[model.range_ratio_boot95_low,model.range_ratio_boot95_high]))<1e-8
        diff=(errs['WJ-WSF']-errs['Profile ML'])/model.range_crb
        p=bci(diff,shared_boot)
        paired.append(dict(noise=noise,n_per_state=n,paired_replicates=1000,
           wsf_minus_ml_mse_over_crb=float(np.mean(diff)),ci95_low=p[0],ci95_high=p[1]))
    s=pd.DataFrame(entries);p=pd.DataFrame(paired)
    s.to_csv(OUT/'wsf_vs_ml_summary.csv',index=False)
    p.to_csv(OUT/'wsf_vs_ml_paired_difference.csv',index=False)
    return s,p

def plot_main_wsf(s, *, out_stem='fig2_ml_vs_wsf'):
    """Render the paired WSF / ML range comparison panel."""
    configure()
    fig,ax=new_axes()
    # Matplotlib fills a two-column legend down the first column, then the next.
    for noise,method in [('A','Profile ML'),('B','Profile ML'),
                         ('A','WJ-WSF'),('B','WJ-WSF')]:
        dat=s[(s.noise==noise)&(s.method==method)].sort_values('n_per_state')
        y=dat.range_mse_over_crb.to_numpy()
        display_method='ML' if method=='Profile ML' else 'WSF'
        st=style_for(noise,display_method)
        ax.plot(dat.n_per_state.to_numpy(),y,**st)
        ax.errorbar(dat.n_per_state.to_numpy(),y,
            yerr=np.vstack((y-dat.range_ci95_low.to_numpy(),dat.range_ci95_high.to_numpy()-y)),
            fmt='none',ecolor='.39',elinewidth=.58,capsize=1.4,capthick=.60,zorder=2)
    ax.axhline(1,color='.4',linestyle=(0,(4.5,2.4)),lw=.80,zorder=1)
    ax.set_xscale('log',base=4);ax.set_xlim(156,3850)
    ax.set_xticks([192,768,3072],labels=['192','768','3072'])
    ax.set_ylim(.35,4.7);ax.set_yticks([1,2,3,4]);ax.set_ylabel('Range MSE / CRB');ax.set_xlabel('Snapshots per state')
    leg=legend(ax)
    assert [t.get_text() for t in leg.get_texts()]==['A: ML','B: ML','A: WSF','B: WSF']
    return save_panel(fig,OUT/out_stem)

def mismatch_summary():
    mis=pd.read_csv(DATA/'mismatch_trials.csv')
    w=pd.read_csv(DATA/'wsf_trials.csv')
    o=pd.read_csv(DATA/'snapshot_scan_trials.csv')
    assert len(mis)==1440
    baseline=[]
    for noise in ('A','B'):
      for method,src in [('Profile ML',o),('WJ-WSF',w)]:
        sub=src.query('noise==@noise and n_per_state==768 and replicate<120')
        if method=='Profile ML':sub=sub.query('delta==0.75')
        assert len(sub)==120
        for _,r in sub.iterrows():
            baseline.append(dict(noise=noise,replicate=int(r.replicate),seed=int(r.seed),
             eps=0.,method=method,range_hat_m=r.range_hat_m,
             theta_hat_deg=r.theta_hat_deg,range_true_m=.93,theta_true_deg=21.7))
    assert all((mis.groupby(['noise','eps','method']).size()==120))
    all_rows=pd.concat([pd.DataFrame(baseline),mis],ignore_index=True)
    boot=np.random.default_rng(2026111717).integers(0,120,size=(BOOT,120),dtype=np.int32)
    rows=[];compare=[]
    for noise in ('A','B'):
      for method in ('Profile ML','WJ-WSF'):
        first=all_rows.query('noise==@noise and eps==0 and method==@method').sort_values('replicate')
        first_sq=(first.range_hat_m.to_numpy()-.93)**2
        for eps in (0.,.05,.15,.30):
            sub=all_rows.query('noise==@noise and eps==@eps and method==@method').sort_values('replicate')
            assert len(sub)==120 and np.array_equal(sub.replicate.to_numpy(),np.arange(120))
            err=sub.range_hat_m.to_numpy()-.93
            sq=err**2
            ae=sub.theta_hat_deg.to_numpy()-21.7
            rmse_boot=np.sqrt(np.mean(sq[boot],axis=1))
            rmse_ci=np.percentile(rmse_boot,[2.5,97.5])
            bias_ci=bci(err,boot)
            diff_ci=bci(sq-first_sq,boot)
            rows.append(dict(noise=noise,method=method,eps=eps,replicates=120,n_per_state=768,
              range_rmse_m=float(np.sqrt(np.mean(sq))),rmse_ci95_low=rmse_ci[0],rmse_ci95_high=rmse_ci[1],
              bearing_rmse_deg=float(np.sqrt(np.mean(ae**2))),range_bias_m=np.mean(err),
              range_bias_ci95_low=bias_ci[0],range_bias_ci95_high=bias_ci[1],
              delta_mse_from_eps0=np.mean(sq-first_sq),delta_mse_ci95_low=diff_ci[0],delta_mse_ci95_high=diff_ci[1]))
    df=pd.DataFrame(rows);df.to_csv(OUT/'mismatch_summary.csv',index=False)
    return df


if __name__=='__main__':
    a,b=wsf_summary();plot_main_wsf(a)
    c=mismatch_summary()
    print('WJ-WSF vs ML (1000 paired each):');print(a.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('paired bootstrap difference:');print(b.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('model mismatch:');print(c.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
