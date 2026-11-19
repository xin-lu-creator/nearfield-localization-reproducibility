"""
File: plot_figures.py
Author: Xin Lu
Date: 2026-11-18
Description: Generate the three figures from the supplied simulation data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from figure_style import configure, new_axes, style_for, legend, save_panel, DPI
from analyze_followup import wsf_summary, plot_main_wsf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduced' / 'figures'


def contrast_panel(kind: str, frame: pd.DataFrame) -> None:
    """Draw one independently saved contrast panel with fixed symbol semantics."""
    if kind not in ('bearing', 'range'):
        raise ValueError(kind)
    fig, ax = new_axes()
    metric, crb, ylabel, yrange, ticks = (
        ('angle_rmse_deg', 'crb_theta_std_deg', 'Bearing RMSE (deg)',
         (.42, 85), [.5, 1, 2, 5, 10, 20, 50]) if kind == 'bearing' else
        ('range_rmse_m', 'crb_range_std_m', 'Range RMSE (m)',
         (.075, 2.1), [.1, .2, .5, 1, 2]))
    # Column-major order is deliberate: legend rows read A: ML, B: ML;
    # then A: CRB, B: CRB. Noise markers never change across figures.
    for noise, method, col in [('A', 'ML', metric), ('A', 'CRB', crb),
                               ('B', 'ML', metric), ('B', 'CRB', crb)]:
        df = frame[frame.noise == noise].sort_values('delta')
        finite = df[np.isfinite(df[col])]
        if method == 'ML':
            lower, upper = (('angle_RMSE_boot95_low', 'angle_RMSE_boot95_high')
                            if kind == 'bearing' else
                            ('range_RMSE_boot95_low', 'range_RMSE_boot95_high'))
            y = finite[col].to_numpy()
            down = np.maximum(0.0, y - finite[lower].to_numpy())
            up = np.maximum(0.0, finite[upper].to_numpy() - y)
            ax.errorbar(finite.delta, y, yerr=np.vstack([down, up]),
                        elinewidth=.64, capsize=1.75, capthick=.64,
                        **style_for(noise, method))
        else:
            ax.plot(finite.delta, finite[col], **style_for(noise, method))
    ax.set_yscale('log')
    ax.set_ylim(yrange)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(v) for v in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-.035, 1.56)
    ax.set_xticks([0, .35, .75, 1.15, 1.50])
    ax.set_xticklabels(['0', '0.35', '0.75', '1.15', '1.50'])
    ax.set_xlabel(r'Power contrast $\delta$')
    ax.set_ylabel(ylabel)
    legend(ax)
    name = f'fig1{"a" if kind == "bearing" else "b"}_{kind}'
    save_panel(fig, OUT / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='Render and validate all three plots')
    parser.parse_args()
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    contrast = pd.read_csv(ROOT / 'data' / 'power_contrast_summary.csv')
    contrast_panel('bearing', contrast)
    contrast_panel('range', contrast)
    # Draw the paired, 1000-replicate WJ one-source WSF / ML benchmark.
    from shutil import copy2
    wsf, _ = wsf_summary()
    pdf, png = plot_main_wsf(wsf)
    copy2(pdf, OUT / 'fig2_ml_vs_wsf.pdf')
    copy2(png, OUT / 'fig2_ml_vs_wsf.png')
    for stem in ['fig1a_bearing', 'fig1b_range', 'fig2_ml_vs_wsf']:
        print(f'Wrote {stem}.pdf and {stem}.png ({DPI} dpi)')


if __name__ == '__main__':
    main()
