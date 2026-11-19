"""
File: figure_style.py
Author: Xin Lu
Date: 2026-11-18
Description: Consistent monochrome figure styling and vector/raster output.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import fitz

PANEL_IN = (3.50, 1.72)
DPI = 900
METHOD_STYLES = {
    'ML': '-',
    'CRB': (0, (4.5, 2.4)),
    'Diff.': (0, (1.2, 2.4)),
    'WSF': (0, (4.5, 2.4)),
}
NOISE_MARKERS = {'A': 'o', 'B': 's'}


def configure() -> None:
    """Initialize consistent publication-style, Times-family plot typography."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times'],
        'text.usetex': True, 'text.latex.preamble': r'\usepackage{mathptmx}',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 7.1, 'axes.labelsize': 8.0,
        'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0,
        'legend.fontsize': 7.0, 'axes.linewidth': .55,
        'lines.linewidth': 1.10, 'lines.markersize': 3.75,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def new_axes():
    """Return a panel with identical plotting area and margins."""
    fig, ax = plt.subplots(figsize=PANEL_IN, dpi=100)
    fig.subplots_adjust(left=.178, right=.983, bottom=.230, top=.955)
    ax.tick_params(which='major', width=.55, length=2.6, pad=2.3)
    ax.tick_params(which='minor', width=.4, length=1.4)
    ax.grid(axis='y', which='major', lw=.37, color='.86', zorder=0)
    return fig, ax


def style_for(noise: str, method: str) -> dict:
    """Select encoding by TWO independent meanings: noise and estimator."""
    if noise not in NOISE_MARKERS or method not in METHOD_STYLES:
        raise ValueError(f'Unsupported figure mapping: {noise}, {method}')
    return dict(color='black', linestyle=METHOD_STYLES[method],
                linewidth=1.12 if method == 'ML' else 1.05,
                marker=NOISE_MARKERS[noise], markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=.7,
                markersize=3.80, zorder=3, label=f'{noise}: {method}')


def legend(ax):
    """Apply the same two-column legend layout to all panels."""
    return ax.legend(loc='upper right', ncol=2, frameon=False,
                     handlelength=1.43, handletextpad=.27,
                     columnspacing=.52, borderaxespad=.09,
                     labelspacing=.16, fontsize=7.0)


def save_panel(fig, stem: Path):
    """Save one independent vector PDF and 900-dpi PNG, validating output."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = stem.with_suffix('.pdf'), stem.with_suffix('.png')
    fig.savefig(pdf, format='pdf')
    fig.savefig(png, format='png', dpi=DPI)
    plt.close(fig)
    with Image.open(png) as im:
        assert im.size == (round(PANEL_IN[0] * DPI), round(PANEL_IN[1] * DPI))
        assert min(im.info.get('dpi', (0, 0))) > 600
    with fitz.open(pdf) as doc:
        p = doc[0]
        assert abs(p.rect.width - PANEL_IN[0] * 72) < .1
        assert abs(p.rect.height - PANEL_IN[1] * 72) < .1
        assert not any('Type3' in f[2].replace(' ', '') for f in p.get_fonts(full=True))
    return pdf, png
