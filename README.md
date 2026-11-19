# Reproducibility code for near-field CRB in unknown colored noise

**Author:** Xin Lu (Iwate University)  
**Source snapshot:** 2026-11-18  
**Software version:** 1.0.0-submission  
**Software license:** MIT

This repository provides code and frozen simulation data supporting the manuscript "Near-Field CRB for a Quasi-Stationary Source in Unknown Colored Noise" by Xin Lu. It reproduces the reported estimator comparisons and numerical checks. It does not contain the article manuscript.

## Requirements

Python 3.11+ and the packages listed in `requirements.txt`. For regenerating vector figures with LaTeX-matched fonts, a local LaTeX installation and `dvipng` are required.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/smoke_test.py
```

## Files

| Location | Purpose |
| --- | --- |
| `src/nearfield_profile_ml.py` | Near-field steering model, EFIM, and profile-likelihood estimator |
| `experiments/` | Source-power contrast, snapshot-count, two-/three-state, WSF, and diffuse-mismatch simulations; bootstrap summaries and plots |
| `verification/` | Independent Fisher information, near-field constant, design bound, two-state equivalence, and numerical diagnostics |
| `data/` | Experiment configurations and recorded trial/summary data used for the reported results |
| `figures/` | Three article figure panels, each as vector PDF and 900 dpi PNG |

## Reproduce the analysis

Regenerate the figures and statistical summaries from the supplied trial data:

```bash
python experiments/analyze_snapshot_scan.py
python experiments/analyze_followup.py
python experiments/plot_figures.py
```

Verify the theoretical and numerical results:

```bash
python verification/fisher_information.py
python verification/werner_jansson_equivalence.py
python verification/design_bound.py
python verification/nearfield_geometry.py
python verification/optimization_audit.py
python verification/verify_wsf_and_diffuse.py
```

Run smaller test simulations (full Monte Carlo runs require more computation):

```bash
python experiments/run_power_contrast.py --smoke --jobs 1
python experiments/run_snapshot_scan.py --replicates 2 --jobs 1
python experiments/werner_jansson_wsf.py --replicates 2 --workers 1
python experiments/power_coupled_diffuse_robustness.py --replicates 2 --workers 1
python experiments/run_multistate_comparison.py --replicates 2 --jobs 1
python experiments/run_multistate_high_snapshots.py --replicates 2 --jobs 1 --n-per-state 1024 --bootstrap 100
```

Generated files are written to `reproduced/`. The supplied `data/` CSV files are not modified by these commands.

## Data and interpretation

The simulations use five sensors at positions −0.24, −0.12, 0, 0.12, and 0.24 m, at 1250 Hz, with sound speed 343 m/s. Noise A and B are spatially correlated, Hermitian positive-definite covariances. The source-power study uses (20°, 0.95 m); the snapshot-count study and WSF benchmark use (21.7°, 0.93 m). Repeated trials are independently generated, while some conditions share draws for paired comparison. Numerical searches use truth-independent initializations but are **not certified global maximizers**. The diffuse-field experiment evaluates a misspecified data model and does **not** claim an exact misspecified-data CRB.

Random-number seed values in the source and configuration files are fixed reproducibility parameters, not release dates. They are intentionally unchanged.

## Citation and license

Use `CITATION.cff` when citing this software. The frozen `v1.0.0-submission` release is permanently archived under the [version-specific Zenodo DOI](https://doi.org/10.5281/zenodo.24364828). The [concept DOI](https://doi.org/10.5281/zenodo.24364827) represents all versions. The corresponding [GitHub Release](https://github.com/xin-lu-creator/nearfield-localization-reproducibility/releases/tag/v1.0.0-submission) contains the release source and the frozen code-and-data ZIP.

Software is licensed under the MIT License (`LICENSE`). Simulation data are supplied as reproducibility material; this license statement does not grant rights to the separately distributed scholarly article.
