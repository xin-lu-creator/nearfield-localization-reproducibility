"""
File: verify_release.py
Author: Xin Lu
Date: 2026-11-18
Description: Check the released data, figure files, and SHA-256 manifest.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
TRIALS = {
    'power_contrast_trials.csv': (1200, ('noise', 'replicate', 'delta')),
    'snapshot_scan_trials.csv': (12000, ('noise', 'replicate', 'delta', 'n_per_state')),
    'wsf_trials.csv': (6000, ('noise', 'replicate', 'n_per_state')),
    'mismatch_trials.csv': (1440, ('noise', 'replicate', 'eps', 'method')),
    'multistate_trials.csv': (4000, ('noise', 'replicate', 'design')),
    'optimization_audit_trials.csv': (1200, ('noise', 'replicate', 'delta')),
    'high_snapshot/high_snapshot_trials.csv': (4000, ('noise', 'replicate')),
}
for name, (expected, fields) in TRIALS.items():
    with (ROOT / 'data' / name).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected, (name, len(rows), expected)
    keys = [tuple(row[k] for k in fields) for row in rows]
    assert len(set(keys)) == len(keys), name
    print('PASS:', name, len(rows))
for name in ('fig1a_bearing', 'fig1b_range', 'fig2_ml_vs_wsf'):
    for ext in ('.pdf', '.png'):
        assert (ROOT / 'figures' / (name + ext)).is_file(), name + ext
checks = 0
for line in (ROOT / 'SHA256SUMS.txt').read_text('utf-8').splitlines():
    if not line.strip():
        continue
    digest, rel = line.split('  ', 1)
    path = ROOT / rel
    assert path.is_file(), rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, rel
    checks += 1
print('PASS:', checks, 'SHA-256 checks; all scientific inputs and figures present')
