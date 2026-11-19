"""
File: smoke_test.py
Author: Xin Lu
Date: 2026-11-18
Description: Quick test of the steering model, Jacobian, and Fisher information.
"""
from __future__ import annotations
from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nearfield_profile_ml import steering_nearfield, efficient_information, noise_covariance
P=np.array([[-.24,0,0],[-.12,0,0],[0,0,0],[.12,0,0],[.24,0,0]],float)
a,D=steering_nearfield(np.deg2rad(20.0),.95,P,1250.)
assert a.shape==(5,) and D.shape==(5,2)
assert np.allclose(np.vdot(a,a),1.0) and np.isfinite(D).all()
N=noise_covariance(P,freq_hz=1250.)
info=efficient_information(np.deg2rad(20),.95,P,1250.,N,(.85,2.35),(192,192))
assert info.shape==(2,2) and np.isfinite(info).all()
print('PASS: near-field model, EFIM, and estimator dependencies')
