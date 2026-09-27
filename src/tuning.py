"""Data-adaptive tuning rules for the Huber threshold tau and the weight cap c.

Every rule is a small pure function so that alternatives can be swapped in
estimator configurations (see src/registry.py).

Clip rules have signature  rule(w, psi, mpi) -> c   (c = np.inf: no clipping)
    w   : raw importance weights pi(A_i|X_i) / pihat_b(A_i|X_i)
    psi : (possibly Huberised) residuals psi_tau(Y_i - mhat(X_i, A_i))
    mpi : direct-method term  sum_a pi(a|X_i) mhat(X_i, a)
Tau rules have signature   rule(r, w) -> tau   (tau = np.inf: no Huberisation)
"""
from __future__ import annotations

from functools import partial
import numpy as np

MAD_CONST = 1.4826


def robust_scale(r: np.ndarray) -> float:
    """Normal-consistent MAD about the median."""
    s = MAD_CONST * np.median(np.abs(r - np.median(r)))
    if s <= 1e-12:  # degenerate -> fall back to IQR / sd
        s = np.subtract(*np.percentile(r, [75, 25])) / 1.349 or np.std(r) + 1e-12
    return float(s)


def tau_none(r, w=None):
    return np.inf


def tau_mad(r, w=None, kappa: float = 1.345):
    """tau = kappa * MAD(residuals)."""
    return kappa * robust_scale(r)


def clip_none(w, psi=None, mpi=None):
    return np.inf


def clip_quantile(w, psi=None, mpi=None, q: float = 0.99):
    """Fixed-quantile clipping of the empirical weight distribution."""
    return float(np.quantile(w, q))


def clip_ionides(w, psi=None, mpi=None):
    """Truncated IS (Ionides 2008): c = mean(w) * sqrt(n)."""
    return float(np.mean(w) * np.sqrt(len(w)))


# named rules used by the final estimators (src/registry.py)
TAU_RULES = {
    "none": tau_none,
    "mad1.345": partial(tau_mad, kappa=1.345),       # RCF-DR / RCF-DR-os fixed threshold
}
CLIP_RULES = {
    "none": clip_none,
    "q99": partial(clip_quantile, q=0.99),            # RCF-DR / RCF-DR-os cap
    "ionides": clip_ionides,                          # DR-clip cap  c = mean(w) sqrt(n)
}
