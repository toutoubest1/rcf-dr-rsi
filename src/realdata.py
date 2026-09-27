"""Shared estimation step of the two real-outcome applications (ASlib and KuaiRec).

The reward model is linear in the context covariates Z, fitted separately for each action with the Huber IRLS
routine of the simulation study (src.nuisance._huber_irls, tuning constant 1.345) or, as a sensitivity analysis,
by least squares (src.nuisance._ols).  It is cross-fitted with folds formed by the unit identifier (instance or
user), so that repeated draws of the same unit never cross folds.  An action with fewer than dim(Z) + 2
training rows in a fold uses a pooled fit.  The eight estimators are REAL_DATA_ESTIMATORS of src/registry.py
with the logging probabilities known.
"""
from __future__ import annotations

import numpy as np

from .nuisance import _huber_irls, _ols
from .registry import REAL_DATA_ESTIMATORS, REAL_DATA_METHODS

N_FOLDS = 5


def crossfit_mhat(Z, A, y, groups, K, loss, rng):
    """n x K cross-fitted reward-model predictions; loss in {'huber', 'ols'}; rng draws the fold labels."""
    ug, inv = np.unique(groups, return_inverse=True)
    fold = rng.permutation(np.arange(len(ug)) % N_FOLDS)[inv]
    n = len(A)
    mhat = np.empty((n, K))
    fit = _huber_irls if loss == "huber" else _ols
    for k in range(N_FOLDS):
        te = fold == k
        tr = ~te
        Zt, At, yt = Z[tr], A[tr], y[tr]
        pooled = None
        for a in range(K):
            m = At == a
            if m.sum() < Z.shape[1] + 2:
                if pooled is None:
                    pooled = fit(Zt, yt)
                b = pooled
            else:
                b = fit(Zt[m], yt[m])
            mhat[te, a] = b[0] + Z[te] @ b[1:]
    return mhat


def estimate_all(pi, A, y, mhat, pb):
    """{method: (estimate, se, tau, huber-truncated fraction, capped fraction)} for the eight estimators."""
    out = {}
    for name in REAL_DATA_METHODS:
        fn, kw = REAL_DATA_ESTIMATORS[name]
        if name in ("DM", "IPW", "SNIPW", "DR"):
            v, se = fn(pi, A, y, mhat, pb, **kw)
            dg = {}
        else:
            v, se, dg = fn(pi, A, y, mhat, pb, return_diag=True, **kw)
        out[name] = (v, se, dg.get("tau", np.nan), dg.get("frac_huber", np.nan), dg.get("frac_clipped", np.nan))
    return out
