"""Robust mean of the DR pseudo-outcome (RM-Huber ablation of the supplement).

Idea: keep the standard cross-fitted DR pseudo-outcome intact,

    Gamma_i = mhat_pi(X_i) + wt_i (Y_i - mhat(X_i, A_i)),     wt_i = min(w_i, c_n),

and replace the sample mean of Gamma by a robust mean estimator whose
robustification is *asymptotically vanishing* (threshold tau_n -> infinity).
Because E[Gamma] = V(pi) whenever either nuisance is correct (and c_n -> inf),
any location estimator that is asymptotically equivalent to the mean keeps the
DR moment property, while bounding the influence of any single observation by
O(tau_n / n) in finite samples.

Contrast with RCF-DR-os: there the *residual* is Huberised at a FIXED tau
(prop. to MAD of residuals).  Its target E[w psi_tau(delta + eps)] / p differs
from E[w delta] at third order in the outcome-model error delta, so the bias
does not vanish with n when mhat is misspecified.

Threshold (RM-Huber): tau_n = C * MAD(Gamma) * sqrt(n / log n) with C = 1 (rate "sqrt").
Every function returns (theta, se).
"""
from __future__ import annotations

import numpy as np
MAD_C = 1.4826


def robust_scale(g: np.ndarray) -> float:
    s = MAD_C * np.median(np.abs(g - np.median(g)))
    if s <= 1e-12:
        q75, q25 = np.percentile(g, [75, 25])
        s = (q75 - q25) / 1.349
    if s <= 1e-12:
        s = np.std(g) + 1e-12
    return float(s)


RATES = {
    "sqrt": lambda n: np.sqrt(n / np.log(n)),
    "n13": lambda n: n ** (1 / 3),
    "n14": lambda n: n ** 0.25,
}


def tau_n(g: np.ndarray, rate: str = "sqrt", C: float = 1.0) -> float:
    return C * robust_scale(g) * RATES[rate](len(g))


# ----------------------------------------------------------------- estimators


def huber_location(g, rate="sqrt", C=1.0, iters=100, **kw):
    """Huber M-estimator of location with threshold tau_n (scale fixed at MAD).

    Solves sum psi_tau(g_i - theta) = 0 by fixed-point iteration from the median.
    SE: sandwich  sd(psi) / (mean psi' sqrt(n)).
    """
    t = tau_n(g, rate, C)
    th = float(np.median(g))
    for _ in range(iters):
        # Newton step on the monotone estimating equation (psi' = 1{|g - th| <= t})
        dens = max(np.mean(np.abs(g - th) <= t), 1e-3)
        step = np.clip(g - th, -t, t).mean() / dens
        th += step
        if abs(step) < 1e-10 * (1 + abs(th)):
            break
    psi = np.clip(g - th, -t, t)
    dpsi = np.mean(np.abs(g - th) <= t)
    se = psi.std(ddof=1) / (max(dpsi, 1e-3) * np.sqrt(len(g)))
    return float(th), float(se)


ROBUST_MEANS = {"huber": huber_location}
