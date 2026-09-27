"""RCF-DR-VR and RCF-DR-VR-SN: robust cross-fitted DR with vanishing robustification.

    V_VR = (1/n) sum_t [ mhat_pi(X_t) + wt_t * psi_{tau_n}(r_t) / p_hat ]
    r_t  = Y_t - mhat(X_t, A_t),   w_t = pi(A_t|X_t) / pi_b(A_t|X_t),   wt_t = min(w_t, c_n)
    tau_n = kappa * MAD(r) * n^beta           (final: kappa = 0.5, beta = 1/4)
    c_n   = wbar * sqrt(n / ln n)             (global cap)
    p_hat = max(mean 1{|r_t| <= tau_n}, 0.5)  (one-step rescaling)

sn = 'global' gives the self-normalised correction (RCF-DR-VR-SN):
    V_SN = (1/n) sum_t mhat_pi(X_t) + sum_t wt_t psi_t / p_hat / sum_t wt_t.
tail = 'exact' adds the tail term K_hat * m_hat_r used only by RCF-DR-VR-tail in the adaptive-logging
supplement (K_hat = mean_t sum_a (pi - c_n pi_b)_+ is the known clipped policy mass, m_hat_r the
weight-weighted mean of the rescaled truncated residuals among the top q_tail = 10% of weights).
The standard error is sd(influence values) / sqrt(n).
"""
from __future__ import annotations

import numpy as np

MAD_C = 1.4826


def _mad(x):
    s = MAD_C * np.median(np.abs(x - np.median(x)))
    return s if s > 1e-12 else float(np.std(x) + 1e-12)


def vr_dr(pi, A, Y, mhat, pbhat, kappa=0.5, beta=0.25, sn="none", tail=None, q_tail=0.10, rescale=True,
          return_diag=False, **kw):
    n = len(A)
    idx = np.arange(n)
    w = pi[idx, A] / pbhat[idx, A]
    mpi = (pi * mhat).sum(axis=1)
    r = Y - mhat[idx, A]

    # ---- residual threshold and one-step rescaling
    tau = kappa * _mad(r) * n ** beta if np.isfinite(kappa) else np.inf
    psi = np.clip(r, -tau, tau)
    p_hat = max(np.mean(np.abs(r) <= tau), 0.5) if (rescale and np.isfinite(tau)) else 1.0
    psi = psi / p_hat

    # ---- global weight cap c_n = wbar sqrt(n / ln n)
    c = np.full(n, w.mean() * np.sqrt(n / np.log(n)))
    wt = np.minimum(w, c)

    d = wt * psi
    # ---- aggregation (possibly self-normalised)
    if sn == "none":
        g = mpi + d
    elif sn == "global":
        g = mpi.copy()
        Cb = d.sum() / wt.sum()
        g += Cb + (wt / wt.mean()) * (psi - Cb)
    else:
        raise ValueError(sn)

    # ---- tail correction (RCF-DR-VR-tail, adaptive-logging supplement only)
    B_tail = 0.0
    if tail is not None:
        if tail != "exact":
            raise ValueError(tail)
        top = w >= np.quantile(w, 1 - q_tail)
        m_r = float((w[top] * psi[top]).sum() / w[top].sum())
        Kt = np.clip(pi - c[:, None] * pbhat, 0, None).sum(axis=1)
        K = float(Kt.mean())
        B_tail = K * m_r
        # influence contribution: K_t m_r + K (w/Wtop)(psi - m_r) on the top set
        infl = Kt * m_r
        infl = infl + np.where(top, K * n * w * (psi - m_r) / w[top].sum(), 0.0)
        g = g + infl
    v = float(g.mean())
    se = float(g.std(ddof=1) / np.sqrt(n))
    if return_diag:
        return v, se, dict(tau=tau, c=float(np.median(c)), frac_clipped=float(np.mean(w > c)),
                           frac_huber=float(np.mean(np.abs(r) > tau)), B_tail=B_tail)
    return v, se
