"""Off-policy value estimators.

Common inputs (one target policy, one replay sample of size n):
    pi    : n x K target-policy probabilities pi(a | X_i)
    A, Y  : logged actions and rewards
    mhat  : n x K outcome-model predictions (cross-fitted or in-sample)
    pbhat : n x K behaviour-propensity predictions

Every estimator returns (value, se) where se is the plug-in standard error
from its (estimated) influence function Gamma_i, so that a Wald 95% CI is
value +/- 1.96 se.  For clipped/Huberised estimators this CI is centred at
the *clipped* estimand, so clipping bias shows up as undercoverage.

General robust DR family (includes DR, clipped DR, robust DR, RCF-DR):

    V = n^{-1} sum_i [ mpi_i + wt_i psi_tau(r_i) ]                    (plain)
    V = n^{-1} sum_i mpi_i + sum_i wt_i psi_tau(r_i) / sum_i wt_i     (self-normalised)

    mpi_i = sum_a pi(a|X_i) mhat(X_i, a),   r_i = Y_i - mhat(X_i, A_i)
    w_i   = pi(A_i|X_i) / pbhat(A_i|X_i),   wt_i = min(w_i, c)
    psi_tau(r) = sign(r) min(|r|, tau)

with c and tau chosen by rules in tuning.py.  The named configurations used in the paper are in
src/registry.py.
"""
from __future__ import annotations

import numpy as np

from .tuning import CLIP_RULES, TAU_RULES


def _pieces(pi, A, Y, mhat, pbhat):
    idx = np.arange(len(A))
    w = pi[idx, A] / pbhat[idx, A]
    mpi = (pi * mhat).sum(axis=1)
    r = Y - mhat[idx, A]
    return w, mpi, r


def _mean_se(g):
    return float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g)))


def dm(pi, A, Y, mhat, pbhat, **kw):
    return _mean_se((pi * mhat).sum(axis=1))


def ips(pi, A, Y, mhat, pbhat, **kw):
    w, _, _ = _pieces(pi, A, Y, mhat, pbhat)
    return _mean_se(w * Y)


def snips(pi, A, Y, mhat, pbhat, **kw):
    w, _, _ = _pieces(pi, A, Y, mhat, pbhat)
    v = float((w * Y).sum() / w.sum())
    g = v + (w / w.mean()) * (Y - v)
    return v, float(g.std(ddof=1) / np.sqrt(len(g)))


def robust_dr(pi, A, Y, mhat, pbhat, clip="none", tau="none", self_norm=False,
              rescale=False, return_diag=False, **kw):
    """General (clipped / Huberised / self-normalised) doubly robust estimator.

    rescale=True ('one-step' / Fisher-consistency correction):
    the Huber correction is divided by  p_hat = mean 1{|r_i| <= tau}  (the
    empirical E[psi'_tau(eps)]).  Rationale: if r = delta(X,A) + eps with a
    small outcome-model error delta and symmetric noise eps, then
    E[w psi_tau(r)] ~= E[w delta] * E[psi'_tau(eps)], i.e. the plain Huber
    correction removes only a fraction P(|eps|<=tau) of the DM bias
    (~82% for Gaussian noise at kappa = 1.345).  Dividing by p_hat restores
    first-order double robustness while keeping each contribution bounded
    by c * tau / p_hat.
    """
    w, mpi, r = _pieces(pi, A, Y, mhat, pbhat)
    t = TAU_RULES[tau](r, w) if isinstance(tau, str) else tau(r, w)
    psi = np.clip(r, -t, t)
    if rescale and np.isfinite(t):
        psi = psi / max(np.mean(np.abs(r) <= t), 0.5)
    c = CLIP_RULES[clip](w, psi, mpi) if isinstance(clip, str) else clip(w, psi, mpi)
    wt = np.minimum(w, c)
    if self_norm:
        wbar = wt.mean()
        corr = float((wt * psi).sum() / wt.sum())
        g = mpi + corr + (wt / wbar) * (psi - corr)
    else:
        g = mpi + wt * psi
    v, se = _mean_se(g)
    if return_diag:
        return v, se, dict(tau=t, c=c, frac_clipped=float(np.mean(w > c)),
                           frac_huber=float(np.mean(np.abs(r) > t)))
    return v, se


def rm_dr(pi, A, Y, mhat, pbhat, clip="none", mean="huber", rate="sqrt", C=1.0, blocks="log",
          return_diag=False, **kw):
    """RM-Huber ablation: robust mean of the (optionally n-cap-clipped) DR pseudo-outcome.

        Gamma_i = mhat_pi(X_i) + min(w_i, c_n) (Y_i - mhat(X_i, A_i))
        V_hat   = RobustMean(Gamma_1, ..., Gamma_n)   with tau_n -> infinity
    """
    from .robust_mean import ROBUST_MEANS, tau_n
    w, mpi, r = _pieces(pi, A, Y, mhat, pbhat)
    c = CLIP_RULES[clip](w, r, mpi)
    g = mpi + np.minimum(w, c) * r
    v, se = ROBUST_MEANS[mean](g, rate=rate, C=C, blocks=blocks)
    if return_diag:
        t = tau_n(g, rate, C) if mean == "huber" else np.nan
        frac = float(np.mean(np.abs(g - np.median(g)) > t)) if np.isfinite(t) else np.nan
        return v, se, dict(c=c, tau=t, frac_clipped=float(np.mean(w > c)), frac_huber=frac)
    return v, se
