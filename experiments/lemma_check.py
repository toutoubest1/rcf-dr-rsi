"""Numerical check of the Huber-remainder lemma (Table S1 of the supplement) and of the Pareto-weight
truncation calculations of Proposition prop:pareto.

Lemma 1 (exact identity, symmetric noise with density f):
    g(d) := E psi_tau(d + eps) = p_tau d + int_0^d int_0^s [f(tau+u) - f(tau-u)] du ds,
    p_tau = P(|eps| <= tau).
Bounds on R(d, tau) := g(d)/p_tau - d:
    (moment)  |R| <= |d| P(|eps| > tau - |d|) / p_tau                       (|d| < tau)
    (cubic)   |R| <= (|d|^3 / 3) sup_{|v - tau| <= |d|} |f'(v)| / p_tau
    leading term R ~ d^3 f'(tau) / (3 p_tau)  as d/tau -> 0.
usage: python -m experiments.lemma_check   (writes results/simulation/tables/theory_*.csv)
"""
import numpy as np
import pandas as pd
from scipy import integrate, stats

from experiments.common import TAB

LAWS = {
    "gauss": stats.norm(),
    "t3": stats.t(3),
    "t2": stats.t(2),
}


def g_exact(law, d, tau):
    """E psi_tau(d + eps) by numerical integration (split at the kinks)."""
    f = law.pdf
    lo, hi = -tau - d, tau - d          # region where |d + e| <= tau
    mid = integrate.quad(lambda e: (d + e) * f(e), lo, hi, limit=200)[0]
    return mid + tau * law.sf(hi) - tau * law.cdf(lo)


def fprime(law, v, h=1e-5):
    return (law.pdf(v + h) - law.pdf(v - h)) / (2 * h)


def lemma1_table():
    rows = []
    for name, law in LAWS.items():
        for tau in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
            p = law.cdf(tau) - law.cdf(-tau)
            for d in [0.1, 0.25, 0.5, 1.0]:
                if d >= tau:
                    continue
                R = g_exact(law, d, tau) / p - d
                vs = np.linspace(tau - d, tau + d, 201)
                cubic = d ** 3 / 3 * np.max(np.abs(fprime(law, vs))) / p
                moment = d * (2 * law.sf(tau - d)) / p
                lead = d ** 3 * fprime(law, tau) / (3 * p)
                rows.append(dict(law=name, tau=tau, delta=d, R=R, lead_term=lead,
                                 bound_cubic=cubic, bound_moment=moment,
                                 cubic_ok=abs(R) <= cubic * (1 + 1e-6) + 1e-12,
                                 moment_ok=abs(R) <= moment * (1 + 1e-6) + 1e-12))
    return pd.DataFrame(rows)


def decay_exponents(df):
    """Fit log|R| ~ a + b log tau for fixed delta (tau >= 3)."""
    out = []
    for (law, d), g in df[df.tau >= 3].groupby(["law", "delta"]):
        g = g[np.abs(g.R) > 1e-14]
        if len(g) >= 3:
            b = np.polyfit(np.log(g.tau), np.log(np.abs(g.R)), 1)[0]
            out.append(dict(law=law, delta=d, slope_log_absR_vs_log_tau=b))
    return pd.DataFrame(out)


def pareto_tail_table(alphas=(1.8, 2.0, 2.4, 3.0)):
    """For P(w > t) = t^-alpha (t >= 1): truncation bias E[(w-c)_+] and truncated second
    moment E[min(w,c)^2] as functions of c; and the n-rate implied by c_n = sqrt(n/log n)."""
    rows = []
    for a in alphas:
        for n in [1e3, 1e4, 1e5, 1e6]:
            c = np.sqrt(n / np.log(n))
            bias = c ** (1 - a) / (a - 1)                       # E[(w - c)_+]
            if a > 2:
                m2 = a / (a - 2) - 2 * c ** (2 - a) / (a - 2)   # E[min(w,c)^2], finite limit
            elif a == 2:
                m2 = 1 + 2 * np.log(c)
            else:
                m2 = (a - 2 * c ** (2 - a)) / (a - 2)
            se = np.sqrt(m2 / n)
            rows.append(dict(alpha=a, n=int(n), c_n=c, trunc_bias=bias, sd_proxy=se, bias_over_se=bias / se))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = lemma1_table()
    df.to_csv(TAB / "theory_lemma1_check.csv", index=False)
    print("all cubic bounds hold:", bool(df.cubic_ok.all()), " all moment bounds hold:", bool(df.moment_ok.all()))
    ex = decay_exponents(df)
    ex.to_csv(TAB / "theory_lemma1_decay.csv", index=False)
    print(ex.round(2).to_string())
    print(df[(df.delta == 0.5)].round(6).to_string())
    pt = pareto_tail_table()
    pt.to_csv(TAB / "theory_pareto_truncation.csv", index=False)
    print(pt.round(4).to_string())
