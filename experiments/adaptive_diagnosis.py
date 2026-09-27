"""Batch decomposition of the adaptive-logging bias (Figure 4, right panel; Supplement S6).

This is the diagnostic run behind the right panel of Figure 4 (base seed 3031): the capped estimator uses the
threshold tau = 0.28 MAD n^(1/3) and the cap c = wbar sqrt(n/ln n), as disclosed in the figure caption.

Regime C-huber-oracle (reward model linear-Huber = wrong, logging probabilities known), poor overlap.
For each replication and batch b we record weight diagnostics and decompose the difference between the
VR estimator (VR-n13: tau = 0.28 MAD n^(1/3), c = wbar sqrt(n/ln n)) and uncapped DR, which is unbiased
here because the logging probabilities are known:

  VR - DR = (1/n) sum_b sum_{t in b} [ wt_t (psi_t/p - r_t)   <- Huberisation part
                                      + (wt_t - w_t) r_t ]     <- clipping part
We also record the noise-free clipping part (1/n) sum (wt_t - w_t) delta_t with delta = mu - mhat
(its expectation is the clipping bias), and the share of the needed correction sum w_t delta_t carried
by clipped mass.
usage: python -m experiments.adaptive_diagnosis
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.dgp import generate_replay
from src.nuisance import crossfit_nuisances
from src.policies import OVERLAP_REGIMES
from src.simulation import RAW, SCENARIOS, _seed

from experiments.common import TAB  # noqa: E402
K13 = 2.8 / 1000 ** (1 / 3)


def one_rep(rep, scenario, n):
    rng = np.random.default_rng(_seed(3031, "diag", scenario, n, rep))
    d = generate_replay(rng, n, **SCENARIOS[scenario])
    bid = d.meta.get("batch_id", np.repeat(np.arange(5), n // 5)[:n])
    nu = crossfit_nuisances(d.X, d.A, d.Y, outcome_specs=("mis_huber",), prop_specs=(), insample=False, seed=rep)
    mh = nu[("m", "mis_huber", True)]
    pi = OVERLAP_REGIMES["poor"](d.X)
    i = np.arange(n)
    w = pi[i, d.A] / d.pb[i, d.A]
    r = d.Y - mh[i, d.A]
    delta = d.mu[i, d.A] - mh[i, d.A]
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    tau = K13 * mad * n ** (1 / 3)
    p = np.mean(np.abs(r) <= tau)
    psi = np.clip(r, -tau, tau) / p
    c = w.mean() * np.sqrt(n / np.log(n))
    wt = np.minimum(w, c)
    rows = []
    for b in range(5):
        m = bid == b
        wb = w[m]
        need = (wb * delta[m]).sum()
        rows.append(dict(
            rep=rep, scenario=scenario, n=n, batch=b,
            mean_w=wb.mean(), ess_frac=wb.sum() ** 2 / (wb ** 2).sum() / m.sum(), max_w=wb.max(),
            frac_clipped=np.mean(wb > c),
            dr_corr=np.mean(wb * r[m]), vr_corr=np.mean(wt[m] * psi[m]), needed_corr=np.mean(wb * delta[m]),
            huber_part=(wt[m] * (psi[m] - r[m])).sum() / n,
            clip_part=((wt[m] - wb) * r[m]).sum() / n,
            clip_part_noisefree=((wt[m] - wb) * delta[m]).sum() / n,
            share_needed_in_clipped=((wb - wt[m]) * delta[m]).sum() / need if abs(need) > 1e-9 else np.nan,
        ))
    return rows


if __name__ == "__main__":
    t0 = time.time(); allr = []
    for sc in ["gauss", "gauss_adapt", "t3", "t3_adapt", "gross", "gross_adapt"]:
        for n in [1000, 2000]:
            res = Parallel(n_jobs=-1)(delayed(one_rep)(r, sc, n) for r in range(200))
            allr += [x for rr in res for x in rr]
            print(f"[diag] {sc} n={n} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(allr)
    df.to_csv(RAW / "adaptdiag_r3.csv.gz", index=False)
    s = df.groupby(["scenario", "n", "batch"]).agg(
        mean_w=("mean_w", "mean"), ess_frac=("ess_frac", "median"), max_w_med=("max_w", "median"),
        max_w_q90=("max_w", lambda x: np.quantile(x, .9)), frac_clipped=("frac_clipped", "mean"),
        dr_corr=("dr_corr", "mean"), vr_corr=("vr_corr", "mean"), needed_corr=("needed_corr", "mean"),
        huber_part=("huber_part", "mean"), clip_part=("clip_part", "mean"),
        clip_part_noisefree=("clip_part_noisefree", "mean"),
        share_needed_in_clipped=("share_needed_in_clipped", "median")).reset_index()
    s.to_csv(TAB / "adaptdiag_r3_by_batch.csv", index=False)
    tot = df.groupby(["scenario", "n", "rep"])[["huber_part", "clip_part", "clip_part_noisefree"]].sum() \
            .groupby(["scenario", "n"]).mean().reset_index()
    tot.to_csv(TAB / "adaptdiag_r3_totals.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(4).to_string()); print(tot.round(4).to_string())
