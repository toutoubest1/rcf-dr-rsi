"""Evaluation metrics for value estimation and policy selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

Z975 = 1.959964


def value_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """df columns: est, se, truth (+ grouping columns). One row per replication."""
    def agg(g):
        err = g["est"] - g["truth"]
        lo = g["est"] - Z975 * g["se"]
        hi = g["est"] + Z975 * g["se"]
        ok = np.isfinite(err)
        return pd.Series(dict(
            bias=err[ok].mean(),
            rmse=np.sqrt((err[ok] ** 2).mean()),
            mae=err[ok].abs().mean(),
            medae=err[ok].abs().median(),
            sd=g["est"][ok].std(ddof=1),
            cover=((lo <= g["truth"]) & (g["truth"] <= hi))[ok].mean(),
            ci_len=(hi - lo)[ok].mean(),
            avg_se=g["se"][ok].mean(),
            # MC standard error of RMSE^2 -> used to judge whether differences are real
            mse_mcse=(err[ok] ** 2).std(ddof=1) / np.sqrt(ok.sum()),
            n_rep=int(ok.sum()),
        ))
    return df.groupby(by, sort=False).apply(agg, include_groups=False).reset_index()


def selection_metrics(est: np.ndarray, truth: np.ndarray, se: np.ndarray | None = None) -> dict:
    """est: R x M estimates, truth: M true values. Returns summary dict."""
    best = int(np.argmax(truth))
    sel = np.argmax(est, axis=1)
    R = est.shape[0]
    regret = truth[best] - truth[sel]
    wc = est[np.arange(R), sel] - truth[sel]
    rho = np.array([spearmanr(est[r], truth).statistic for r in range(R)])
    top3 = np.argsort(truth)[-3:]
    return dict(
        p_best=float(np.mean(sel == best)),
        p_top3=float(np.mean(np.isin(sel, top3))),
        # true best policy among the 3 with the highest ESTIMATED values
        best_in_est_top3=float(np.mean([best in np.argsort(est[r])[-3:] for r in range(R)])),
        regret=float(regret.mean()),
        regret_q90=float(np.quantile(regret, 0.9)),
        spearman=float(np.nanmean(rho)),
        winners_curse=float(wc.mean()),
        winners_curse_med=float(np.median(wc)),
        regret_median=float(np.median(regret)),
        n_rep=R,
        # naive CI for the selected policy (selected and estimated on the same data)
        sel_cover=(float(np.mean(np.abs(est[np.arange(R), sel] - truth[sel])
                                 <= Z975 * se[np.arange(R), sel])) if se is not None else np.nan),
    )
