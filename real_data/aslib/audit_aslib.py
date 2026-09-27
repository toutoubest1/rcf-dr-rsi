"""Audit of all ASlib runtime scenarios and scenario selection (no estimator is run here).

usage: python -m real_data.aslib.audit_aslib   -> results/aslib/tables/aslib_audit.csv, results/aslib/tables/aslib_audit_solvers_<scenario>.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from real_data.aslib.aslib_io import DATA, load_scenario  # noqa: E402
from real_data.aslib.paths import TAB as OUT  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)


def bowley(x):
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return float((q3 + q1 - 2 * q2) / (q3 - q1)) if q3 > q1 else np.nan


def mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def audit(name):
    desc, runs, feats = load_scenario(name)
    perf_col = "runtime" if "runtime" in runs.columns else "PAR10"   # PAR10 scenarios: ok rows hold the runtime
    runs = runs.rename(columns={perf_col: "runtime"})
    cutoff = float(desc["algorithm_cutoff_time"])
    algs = sorted(runs["algorithm"].unique())
    insts = sorted(runs["instance_id"].unique())
    reps = int(runs["repetition"].max())
    r1 = runs[runs["repetition"] == 1]
    status = r1["runstatus"].str.lower()
    # full matrix of observed outcomes (repetition 1)
    M = r1.pivot_table(index="instance_id", columns="algorithm", values="runtime", aggfunc="first").reindex(index=insts, columns=algs)
    S = r1.pivot_table(index="instance_id", columns="algorithm", values="runstatus", aggfunc="first").reindex(index=insts, columns=algs)
    n_cells = len(insts) * len(algs)
    missing_cells = int(M.isna().sum().sum())
    ok = (S == "ok")
    to = (S == "timeout")
    other_fail = S.notna() & ~ok & ~to
    # runtime reward matrix with the ASlib convention: unsolved (timeout/memout/crash/other) -> cutoff (PAR1 value)
    R = M.where(ok, cutoff).clip(lower=1e-3)
    logR = np.log(R.values)
    rt_ok = M.values[ok.values]
    rt_ok = rt_ok[np.isfinite(rt_ok)]
    # instance features
    fcols = [c for c in feats.columns if c not in ("instance_id", "repetition")]
    F = feats[feats["repetition"] == feats["repetition"].min()].set_index("instance_id").reindex(insts)[fcols]
    frac_feat_missing = float(F.isna().mean().mean())
    Fx = F.apply(lambda c: c.fillna(c.median())).values.astype(float)
    Fx = (Fx - Fx.mean(0)) / (Fx.std(0) + 1e-12)
    Fx = np.nan_to_num(Fx)
    # residuals of a cross-fitted per-solver ridge model of log runtime (proxy for the reward-model residuals)
    resid = []
    kf = KFold(5, shuffle=True, random_state=0)
    for j in range(len(algs)):
        y = logR[:, j]
        pred = np.empty_like(y)
        for tr, te in kf.split(Fx):
            m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Fx[tr], y[tr])
            pred[te] = m.predict(Fx[te])
        resid.append(y - pred)
    resid = np.concatenate(resid)
    resid_solved = resid[ok.values.T.ravel()]
    # solver variation
    mean_log = np.nanmean(logR, axis=0)
    sbs = int(np.argmin(mean_log))
    vbs_log = np.nanmin(logR, axis=1)
    best = np.nanargmin(logR, axis=1)
    share_best = np.bincount(best, minlength=len(algs)) / len(insts)
    par10 = M.where(ok, 10 * cutoff)
    row = dict(
        scenario=name, perf_column=perf_col, n_instances=len(insts), n_solvers=len(algs), n_features=len(fcols), repetitions=reps,
        cutoff=cutoff, frac_missing_outcomes=missing_cells / n_cells, frac_feature_missing=frac_feat_missing,
        all_outcomes_observed=missing_cells == 0,
        timeout_rate=float(to.values.sum() / n_cells), other_fail_rate=float(other_fail.values.sum() / n_cells),
        unsolved_rate=float((~ok).values.sum() / n_cells),
        timeout_rate_solver_min=float((~ok).mean().min()), timeout_rate_solver_med=float((~ok).mean().median()),
        timeout_rate_solver_max=float((~ok).mean().max()),
        frac_inst_unsolved_by_all=float((~ok).all(axis=1).mean()),
        rt_q10=np.quantile(rt_ok, .1), rt_q50=np.quantile(rt_ok, .5), rt_q90=np.quantile(rt_ok, .9),
        rt_q99=np.quantile(rt_ok, .99), rt_min=rt_ok.min(),
        skew_rt_solved=float(skew(rt_ok)), skew_logrt_solved=float(skew(np.log(np.clip(rt_ok, 1e-3, None)))),
        skew_logrt_all=float(skew(logR.ravel())), bowley_logrt_all=bowley(logR.ravel()),
        frac_at_cutoff=float((~ok).values.mean()),
        resid_skew=float(skew(resid)), resid_bowley=bowley(resid), resid_exkurt=float(kurtosis(resid)),
        resid_mad=mad(resid), resid_sd=float(resid.std()), resid_frac_gt3mad=float(np.mean(np.abs(resid - np.median(resid)) > 3 * mad(resid))),
        resid_skew_solved=float(skew(resid_solved)), resid_bowley_solved=bowley(resid_solved),
        sbs=algs[sbs], sbs_mean_log=float(mean_log[sbs]), vbs_mean_log=float(vbs_log.mean()),
        gap_log=float(mean_log[sbs] - vbs_log.mean()), sd_solver_mean_log=float(np.std(mean_log)),
        n_solvers_best_ge5pct=int((share_best >= 0.05).sum()), share_sbs_best=float(share_best[sbs]),
        par10_sbs=float(par10.mean().min()), par10_vbs=float(par10.min(axis=1).mean()),
    )
    solv = pd.DataFrame(dict(solver=algs, unsolved_rate=(~ok).mean().values, timeout_rate=to.mean().values,
                             mean_log_runtime=mean_log, share_best=share_best,
                             median_runtime_solved=[np.nanmedian(M[a][ok[a]]) for a in algs]))
    solv.to_csv(OUT / f"aslib_audit_solvers_{name}.csv", index=False)
    return row


if __name__ == "__main__":
    names = sorted(n for n in os.listdir(DATA) if (DATA / n / "description.txt").exists() and not n.endswith("-ALGO"))
    rows = []
    for n in names:
        try:
            desc = __import__("yaml").safe_load(open(DATA / n / "description.txt"))
            pt = desc.get("performance_type")
            pt = pt if isinstance(pt, list) else [pt]
            if pt[0] != "runtime":
                print("skip (not runtime-primary)", n, pt)
                continue
            rows.append(audit(n))
            print("ok", n, flush=True)
        except Exception as e:
            print("ERR", n, repr(e)[:200], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aslib_audit.csv", index=False)
    print(df[["scenario", "n_instances", "n_solvers", "n_features", "repetitions", "frac_missing_outcomes", "unsolved_rate",
              "skew_logrt_all", "resid_bowley", "resid_exkurt", "gap_log", "n_solvers_best_ge5pct"]].round(3).to_string())
