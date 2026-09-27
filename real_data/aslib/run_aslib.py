"""ASlib application: estimator runs under semi-synthetic logging over the full solver-instance outcome table.

Pre-registered in docs/aslib/PREREGISTRATION.md. The eight estimators are src/registry.py
REAL_DATA_ESTIMATORS (frozen tuning); the cross-fitted reward model is src/realdata.crossfit_mhat.

usage: python -m real_data.aslib.run_aslib SCENARIO [n_reps] [n_jobs]      (final: 500 replications)
output: results/aslib/raw/<scenario>/cell_<overlap>_<n>.pkl.gz (one row per rep x variant x policy x method)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from real_data.aslib.aslib_design import (Design, reward_matrix, calibrate_kappa, pop_ess, ESS_TARGETS_FINAL,  # noqa: E402
                                          SCEN_ID)
from real_data.aslib.paths import RAW  # noqa: E402
from src.diagnostics import hill_alpha  # noqa: E402
from src.realdata import crossfit_mhat, estimate_all  # noqa: E402

SIZES = [250, 500, 1000, 2000, 4000]
OVERLAPS = ["good", "moderate", "poor"]
MASTER = 20260925


def one_rep(ctx, overlap, n, rep):
    ss = np.random.SeedSequence([MASTER, SCEN_ID[ctx["scenario"]], OVERLAPS.index(overlap), SIZES.index(n), rep])
    streams = {"eval": ss.spawn(3), "allsolved": ss.spawn(3)}   # (sample, action, fold) per population
    rows = []
    PB = ctx["pb"][overlap]
    K = ctx["K"]
    for pop_name, pop in ctx["pops"].items():
        r_samp, r_act, r_fold = [np.random.default_rng(s) for s in streams[pop_name]]
        idx = r_samp.choice(pop, n, replace=True)
        pb = PB[idx]
        u = r_act.random(n)
        A = np.minimum((pb.cumsum(1) < u[:, None]).sum(1), K - 1)
        Z = ctx["pcs"][idx]
        wt = ctx["pols"]["Ridge-sm3 (target)"][idx, A] / pb[np.arange(n), A]
        a_hat, _, _ = hill_alpha(wt)
        wdiag = dict(ess_frac=float(wt.sum() ** 2 / (wt ** 2).sum() / n), max_w=float(wt.max()), alpha_hat=a_hat,
                     frac_capped_vrsn=float(np.mean(wt > wt.mean() * np.sqrt(n / np.log(n)))))
        fold_seed = int(r_fold.integers(2 ** 31))
        for variant in ctx["variants"][pop_name]:
            reward, loss = variant.split("|")
            Y = ctx["Y"][reward]
            y = Y[idx, A]
            fr = np.random.default_rng(fold_seed)          # same folds for every reward variant
            mhat = crossfit_mhat(Z, A, y, idx, K, loss, fr)
            for pname, P in ctx["pols"].items():
                pi = P[idx]
                truth = ctx["truth"][(pop_name, reward)][pname]
                for m, (v, se, tau, fh, fc) in estimate_all(pi, A, y, mhat, pb).items():
                    rows.append((pop_name, reward, loss, pname, m, rep, v, se, truth, tau, fh, fc,
                                 wdiag["ess_frac"], wdiag["max_w"], wdiag["alpha_hat"], wdiag["frac_capped_vrsn"]))
    return rows


COLS = ["pop", "reward", "rmodel", "policy", "method", "rep", "est", "se", "truth", "tau", "frac_huber",
        "frac_clipped", "ess_frac", "max_w", "alpha_hat", "frac_capped_vrsn"]


def build_context(scenario):
    D = Design(scenario)
    pe = D.target()
    pb = {}
    for ov in OVERLAPS:
        tgt = ESS_TARGETS_FINAL[ov]
        k = 0.0 if tgt is None else calibrate_kappa(pe[D.tr], D.g_hat[D.tr], tgt)[0]
        pb[ov] = D.logging(k)
    pols = D.candidates()
    Y = {r: reward_matrix(D.d, r) for r in ("logPAR10", "logPAR1", "speed")}
    solved_all = np.where(D.d["ok"].all(1))[0]
    pops = {"eval": D.ev, "allsolved": np.intersect1d(D.ev, solved_all)}
    variants = {"eval": [f"{D.primary}|huber", f"{'logPAR1' if D.primary == 'logPAR10' else 'logPAR10'}|huber",
                         "speed|huber", f"{D.primary}|ols"],
                "allsolved": ["logPAR1|huber"]}
    truth = {}
    for pop_name, pop in pops.items():
        for r in Y:
            truth[(pop_name, r)] = {p: float((P[pop] * Y[r][pop]).sum(1).mean()) for p, P in pols.items()}
    return dict(scenario=scenario, K=D.K, pcs=D.pcs, pb=pb, pols=pols, Y=Y, pops=pops, variants=variants,
                truth=truth, design=D)


def main():
    scenario = sys.argv[1]
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ctx = build_context(scenario)
    out = RAW / scenario
    out.mkdir(parents=True, exist_ok=True)
    for ov in OVERLAPS:
        for n in SIZES:
            f = out / f"cell_{ov}_{n}.pkl.gz"
            if f.exists():
                print("skip", f.name, flush=True)
                continue
            t = time.time()
            res = Parallel(n_jobs=n_jobs, batch_size=4)(delayed(one_rep)(ctx, ov, n, r) for r in range(n_reps))
            df = pd.DataFrame([row for rr in res for row in rr], columns=COLS)
            df.insert(0, "n", n)
            df.insert(0, "overlap", ov)
            df.to_pickle(f)
            print(f"{scenario} {ov} n={n}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
