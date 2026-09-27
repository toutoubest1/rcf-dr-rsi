"""KuaiRec semi-synthetic OPE study with the frozen estimators"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from real_data.kuairec.kuairec_design import KuaiRec, calibrate_kappa, pop_ess, zrow  # noqa: E402
from real_data.kuairec.paths import RAW, TAB  # noqa: E402
from src.diagnostics import hill_alpha  # noqa: E402
from src.realdata import crossfit_mhat, estimate_all  # noqa: E402
from src.vr import _mad  # noqa: E402  (MAD with SD fallback, as used inside vr_dr)

MASTER = 20260925
DS_ID = {"kuairec": 11}
KUAIREC = dict(sizes=[250, 500, 1000, 2000, 4000], overlaps=["good", "moderate", "poor"],
               ess_targets={"good": None, "moderate": 0.05, "poor": 0.01},
               variants=["log1p_wr|huber", "wr|huber", "log1p_wr|ols"])
TARGET = "Ridge-sm3 (target)"
COLS = ["variant", "policy", "method", "rep", "est", "se", "truth", "tau", "frac_huber", "frac_clipped",
        "resid_mad", "ess_frac", "max_w", "alpha_hat", "frac_capped_vrsn"]


def run_estimators(pols, idx, A, y, mhat, pb, truth, variant, rep, n):
    rows = []
    r = y - mhat[np.arange(n), A]
    rmad = float(_mad(r))
    for pname, P in pols.items():
        pi = P[idx]
        w = pi[np.arange(n), A] / pb[np.arange(n), A]
        a_hat = hill_alpha(w)[0] if w.max() > w.min() else np.inf
        wd = (float(w.sum() ** 2 / (w ** 2).sum() / n) if w.sum() > 0 else 0.0, float(w.max()), a_hat,
              float(np.mean(w > w.mean() * np.sqrt(n / np.log(n)))))
        for m, (v, se, tau, fh, fc) in estimate_all(pi, A, y, mhat, pb).items():
            rows.append((variant, pname, m, rep, v, se, truth[pname], tau, fh, fc, rmad) + wd)
    return rows


def kuairec_context():
    d = KuaiRec()
    pb = {}
    kap = {}
    for ov, tgt in KUAIREC["ess_targets"].items():
        k = 0.0 if tgt is None else calibrate_kappa(d.target[d.tr], np.log(d.hist_share[d.tr] + 0.01), tgt)[0]
        kap[ov] = k
        pb[ov] = d.logging(k)
    truth = {r: {p: d.truth(P, r) for p, P in d.cands.items()} for r in ("log1p_wr", "wr")}
    d.S = None                                     # large frame not needed by the replications
    return dict(d=d, pb=pb, kappa=kap, truth=truth)


def kuairec_rep(ctx, ov, n, rep):
    d = ctx["d"]
    ss = np.random.SeedSequence([MASTER, DS_ID["kuairec"], KUAIREC["overlaps"].index(ov), KUAIREC["sizes"].index(n), rep])
    r_s, r_a, r_v, r_f = [np.random.default_rng(s) for s in ss.spawn(4)]
    idx = r_s.choice(d.ev, n, replace=True)
    pb = ctx["pb"][ov][idx]
    A = np.minimum((pb.cumsum(1) < r_a.random(n)[:, None]).sum(1), d.K - 1)
    u_draw = r_v.random(n)                         # same video draw for every reward variant
    lo = d.start[idx * d.K + A]
    hi = d.stop[idx * d.K + A]
    j = lo + np.floor(u_draw * (hi - lo)).astype(int)
    fold_seed = int(r_f.integers(2 ** 31))
    Z = d.pcs[idx]
    rows = []
    for variant in KUAIREC["variants"]:
        reward, loss = variant.split("|")
        y = np.log1p(d.wr[j]) if reward == "log1p_wr" else d.wr[j]
        mhat = crossfit_mhat(Z, A, y, idx, d.K, loss, np.random.default_rng(fold_seed))
        rows += run_estimators(d.cands, idx, A, y, mhat, pb, ctx["truth"][reward], variant, rep, n)
    return rows


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cfg = KUAIREC
    t0 = time.time()
    ctx = kuairec_context()
    print(f"context built in {time.time() - t0:.0f}s", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for ov in cfg["overlaps"]:
        for n in cfg["sizes"]:
            f = RAW / f"cell_{ov}_{n}.pkl.gz"
            if f.exists():
                continue
            t = time.time()
            res = Parallel(n_jobs=n_jobs, batch_size=4)(delayed(kuairec_rep)(ctx, ov, n, r) for r in range(n_reps))
            df = pd.DataFrame([row for rr in res for row in rr], columns=COLS)
            df.insert(0, "n", n)
            df.insert(0, "overlap", ov)
            df.to_pickle(f)
            print(f"kuairec {ov} n={n}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
