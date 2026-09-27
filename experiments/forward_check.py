"""Forward-fitting check (Supplement S6): does forward (chronological) fitting restore the martingale property

For batched adaptive logging with KNOWN logging probabilities, we compute per batch b the mean of the
uncapped DR score minus V, xi_t = mhat_pi(X_t) + w_t (Y_t - mhat(X_t, A_t)) - V, averaged over
replications. By the martingale-difference lemma of Section 3.7 (lem:mds): if mhat is predictable (fit on past batches only),
E[xi_t | H_{t-1}] = 0 exactly, so every batch mean should be 0 up to MC error. With ordinary K-fold
cross-fitting mhat uses future batches and the property holds only approximately. We also record the
clipped VR-SN score's batch means (the clipping drift of lem:drift).
usage: python -m experiments.forward_check
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.dgp import generate_replay
from src.nuisance import crossfit_nuisances
from src.policies import OVERLAP_REGIMES
from src.simulation import RAW, SCENARIOS, _seed, get_truth

from experiments.common import TAB  # noqa: E402


def one_rep(rep, scenario, n, spec, V):
    rng = np.random.default_rng(_seed(4141, "fwdchk", scenario, n, spec, rep))
    d = generate_replay(rng, n, **SCENARIOS[scenario])
    b = d.meta["batch_id"]
    nu = crossfit_nuisances(d.X, d.A, d.Y, outcome_specs=(spec,), prop_specs=(), insample=False, seed=rep,
                            forward=("fwd", "fwd0"), batch=b)
    pi = OVERLAP_REGIMES["poor"](d.X)
    i = np.arange(n)
    w = pi[i, d.A] / d.pb[i, d.A]
    c = w.mean() * np.sqrt(n / np.log(n))
    rows = []
    for scheme in [True, "fwd", "fwd0"]:
        mh = nu[("m", spec, scheme)]
        r = d.Y - mh[i, d.A]
        xi = (pi * mh).sum(1) + w * r - V
        xic = (pi * mh).sum(1) + np.minimum(w, c) * r - V
        for bb in range(5):
            m = b == bb
            rows.append(dict(rep=rep, scenario=scenario, n=n, spec=spec,
                             scheme={True: "kfold", "fwd": "forward", "fwd0": "forward-strict"}[scheme],
                             batch=bb, dr_score=xi[m].mean(), clipped_score=xic[m].mean()))
    return rows


if __name__ == "__main__":
    V = get_truth([OVERLAP_REGIMES["poor"]])["poor"]["value"]
    t0 = time.time(); allr = []
    for sc in ["gauss_adapt", "t3_adapt"]:
        for spec in ["correct_huber", "mis_huber"]:
            res = Parallel(n_jobs=-1)(delayed(one_rep)(r, sc, 2000, spec, V) for r in range(400))
            allr += [x for rr in res for x in rr]
            print(f"[fwdchk] {sc} {spec} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(allr)
    df.to_csv(RAW / "fwdcheck_final.csv.gz", index=False)
    s = df.groupby(["scenario", "spec", "scheme", "batch"]).agg(
        dr_mean=("dr_score", "mean"), dr_mcse=("dr_score", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        clipped_mean=("clipped_score", "mean"),
        clipped_mcse=("clipped_score", lambda x: x.std(ddof=1) / np.sqrt(len(x)))).reset_index()
    s["dr_t"] = s.dr_mean / s.dr_mcse
    s.to_csv(TAB / "fwdcheck_final.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())
