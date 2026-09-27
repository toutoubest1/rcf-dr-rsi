"""Monte Carlo driver: scenarios, nuisance regimes, replication loop, truth cache.

Raw replication results are written to results/simulation/raw/<tag>.csv.gz (never overwritten).
True policy values are cached in results/simulation/truth_cache.json (4e6 Monte Carlo draws each).
"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .dgp import generate_replay, true_value
from .estimators import rm_dr, robust_dr
from .registry import SIM_ESTIMATORS
from .vr import vr_dr
from .nuisance import crossfit_nuisances
from .diagnostics import overlap_diagnostic
from . import policies as _pol
from .policies import OVERLAP_REGIMES, PolicySpec

ROOT = Path(__file__).resolve().parents[1]
SIMROOT = ROOT / "results" / "simulation"
RAW = SIMROOT / "raw"
TRUTH_FILE = SIMROOT / "truth_cache.json"
N_MC_TRUTH = 4_000_000

# ------------------------------------------------------------------ scenarios (noise laws, logging)
SCENARIOS = {
    "gauss":     dict(noise="gauss"),
    "t3":        dict(noise="t3"),
    "t2":        dict(noise="t2"),
    "contam05":  dict(noise="contam", rate=0.05, sigma_out=10.0),          # contaminated normal
    "gross":     dict(noise="gross", rate=0.02, lo=20.0, hi=40.0),           # symmetric gross errors
    "gross_pos": dict(noise="gross_pos", rate=0.02, lo=20.0, hi=40.0),       # one-sided gross errors
    # batched adaptive logging (5 batches, logger greedily refitted between batches)
    "gauss_adapt": dict(noise="gauss", logging="batched", batches=5, gamma=2.0),
    "t3_adapt":    dict(noise="t3", logging="batched", batches=5, gamma=2.0),
    "gross_adapt": dict(noise="gross", rate=0.02, lo=20.0, hi=40.0, logging="batched", batches=5, gamma=2.0),
    # logger that partly exploits the nonlinear reward terms (strong propensity misspecification)
    "gauss_hlog": dict(noise="gauss", hgamma=1.5),
    "t3_hlog":    dict(noise="t3", hgamma=1.5),
    "gross_hlog": dict(noise="gross", rate=0.02, lo=20.0, hi=40.0, hgamma=1.5),
}

# nuisance regimes: (outcome spec, propensity spec, fitting scheme)
#   fitting scheme: True = 5-fold cross-fitting; "fwd" = forward fitting; "fwd0" = forward-strict
REGIMES = {
    # IID grid: A = both correct, B = propensity (strongly) wrong, C = reward model wrong, D = both wrong
    "A-huber": ("correct_huber", "correct", True),
    "B-strong": ("correct_huber", "mis_strong", True),
    "C-huber": ("mis_huber", "correct", True),
    "D-strong": ("mis_huber", "mis_strong", True),
    # known logging probabilities (oracle propensity)
    "A-huber-oracle": ("correct_huber", "oracle", True),
    "C-huber-oracle": ("mis_huber", "oracle", True),
    # h-logger (strong propensity misspecification); 'correct_h' is the correctly specified propensity model
    "A-h": ("correct_huber", "correct_h", True),
    "B-h": ("correct_huber", "mis_strong", True),
    "C-h": ("mis_huber", "correct_h", True),
    "D-h": ("mis_huber", "mis_strong", True),
    # forward (chronological) fitting for batched adaptive logs, known logging probabilities
    "A-fwd-oracle": ("correct_huber", "oracle", "fwd"),
    "C-fwd-oracle": ("mis_huber", "oracle", "fwd"),
    "A-fwd0-oracle": ("correct_huber", "oracle", "fwd0"),
    "C-fwd0-oracle": ("mis_huber", "oracle", "fwd0"),
}


def _key(spec: PolicySpec, corr: str) -> str:
    return f"{_pol.DGP_VERSION}|{spec.name}|lam={spec.lam}|alpha={spec.alpha}|pid={spec.pid}|eta={spec.eta}|{corr}"


def get_truth(specs, corr="indep", n_mc=N_MC_TRUTH):
    RAW.mkdir(parents=True, exist_ok=True)
    cache = json.loads(TRUTH_FILE.read_text()) if TRUTH_FILE.exists() else {}
    out, changed = {}, False
    for s in specs:
        k = _key(s, corr)
        if k not in cache:
            v, se, diag = true_value(s, corr=corr, n_mc=n_mc, return_diag=True)
            cache[k] = dict(value=v, mc_se=se, **diag)
            changed = True
        out[s.name] = cache[k]
    if changed:
        TRUTH_FILE.write_text(json.dumps(cache, indent=1))
    return out


def _seed(*parts):
    return [zlib.crc32(str(p).encode()) for p in parts]


def run_replication(rep, scenario, n, corr, policies, methods, regimes, base_seed,
                    registry=None, n_folds=5):
    reg = SIM_ESTIMATORS if registry is None else registry
    rng = np.random.default_rng(_seed(base_seed, scenario, n, corr, rep))
    cfg = dict(SCENARIOS[scenario])
    data = generate_replay(rng, n, corr=corr, **cfg)
    X, A, Y, pbt = data.X, data.A, data.Y, data.pb
    need_m = sorted({REGIMES[r][0] for r in regimes})
    need_p = sorted({REGIMES[r][1] for r in regimes} - {"oracle"})
    fwd = sorted({REGIMES[r][2] for r in regimes if isinstance(REGIMES[r][2], str)})
    bid = data.meta.get("batch_id")
    nuis = crossfit_nuisances(X, A, Y, outcome_specs=need_m, prop_specs=need_p,
                              n_folds=n_folds, seed=rep, insample=False, pb_true=pbt,
                              forward=fwd, batch=bid)
    return _estimate_all(rep, scenario, n, corr, policies, methods, list(regimes), reg, nuis, X, A, Y,
                         batch=bid)


def _estimate_all(rep, scenario, n, corr, policies, methods, regimes, reg, nuis, X, A, Y, batch=None):
    rows = []
    for pol in policies:
        pi = pol(X)
        for rg in regimes:
            ms, ps, cf = REGIMES[rg]
            mhat = nuis[("m", ms, cf)]
            pbhat = nuis[("pb", ps, cf if not isinstance(cf, str) else True)]
            ii = np.arange(len(A))
            od = overlap_diagnostic(pi[ii, A] / pbhat[ii, A])
            odiag = dict(ess_frac=od["ess_frac"], alpha_hat=od["alpha_hat"], max_w=od["max_w"],
                         overlap_cat=od["category"])
            for meth in methods:
                f, kw = reg[meth]
                diag = {}
                if f in (robust_dr, rm_dr, vr_dr):
                    v, se, diag = f(pi, A, Y, mhat, pbhat, return_diag=True, batch=batch, **kw)
                else:
                    v, se = f(pi, A, Y, mhat, pbhat, batch=batch, **kw)
                diag.update(odiag)
                rows.append(dict(rep=rep, scenario=scenario, n=n, corr=corr, policy=pol.name,
                                 regime=rg, method=meth, est=v, se=se, **diag))
    return rows


def run_experiment(tag, scenarios, overlaps=None, policies=None, ns=(1000,), corrs=("indep",),
                   methods=None, regimes=("A",), n_rep=200, base_seed=2026, n_jobs=-1,
                   registry=None, overwrite=False):
    """Run a grid and save long-format raw results to results/raw/<tag>.csv.gz.

    Either `overlaps` (names in OVERLAP_REGIMES) or explicit `policies` is used.
    Scenario x overlap pairs may be restricted by passing scenarios as list of
    (scenario, overlap) tuples.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{tag}.csv.gz"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; results are never overwritten (use a new tag)")
    methods = methods or list((registry or SIM_ESTIMATORS).keys())
    jobs = []
    for corr in corrs:
        for n in ns:
            if policies is not None:
                for sc in scenarios:
                    jobs.append((sc, n, corr, list(policies)))
            else:
                for item in scenarios:
                    if isinstance(item, tuple):
                        sc, ov = item
                        jobs.append((sc, n, corr, [OVERLAP_REGIMES[ov]]))
                    else:
                        jobs.append((item, n, corr, [OVERLAP_REGIMES[o] for o in overlaps]))
    t0 = time.time()
    all_rows = []
    for sc, n, corr, pols in jobs:
        truth = get_truth(pols, corr)
        res = Parallel(n_jobs=n_jobs)(
            delayed(run_replication)(r, sc, n, corr, pols, methods, list(regimes), base_seed, registry)
            for r in range(n_rep))
        rows = [x for rr in res for x in rr]
        for x in rows:
            x["truth"] = truth[x["policy"]]["value"]
        all_rows.extend(rows)
        print(f"[{tag}] {sc:10s} n={n} corr={corr} policies={len(pols)} done "
              f"({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(path, index=False, compression="gzip")
    meta = dict(tag=tag, scenarios=[str(s) for s in scenarios], overlaps=overlaps, ns=list(ns),
                corrs=list(corrs), methods=methods, regimes=list(regimes), n_rep=n_rep,
                base_seed=base_seed, runtime_s=time.time() - t0,
                policies=[p.__dict__ for p in (policies or [])])
    (RAW / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return df
