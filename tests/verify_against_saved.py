"""Re-run a few replications with the repository code and compare them with the saved replication files.
"""
import sys

import numpy as np
import pandas as pd

TOL = 1e-9


def _cmp(new, old, keys, label):
    m = new.merge(old, on=keys, suffixes=("_new", "_old"))
    assert len(m) == len(new), f"{label}: {len(new) - len(m)} rows not found in the saved results"
    err = np.nanmax(np.abs(m.est_new - m.est_old))
    err_se = np.nanmax(np.abs(m.se_new - m.se_old))
    print(f"{label:45s} rows={len(m):5d}  max|est diff|={err:.2e}  max|se diff|={err_se:.2e}")
    assert err < TOL and err_se < TOL, label


def sim():
    from src.policies import OVERLAP_REGIMES, candidate_policies_v3
    from src.registry import SIM_ESTIMATORS
    from src.simulation import RAW, run_replication
    from experiments.final import ADAPT, CONTAM, MAIN, SEL
    from experiments.threshold_rates import METHODS as RATE
    cases = [  # (saved tag, scenario, n, corr, policies, methods, regimes, reps, base_seed)
        ("grid_final_n500", "t3", 500, "indep", [OVERLAP_REGIMES[o] for o in ("good", "moderate", "poor")], MAIN,
         ["A-huber", "B-strong", "C-huber", "D-strong"], [0, 7], 4040),
        ("grid_final_n1000", "gross_pos", 1000, "indep", [OVERLAP_REGIMES["poor"]], MAIN,
         ["A-huber", "B-strong", "C-huber", "D-strong"], [3], 4040),
        ("select_final", "contam05", 1000, "indep", candidate_policies_v3(20), SEL, ["A-huber", "C-huber"], [0], 4040),
        ("adapt_final_n1000", "gross_adapt", 1000, "indep", [OVERLAP_REGIMES["poor"]], ADAPT,
         ["A-huber-oracle", "C-huber-oracle", "A-fwd-oracle", "C-fwd-oracle", "A-fwd0-oracle", "C-fwd0-oracle",
          "C-huber"], [0], 4040),
        ("contam_final_n1000", "gross_pos", 1000, "indep", [OVERLAP_REGIMES["poor"]], CONTAM,
         ["A-huber-oracle", "C-huber-oracle"], [1], 4040),
        ("hlog_final", "gross_hlog", 500, "indep", [OVERLAP_REGIMES[o] for o in ("moderate", "poor")], MAIN,
         ["A-h", "B-h", "C-h", "D-h"], [0], 4040),
        ("ar1_final", "contam05", 1000, "ar1", [OVERLAP_REGIMES[o] for o in ("moderate", "poor")], MAIN,
         ["A-huber", "C-huber"], [0], 4040),
        ("rates_r3_n5000", "gauss", 5000, "indep", [OVERLAP_REGIMES["poor"]], RATE,
         ["C-huber-oracle", "A-huber-oracle"], [0], 3030),
    ]
    for tag, sc, n, corr, pols, meth, regs, reps, seed in cases:
        old = pd.read_csv(RAW / f"{tag}.csv.gz")
        old = old[(old["scenario"] == sc) & (old["n"] == n) & (old["corr"] == corr) & old["rep"].isin(reps)]
        rows = []
        for r in reps:
            rows += run_replication(r, sc, n, corr, pols, meth, regs, seed, SIM_ESTIMATORS)
        _cmp(pd.DataFrame(rows), old, ["rep", "scenario", "n", "corr", "policy", "regime", "method"], f"sim {tag} {sc}")


def aslib():
    from real_data.aslib.paths import RAW
    from real_data.aslib.run_aslib import build_context, one_rep, COLS
    for scen, ov, n, rep in [("SAT03-16_INDU", "poor", 1000, 3), ("ASP-POTASSCO", "moderate", 250, 0)]:
        ctx = build_context(scen)
        new = pd.DataFrame(one_rep(ctx, ov, n, rep), columns=COLS)
        old = pd.read_pickle(RAW / scen / f"cell_{ov}_{n}.pkl.gz")
        old = old[old.rep == rep]
        _cmp(new, old, ["pop", "reward", "rmodel", "policy", "method", "rep"], f"aslib {scen} {ov} n={n}")


def kuairec():
    from real_data.kuairec.paths import RAW
    from real_data.kuairec.run_kuairec import COLS, kuairec_context, kuairec_rep
    ctx = kuairec_context()
    for ov, n, rep in [("poor", 1000, 5), ("good", 250, 0)]:
        new = pd.DataFrame(kuairec_rep(ctx, ov, n, rep), columns=COLS)
        old = pd.read_pickle(RAW / f"cell_{ov}_{n}.pkl.gz")
        old = old[old.rep == rep]
        _cmp(new, old, ["variant", "policy", "method", "rep"], f"kuairec {ov} n={n}")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["sim", "aslib", "kuairec"]
    for p in parts:
        {"sim": sim, "aslib": aslib, "kuairec": kuairec}[p]()
    print("all checks passed")
