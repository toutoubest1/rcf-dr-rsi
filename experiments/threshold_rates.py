"""Threshold-rate ablation (Supplement S9, "Threshold rate"): tau_n ~ n^(1/4) versus n^(1/3) at large n.

Poor overlap, oracle (known) logging probabilities, regimes C-huber-oracle (reward model wrong) and
A-huber-oracle, noise laws gauss / t3 / contam05 / gross_pos, n = 5000, 10000, 20000 with 200 / 150 / 100
replications, base seed 3030 (these are the runs rates_r3_n* of the development log).  Both thresholds are
anchored so that tau = 2.8 MAD at n = 1000.  DR and RCF-DR-os are included as references.

usage: python -m experiments.threshold_rates           # simulation (about 5 min on 2 cores)
       python -m experiments.threshold_rates summary   # results/simulation/tables/threshold_rates_value_metrics.csv
"""
import sys

import pandas as pd

from experiments.common import RAW, TAB
from src.metrics import value_metrics
from src.registry import SIM_ESTIMATORS
from src.simulation import run_experiment

METHODS = ["DR", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-n13"]
RUNS = [(5000, 200), (10000, 150), (20000, 100)]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        d = pd.concat([pd.read_csv(RAW / f"rates_r3_n{n}.csv.gz") for n, _ in RUNS])
        d = d[d.method.isin(METHODS)]
        m = value_metrics(d, ["scenario", "regime", "n", "method"])
        m.to_csv(TAB / "threshold_rates_value_metrics.csv", index=False)
        print(m.round(4).to_string())
    else:
        for n, R in RUNS:
            run_experiment(f"rates_r3_n{n}", ["gauss", "t3", "contam05", "gross_pos"], overlaps=["poor"],
                           ns=(n,), n_rep=R, regimes=["C-huber-oracle", "A-huber-oracle"],
                           methods=METHODS, registry=SIM_ESTIMATORS, base_seed=3030)
