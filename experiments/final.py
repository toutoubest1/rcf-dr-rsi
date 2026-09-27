"""Final simulation study of the paper (all stages; base seed 4040).

usage: python -m experiments.final <stage> [version]
Stages (tags written to results/simulation/raw/<stage>_<version>...csv.gz, version defaults to "final"):
  grid     IID grid: 6 noise laws x 3 overlap levels x regimes A/B/C/D, n in {500, 1000, 2000, 5000}, 500 reps
  grid10k  n = 10 000 validation runs: poor overlap, regimes A/C, 200 reps
  hlog     strong propensity misspecification (logger exploits the nonlinear reward terms), 300 reps
  ar1      AR(1) covariates (rho = 0.5), 300 reps
  select   policy selection among M = 20 candidates (the first 10 form the M = 10 set), n = 1000, 500 reps
  adapt    batched adaptive logging; K-fold, forward and forward-strict fitting; n in {1000, 2000, 5000}, 300 reps
  contam   one-sided contamination bias versus n (n up to 20 000), 200 reps
Estimators and tuning constants: src/registry.py (SIM_ESTIMATORS).
"""
import sys

from src.policies import candidate_policies_v3
from src.registry import SIM_ESTIMATORS as FINAL
from src.simulation import run_experiment

MAIN = ["DM", "DR", "DR-clip", "RCF-DR", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RM-huber"]
SEL = ["DM", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
ADAPT = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RCF-DR-VR-tail"]
CONTAM = ["RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-n13", "RCF-DR-VR-SN"]
NOISE = ["gauss", "t3", "t2", "contam05", "gross", "gross_pos"]
SEED = 4040

if __name__ == "__main__":
    stage = sys.argv[1]
    v = sys.argv[2] if len(sys.argv) > 2 else "final"
    kw = dict(registry=FINAL, base_seed=SEED)
    if stage == "grid":
        for n in (500, 1000, 2000, 5000):
            run_experiment(f"grid_{v}_n{n}", NOISE, overlaps=["good", "moderate", "poor"], ns=(n,), n_rep=500,
                           regimes=["A-huber", "B-strong", "C-huber", "D-strong"], methods=MAIN, **kw)
    elif stage == "grid10k":
        run_experiment(f"grid10k_{v}", ["gauss", "t3", "contam05", "gross_pos"], overlaps=["poor"], ns=(10000,),
                       n_rep=200, regimes=["A-huber", "C-huber"], methods=MAIN, **kw)
    elif stage == "hlog":
        run_experiment(f"hlog_{v}", ["gauss_hlog", "t3_hlog", "gross_hlog"], overlaps=["moderate", "poor"],
                       ns=(500, 1000, 2000, 5000), n_rep=300, regimes=["A-h", "B-h", "C-h", "D-h"],
                       methods=MAIN, **kw)
    elif stage == "ar1":
        run_experiment(f"ar1_{v}", ["gauss", "t3", "contam05"], overlaps=["moderate", "poor"], ns=(1000, 2000),
                       corrs=("ar1",), n_rep=300, regimes=["A-huber", "C-huber"], methods=MAIN, **kw)
    elif stage == "select":
        run_experiment(f"select_{v}", ["gauss", "t3", "contam05", "gross"], policies=candidate_policies_v3(20),
                       n_rep=500, regimes=["A-huber", "C-huber"], methods=SEL, **kw)
    elif stage == "adapt":
        for n in (1000, 2000, 5000):
            run_experiment(f"adapt_{v}_n{n}", ["gauss_adapt", "t3_adapt", "gross_adapt"], overlaps=["poor"],
                           ns=(n,), n_rep=300,
                           regimes=["A-huber-oracle", "C-huber-oracle", "A-fwd-oracle", "C-fwd-oracle",
                                    "A-fwd0-oracle", "C-fwd0-oracle", "C-huber"], methods=ADAPT, **kw)
    elif stage == "contam":
        for n in (1000, 2000, 5000, 10000, 20000):
            run_experiment(f"contam_{v}_n{n}", ["gross_pos"], overlaps=["poor"], ns=(n,), n_rep=200,
                           regimes=["A-huber-oracle", "C-huber-oracle"], methods=CONTAM, **kw)
    else:
        raise SystemExit(__doc__)
