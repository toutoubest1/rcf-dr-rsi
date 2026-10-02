# rcf-dr-rsi

The main estimator is **RCF-DR-VR-SN**. It is a cross-fitted doubly robust (DR) off-policy value estimator with:
- a Huber residual threshold τ_n = 0.5·MAD(r)·n^{1/4};
- a weight cap c_n = w̄·√(n / log n);
- one-step rescaling;
- self-normalisation of the correction term.


## Installation

```
git clone https://github.com/<user>/rcf-dr-rsi.git
cd rcf-dr-rsi
python -m venv .venv && . .venv/bin/activate      # Python 3.11 was used
pip install -r requirements.txt                    # or: conda env create -f environment.yml
pytest tests/                                      # 15 unit tests, < 10 s
```

All commands are run from the repository root with `python -m ...`. No installation step or `PYTHONPATH` is
needed.

## Repository layout

```
src/                     shared methodology
  estimators.py          DM, IPW, SNIPW, DR / DR-clip / RCF-DR / RCF-DR-os (robust_dr), RM-Huber (rm_dr)
  vr.py                  RCF-DR-VR, RCF-DR-VR-SN (sn="global"), RCF-DR-VR-tail (tail="exact")
  tuning.py              threshold and cap rules (MAD threshold, 99% quantile cap, Ionides cap)
  robust_mean.py         Huber location for the RM-Huber ablation
  registry.py            the named estimator configurations and all tuning constants used in the paper
  diagnostics.py         overlap diagnostic: ESS/n, maximum weight and its share, Hill tail index, warning rule
  dgp.py, policies.py    simulation data-generating process, logging policies, target and candidate policies
  nuisance.py            reward and propensity models, K-fold cross-fitting, forward / forward-strict fitting
  simulation.py          Monte Carlo driver, scenarios, nuisance regimes, truth cache
  metrics.py             value and policy-selection metrics
  realdata.py            cross-fitted reward model and estimator calls shared by ASlib and KuaiRec
experiments/             simulation study (Sections 4 and S2-S9)
  final.py               all final stages (grid, grid10k, hlog, ar1, select, adapt, contam); seed 4040
  forward_check.py       martingale check of forward fitting (Table S27)
  adaptive_diagnosis.py  batch decomposition behind Figure 4 (right panel)
  threshold_rates.py     threshold-rate ablation (Supplement S9)
  lemma_check.py         numerical check of the Huber-remainder lemma (Table S1)
  analyze_final.py       Figures 1-5, S1 and simulation summary tables
  make_paper_tables.py   LaTeX tables of the simulation study (main text and supplement)
real_data/aslib/         ASlib application (Sections 5 and S10)
real_data/kuairec/       KuaiRec application (Sections 6 and S11)
scripts/                 shell scripts with the exact commands of the final runs
data/                    external data (not included; see data/README.md)
tests/                   unit tests and a check against the saved replications
```


## External data

Neither dataset is redistributed here. Download instructions are in `data/README.md`.

- **ASlib**: `coseal/aslib_data`, commit `551b22be`, scenarios SAT03-16_INDU and ASP-POTASSCO.
- **KuaiRec 2.0**: Zenodo record 18164998 (`KuaiRec.zip`, CC BY-SA 4.0).


## Citation and licence

Please cite the paper; the citation will be added on publication. The data are subject to their own licences:
the ASlib scenario licences, and CC BY-SA 4.0 for KuaiRec.
