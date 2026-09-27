#!/bin/sh
# Final simulation study (base seed 4040).  About 2.5 h on 2 cores with the included truth cache
# (results/simulation/truth_cache.json); each stage writes results/simulation/raw/.
# run_experiment never overwrites an existing tag: delete or move results/simulation/raw/<tag>.csv.gz to rerun.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/simulation/raw
python -m experiments.final grid         > results/simulation/raw/grid_final.log 2>&1
python -m experiments.final select       > results/simulation/raw/select_final.log 2>&1
python -m experiments.final adapt        > results/simulation/raw/adapt_final.log 2>&1
python -m experiments.forward_check      > results/simulation/raw/fwdcheck_final.log 2>&1
python -m experiments.final contam       > results/simulation/raw/contam_final.log 2>&1
python -m experiments.final hlog         > results/simulation/raw/hlog_final.log 2>&1
python -m experiments.final grid10k      > results/simulation/raw/grid10k_final.log 2>&1
python -m experiments.final ar1          > results/simulation/raw/ar1_final.log 2>&1
python -m experiments.adaptive_diagnosis > results/simulation/raw/adaptdiag_r3.log 2>&1   # Figure 4 right panel
python -m experiments.threshold_rates    > results/simulation/raw/rates_r3.log 2>&1       # threshold-rate ablation
echo done
