#!/bin/sh
# Simulation summaries, Figures 1-5 and S1, and all simulation LaTeX tables (main text and supplement).
set -e
cd "$(dirname "$0")/.."
python -m experiments.analyze_final grid select adapt contam extra
python -m experiments.make_paper_tables
python -m experiments.lemma_check
python -m experiments.threshold_rates summary
