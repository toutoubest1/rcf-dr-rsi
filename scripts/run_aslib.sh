#!/bin/sh
# ASlib application (Sections 5 and S10). Requires data/aslib/aslib_data (see data/README.md).
# About 55 min per scenario on 2 cores.
set -e
cd "$(dirname "$0")/.."
python -m real_data.aslib.audit_aslib                       # scenario audit and selection
python -m real_data.aslib.design_summary                    # overlap calibration, exact policy values
python -m real_data.aslib.run_aslib SAT03-16_INDU 500 2
python -m real_data.aslib.run_aslib ASP-POTASSCO 500 2
python -m real_data.aslib.analyze_aslib                     # summaries and figures
python -m real_data.aslib.analyze_aslib md
python -m real_data.aslib.make_tables                       # LaTeX tables
