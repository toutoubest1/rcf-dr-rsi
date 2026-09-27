#!/bin/sh
# KuaiRec application (Sections 6 and S11). Requires data/kuairec/reduced (see data/README.md).
# About 20 min on 2 cores.
set -e
cd "$(dirname "$0")/.."
python -m real_data.kuairec.audit_kuairec                   # data audit
python -m real_data.kuairec.design_summary                  # kappa calibration, exact policy values
python -m real_data.kuairec.run_kuairec 500 2
python -m real_data.kuairec.analyze_kuairec                 # summaries
python -m real_data.kuairec.make_main_figure                # Figure 7
python -m real_data.kuairec.make_tables                     # Table 6, Tables S34-S42, Figures S4-S6
