"""LaTeX tables of the ASlib application from the result summaries (no estimation)."""
import sys
from pathlib import Path

import pandas as pd

from real_data.aslib.paths import ROOT, TAB

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
SCEN = [("SAT03-16_INDU", "SAT03-16\\_INDU", "logPAR10 [primary]"), ("ASP-POTASSCO", "ASP-POTASSCO", "logPAR1 [primary]")]
OVL = {"good": "Uniform", "moderate": "Moderate", "poor": "Poor"}
MS = ["DM", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]


def main():
    for sub in ("tables", "supp_tables"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(TAB / "aslib_target_summary.csv")
    SEL = pd.read_csv(TAB / "aslib_selection_summary.csv")
    C = pd.read_csv(TAB / "aslib_criteria.csv")

    L = [r"""\begin{table}[t]
\centering\spacingset{1}
\caption{ASlib target-policy results for the primary reward (negative log runtime under each scenario's timeout convention), 500 replications per cell, exact policy values. RMSE and 95\% Wald coverage; complete results in Supplement~S10.}
\label{tab:aslib}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllcccccccc}
\toprule
 & & & \multicolumn{6}{c}{RMSE} & \multicolumn{2}{c}{Coverage} \\
\cmidrule(lr){4-9}\cmidrule(lr){10-11}
Scenario & Overlap & $n$ & """ + " & ".join(MS) + r""" & RCF-DR-os & RCF-DR-VR-SN \\
\midrule"""]
    for k, (scen, lab, var) in enumerate(SCEN):
        x = S[(S.scenario == scen) & (S.variant == var)]
        r = x.pivot_table(index=["overlap", "n"], columns="method", values="rmse")
        c = x.pivot_table(index=["overlap", "n"], columns="method", values="coverage")
        first = True
        for ov in ["moderate", "poor"]:
            for i, n in enumerate([250, 1000, 4000]):
                key = (ov, n)
                L.append(f"{lab if first else ''} & {OVL[ov] if i == 0 else ''} & {n} & "
                         + " & ".join(f"{r.loc[key, m]:.3f}" for m in MS) + " & "
                         + " & ".join(f"{c.loc[key, m]:.2f}" for m in ["RCF-DR-os", "RCF-DR-VR-SN"]) + r" \\")
                first = False
        if k == 0:
            L.append(r"\midrule")
    L.append(r"""\bottomrule
\end{tabular}}
\end{table}""")
    (OUT / "tables" / "tab_aslib.tex").write_text("\n".join(L) + "\n")

    L = [r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{ASlib policy selection among nine candidate solver-selection policies (primary reward, 500 replications): mean regret $V(\pi^\ast)-V(\hat\pi)$, with the probability of selecting the true best policy in parentheses.}
\label{tab:S-aslib-selection}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{lll""" + "c" * len(MS) + r"""}
\toprule
Scenario & Overlap & $n$ & """ + " & ".join(MS) + r""" \\
\midrule"""]
    for k, (scen, lab, var) in enumerate(SCEN):
        y = SEL[(SEL.scenario == scen) & (SEL.variant == var)]
        rg = y.pivot_table(index=["overlap", "n"], columns="method", values="regret")
        pb = y.pivot_table(index=["overlap", "n"], columns="method", values="p_best")
        first = True
        for ov in ["good", "moderate", "poor"]:
            for i, n in enumerate([250, 500, 1000, 2000, 4000]):
                key = (ov, n)
                L.append(f"{lab if first else ''} & {OVL[ov] if i == 0 else ''} & {n} & "
                         + " & ".join(f"{rg.loc[key, m]:.3f} ({pb.loc[key, m]:.2f})" for m in MS) + r" \\")
                first = False
        if k == 0:
            L.append(r"\midrule")
    L.append(r"""\bottomrule
\end{tabular}}
\end{table}""")
    (OUT / "supp_tables" / "aslib_selection.tex").write_text("\n".join(L) + "\n")

    L = [r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{ASlib sensitivity analyses: ranges over the ten moderate and poor-overlap cells of the RMSE ratios of each comparison estimator to RCF-DR-VR-SN (values above one favour RCF-DR-VR-SN), the number of cells in which RCF-DR-VR-SN has lower RMSE than both DR and DR-clip, and the RCF-DR-VR-SN coverage path at moderate and poor overlap ($n=250,\dots,4000$).}
\label{tab:S-aslib-sensitivity}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{llcccccl}
\toprule
Scenario & Variant & DR & DR-clip & RCF-DR-os & SNIPW & Cells & RCF-DR-VR-SN coverage (moderate; poor) \\
\midrule"""]
    for k, (scen, lab, _) in enumerate(SCEN):
        c = C[(C.scenario == scen) & C.overlap.isin(["moderate", "poor"])]
        for v in c.variant.unique():
            cc = c[c.variant == v]

            def rng(col):
                lo = min(float(s.split("-")[0]) for s in cc[col])
                hi = max(float(s.split("-")[1]) for s in cc[col])
                return f"{lo:.2f}--{hi:.2f}"
            cells = sum(int(s.split("/")[0]) for s in cc.C2a_cells)
            cov = "; ".join(s.replace(" / ", ", ") for s in cc.cov_VRSN)
            L.append(f"{lab} & {v.replace('[primary]', '(primary)').replace('_', chr(92) + '_')} & {rng('rmse_ratio_DR')} & "
                     f"{rng('rmse_ratio_DRclip')} & {rng('rmse_ratio_os')} & {rng('rmse_ratio_SNIPW')} & {cells}/10 & {cov} \\\\")
        if k == 0:
            L.append(r"\midrule")
    L.append(r"""\bottomrule
\end{tabular}}
\end{table}""")
    (OUT / "supp_tables" / "aslib_sensitivity.tex").write_text("\n".join(L) + "\n")
    print("wrote tables/tab_aslib.tex, supp_tables/aslib_selection.tex, supp_tables/aslib_sensitivity.tex")


if __name__ == "__main__":
    main()
