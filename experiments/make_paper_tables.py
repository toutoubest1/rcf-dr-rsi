"""LaTeX tables for the manuscript (main text and supplement), built only from the final result summaries
written by experiments/analyze_final.py.  No simulation is run here.

usage: python -m experiments.make_paper_tables [out_dir]      (default: repository root)
Writes <out_dir>/tables/*.tex (main text) and <out_dir>/supp_tables/*.tex (supplement).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "simulation" / "tables"

LAB = {"DM": "DM", "DR": "DR", "DR-clip": "DR-clip", "RCF-DR": "RCF-DR (no rescale)", "RCF-DR-os": "RCF-DR-os",
       "RCF-DR-VR": "RCF-DR-VR", "RCF-DR-VR-SN": "RCF-DR-VR-SN", "RM-huber": "RM-Huber",
       "RCF-DR-VR-tail": "RCF-DR-VR-tail", "RCF-DR-VR-n13": "RCF-DR-VR ($n^{1/3}$)"}
NOISE = {"gauss": "Gaussian", "t3": "$t_3$", "t2": "$t_2$", "contam05": "Contam.\\ 5\\%", "gross": "Gross (sym.)",
         "gross_pos": "Gross (one-sided)", "gauss_adapt": "Gaussian", "t3_adapt": "$t_3$", "gross_adapt": "Gross (sym.)",
         "gauss_hlog": "Gaussian", "t3_hlog": "$t_3$", "gross_hlog": "Gross (sym.)"}
REG = {"A-huber": "A", "B-strong": "B", "C-huber": "C", "D-strong": "D"}
MAIN8 = ["DM", "DR", "DR-clip", "RCF-DR", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RM-huber"]
KEY5 = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]


def f3(x, sign=False):
    if not np.isfinite(x):
        return "--"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    if s in ("+0.000", "-0.000"):
        s = "0.000"
    return s.replace("-", "$-$")


def f2(x):
    return f"{x:.2f}" if np.isfinite(x) else "--"


def cell(r):
    return f"{f3(r.bias, True)} / {f3(r.rmse)} / {f2(r.cover)}"


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == "supp_tables" and "\\begin{tabular}" in body and "resizebox" not in body:
        body = body.replace("\\begin{tabular}", "\\adjustbox{max width=\\textwidth}{%\n\\begin{tabular}", 1)
        body = body.replace("\\end{tabular}", "\\end{tabular}}", 1)
    path.write_text(body)
    print("wrote", path)


# ------------------------------------------------------------------ main tables
def t1_representative(m, out):
    rows = []
    for rg in ["A", "B", "C", "D"]:
        first_rg = True
        for sc in ["gauss", "t3", "gross"]:
            for n in (1000, 5000):
                mm = m[(m.reg == rg) & (m.scenario == sc) & (m.n == n) & (m.policy == "poor")].set_index("method")
                cells = [cell(mm.loc[k]) for k in KEY5]
                lead = rg if first_rg else ""
                first_rg = False
                nl = NOISE[sc] if n == 1000 else ""
                rows.append(f"{lead} & {nl} & {n} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if rg != "D" else "")
    head = " & ".join(["Reg.", "Noise", "$n$"] + [LAB[k] for k in KEY5])
    body = (r"\begin{table}[t]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{Representative value-estimation results, poor overlap. Each cell reports bias / RMSE / "
            r"empirical coverage of the nominal 95\% Wald interval over 500 Monte Carlo replications "
            r"(Monte Carlo standard error of coverage $\approx 0.01$). Regimes: A both nuisances correct; "
            r"B propensity misspecified; C reward model misspecified; D both misspecified. DM, RCF-DR without "
            r"rescaling, RM-Huber, the other noise laws and the other overlap levels are reported in Section~S3 of the Supplementary Material.}"
            "\n" r"\label{tab:representative}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\resizebox{\textwidth}{!}{%" "\n"
            r"\begin{tabular}{lll" + "c" * len(KEY5) + "}\n" r"\toprule" "\n" + head + r" \\" "\n" r"\midrule" "\n"
            + "\n".join(r for r in rows if r) + "\n" r"\bottomrule" "\n" r"\end{tabular}}" "\n" r"\end{table}" "\n")
    write(out / "tables" / "tab_representative.tex", body)


def selection_table(s, M, regime, label, caption, out_path):
    meths = ["DM", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
    cols = [("p_best", "P(best)", f2), ("best_in_est_top3", "Top-3", f2), ("regret", "Mean regret", f3),
            ("regret_median", "Med.\\ regret", f3), ("spearman", "Spearman", f2),
            ("winners_curse", "Winner's curse", lambda x: f3(x, True)), ("sel_cover", "Sel.\\ cover", f2)]
    rows = []
    for sc in ["gauss", "t3", "contam05", "gross"]:
        t = s[(s.M == M) & (s.regime == regime) & (s.scenario == sc)].set_index("method")
        for i, mt in enumerate(meths):
            r = t.loc[mt]
            lead = NOISE[sc] if i == 0 else ""
            rows.append(f"{lead} & {LAB[mt]} & " + " & ".join(fn(r[c]) for c, _, fn in cols) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    head = "Noise & Estimator & " + " & ".join(h for _, h, _ in cols)
    body = (r"\begin{table}[t]" "\n" r"\centering\spacingset{1}" "\n" rf"\caption{{{caption}}}" "\n" rf"\label{{{label}}}" "\n"
            r"\footnotesize" "\n" r"\resizebox{\textwidth}{!}{%" "\n" r"\begin{tabular}{llccccccc}" "\n" r"\toprule" "\n" + head + r" \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}}" "\n" r"\end{table}" "\n")
    write(out_path, body)


def adaptive_table(a, regime, label, caption, out_path, placement="t"):
    meths = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RCF-DR-VR-tail"]
    rows = []
    for sc in ["gauss_adapt", "t3_adapt", "gross_adapt"]:
        for i, mt in enumerate(meths):
            cells = []
            for n in (1000, 2000, 5000):
                r = a[(a.regime == regime) & (a.scenario == sc) & (a.n == n) & (a.method == mt)].iloc[0]
                cells += [f3(r.bias, True), f3(r.rmse), f2(r.cover)]
            lead = NOISE[sc] if i == 0 else ""
            rows.append(f"{lead} & {LAB[mt]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = (rf"\begin{{table}}[{placement}]" "\n" r"\centering\spacingset{1}" "\n" rf"\caption{{{caption}}}" "\n"
            rf"\label{{{label}}}" "\n" r"\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n"
            r"\begin{tabular}{ll" + "rrc" * 3 + "}\n" r"\toprule" "\n"
            r" & & \multicolumn{3}{c}{$n=1000$} & \multicolumn{3}{c}{$n=2000$} & \multicolumn{3}{c}{$n=5000$} \\"
            "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}" "\n"
            r"Noise & Estimator & Bias & RMSE & Cov. & Bias & RMSE & Cov. & Bias & RMSE & Cov. \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out_path, body)


# ------------------------------------------------------------------ supplement tables
def full_grid_tables(m, out):
    """One longtable per regime and metric group: rows (noise, overlap, n), columns = 8 estimators."""
    for rg in ["A", "B", "C", "D"]:
        for metric, title, fn in [("bias", "bias", lambda x: f3(x, True)), ("rmse", "RMSE", f3),
                                  ("cover", "coverage of the nominal 95\\% Wald interval", f2),
                                  ("ratio_se", "ratio of average estimated SE to Monte Carlo SD", f2)]:
            rows = []
            for sc in ["gauss", "t3", "t2", "contam05", "gross", "gross_pos"]:
                for ov in ["good", "moderate", "poor"]:
                    for n in (500, 1000, 2000, 5000):
                        mm = m[(m.reg == rg) & (m.scenario == sc) & (m.policy == ov) & (m.n == n)].set_index("method")
                        vals = [fn(mm.loc[k, metric]) for k in MAIN8]
                        rows.append(f"{NOISE[sc]} & {ov} & {n} & " + " & ".join(vals) + r" \\")
                rows.append(r"\midrule")
            rows = rows[:-1]
            head = "Noise & Overlap & $n$ & " + " & ".join(LAB[k].replace(" (no rescale)", "$^\\dagger$") for k in MAIN8)
            body = (r"{\spacingset{1}\scriptsize\setlength{\tabcolsep}{3pt}" "\n" r"\begin{longtable}{lll" + "r" * 8 + "}\n"
                    rf"\caption{{Regime {rg}: {title} (500 replications). $^\dagger$RCF-DR without the $\hat p$ rescaling.}}"
                    rf"\label{{tab:S-{rg}-{metric}}}\\" "\n" r"\toprule" "\n" + head + r" \\ \midrule" "\n"
                    r"\endfirsthead" "\n" r"\toprule" "\n" + head + r" \\ \midrule" "\n" r"\endhead" "\n"
                    r"\bottomrule" "\n" r"\endfoot" "\n" + "\n".join(rows) + "\n" r"\end{longtable}}" "\n")
            write(out / "supp_tables" / f"grid_{rg}_{metric}.tex", body)


def relative_efficiency(m, out):
    mm = m[m.method.isin(MAIN8)].copy()
    mm["best"] = mm.groupby(["scenario", "policy", "regime", "n"]).rmse.transform("min")
    mm["ratio"] = mm.rmse / mm.best
    rows = []
    sym = mm[mm.scenario != "gross_pos"]
    for rg in ["A", "B", "C", "D"]:
        for k in MAIN8:
            a = mm[(mm.reg == rg) & (mm.method == k)]
            b = sym[(sym.reg == rg) & (sym.method == k)]
            gm = lambda x: float(np.exp(np.log(x).mean()))
            rows.append(f"{rg if k == 'DM' else ''} & {LAB[k]} & {f2(gm(b.ratio))} & {f2(b.ratio.max())} & "
                        f"{f2(b.cover.mean())} & {f2(b.cover.min())} & {f2(gm(a.ratio))} & {f2(a.cover.min())}" + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{Relative efficiency over the full IID grid (6 noise laws $\times$ 3 overlaps $\times$ 4 sample "
            r"sizes). Ratio = RMSE / smallest RMSE among the eight estimators in the same cell; geometric mean and maximum "
            r"over cells. ``Symmetric'' excludes the one-sided gross-contamination law.}" "\n"
            r"\label{tab:S-releff}" "\n" r"\footnotesize" "\n" r"\begin{tabular}{llcccccc}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{4}{c}{Symmetric noise} & \multicolumn{2}{c}{All noise laws} \\"
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}" "\n"
            r"Reg. & Estimator & Geo.\ ratio & Max ratio & Mean cov. & Min cov. & Geo.\ ratio & Min cov. \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "releff.tex", body)


def overlap_table(m, out):
    ov = m[m.method == "DR"].groupby(["policy", "regime", "n"])[["ess_med", "alpha_med"]].median().reset_index()
    rows = []
    for pol in ["good", "moderate", "poor"]:
        for rg, lab in [("A-huber", "A/C (correct propensity model)"), ("B-strong", "B/D (misspecified propensity model)")]:
            v = ov[(ov.policy == pol) & (ov.regime == rg)].set_index("n")
            rows.append(f"{pol} & {lab} & " + " & ".join(f"{v.loc[n, 'ess_med']:.2f} / {v.loc[n, 'alpha_med']:.2f}"
                                                         for n in (500, 1000, 2000, 5000)) + r" \\")
    cat = pd.read_csv(TAB / "final_diagnostic_categories.csv")
    rows2 = []
    for pol in ["good", "moderate", "poor"]:
        c = cat[cat.policy == pol].set_index("n")
        rows2.append(f"{pol} & " + " & ".join(f"{c.loc[n, 'stable']:.2f} / {c.loc[n, 'borderline']:.2f} / "
                                              f"{c.loc[n, 'non-root-n']:.2f}" for n in (500, 1000, 2000, 5000)) + r" \\")
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{Overlap diagnostics in the IID grid. Top: median ESS$/n$ / median Hill $\hat\alpha$ of the "
            r"weights actually used by the estimator (estimated propensity model). Bottom: share of replications "
            r"(regimes A and C) classified stable / borderline / non-root-$n$.}" "\n" r"\label{tab:S-overlap}" "\n"
            r"\footnotesize" "\n" r"\begin{tabular}{llcccc}" "\n" r"\toprule" "\n"
            r"Overlap & Weights & $n=500$ & $n=1000$ & $n=2000$ & $n=5000$ \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\midrule" "\n"
            r"Overlap & & \multicolumn{4}{c}{stable / borderline / non-root-$n$} \\" "\n" r"\midrule" "\n"
            + "\n".join(r.replace(" & ", " & & ", 1) for r in rows2) + "\n"
            r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "overlap.tex", body)


def diagnostic_alpha_table(out):
    g = pd.read_csv(TAB / "final_diagnostic_by_alpha.csv")
    bins = list(dict.fromkeys(g.abin))
    lab = {b: b.replace("(", "$(").replace("]", "]$").replace("inf", "\\infty") for b in bins}
    rows = []
    for b in bins:
        d = g[(g.abin == b) & (g.method == "DR")].iloc[0]
        s = g[(g.abin == b) & (g.method == "RCF-DR-VR-SN")].iloc[0]
        rows.append(f"{lab[b]} & {int(d['count'])} & {f2(d.cover)} & {f3(d.rmse)} & {f2(s.cover)} & {f3(s.rmse)}" + r" \\")
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{Wald coverage and RMSE binned by the estimated weight tail index $\hat\alpha$ (Hill, "
            r"$k=\lceil 2\sqrt n\rceil$). Pooled over all $n$, overlaps, regimes A and C, and the symmetric noise laws "
            r"Gaussian, $t_3$, contaminated and gross. Count = number of replications per estimator.}" "\n"
            r"\label{tab:S-alpha}" "\n" r"\footnotesize" "\n" r"\begin{tabular}{lrcccc}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{2}{c}{DR} & \multicolumn{2}{c}{RCF-DR-VR-SN} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
            r"$\hat\alpha$ bin & Count & Coverage & RMSE & Coverage & RMSE \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "diag_alpha.tex", body)


def forward_table(a, out):
    meths = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
    fits = [("huber-oracle", "K-fold"), ("fwd-oracle", "forward"), ("fwd0-oracle", "forward-strict")]
    rows = []
    for rw in ["A", "C"]:
        for sc in ["gauss_adapt", "t3_adapt", "gross_adapt"]:
            for mt in meths:
                for j, (suf, fl) in enumerate(fits):
                    cells = []
                    for n in (1000, 2000, 5000):
                        r = a[(a.regime == f"{rw}-{suf}") & (a.scenario == sc) & (a.n == n) & (a.method == mt)].iloc[0]
                        cells += [f3(r.bias, True), f3(r.rmse), f2(r.cover)]
                    lead = f"{rw} & {NOISE[sc]} & {LAB[mt]}" if j == 0 else " & & "
                    rows.append(f"{lead} & {fl} & " + " & ".join(cells) + r" \\")
            rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"{\spacingset{1}\scriptsize\setlength{\tabcolsep}{2.5pt}" "\n" r"\begin{longtable}{llll" + "rrc" * 3 + "}\n"
            r"\caption{Adaptive logging, known logging probabilities: K-fold versus forward (chronological) "
            r"reward-model fitting. Forward-strict sets $\hat m=0$ in the first batch. A = reward model correct, "
            r"C = misspecified. Bias / RMSE / coverage, 300 replications.}\label{tab:S-forward}\\" "\n"
            r"\toprule" "\n"
            r" & & & & \multicolumn{3}{c}{$n=1000$} & \multicolumn{3}{c}{$n=2000$} & \multicolumn{3}{c}{$n=5000$} \\" "\n"
            r"Reg. & Noise & Estimator & Fit & Bias & RMSE & Cov. & Bias & RMSE & Cov. & Bias & RMSE & Cov. \\ \midrule" "\n"
            r"\endfirsthead" "\n" r"\toprule" "\n"
            r"Reg. & Noise & Estimator & Fit & Bias & RMSE & Cov. & Bias & RMSE & Cov. & Bias & RMSE & Cov. \\ \midrule" "\n"
            r"\endhead" "\n" r"\bottomrule" "\n" r"\endfoot" "\n" + "\n".join(rows) + "\n" r"\end{longtable}}" "\n")
    write(out / "supp_tables" / "forward.tex", body)


def estprop_table(a, out):
    meths = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RCF-DR-VR-tail"]
    rows = []
    for sc in ["gauss_adapt", "t3_adapt", "gross_adapt"]:
        for i, mt in enumerate(meths):
            cells = []
            for n in (1000, 2000, 5000):
                r = a[(a.regime == "C-huber") & (a.scenario == sc) & (a.n == n) & (a.method == mt)].iloc[0]
                cells += [f3(r.bias, True), f3(r.rmse), f2(r.cover)]
            rows.append(f"{NOISE[sc] if i == 0 else ''} & {LAB[mt]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{Adaptive logging, regime C, with the logging probabilities \emph{estimated} by a pooled "
            r"multinomial logistic model that ignores the batch structure (compare Table~\ref{tab:adaptive}, which "
            r"uses the recorded probabilities). 300 replications.}" "\n" r"\label{tab:S-adapt-estprop}" "\n"
            r"\footnotesize" "\n" r"\setlength{\tabcolsep}{4pt}" "\n" r"\begin{tabular}{ll" + "rrc" * 3 + "}\n"
            r"\toprule" "\n"
            r" & & \multicolumn{3}{c}{$n=1000$} & \multicolumn{3}{c}{$n=2000$} & \multicolumn{3}{c}{$n=5000$} \\" "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}" "\n"
            r"Noise & Estimator & Bias & RMSE & Cov. & Bias & RMSE & Cov. & Bias & RMSE & Cov. \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "adapt_estprop.tex", body)


def fwdcheck_table(out):
    f = pd.read_csv(TAB / "fwdcheck_final.csv")
    rows = []
    for sc in ["gauss_adapt", "t3_adapt"]:
        for spec, sl in [("correct_huber", "correct"), ("mis_huber", "misspecified")]:
            for sch in ["kfold", "forward", "forward-strict"]:
                v = f[(f.scenario == sc) & (f.spec == spec) & (f.scheme == sch)].sort_values("batch")
                dr = " & ".join(f"{f3(x, True)}" for x in v.dr_mean)
                cl = " & ".join(f"{f3(x, True)}" for x in v.clipped_mean)
                rows.append(f"{NOISE[sc]} & {sl} & {sch} & {dr} & {cl}" + r" \\")
            rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{Martingale check under batched adaptive logging ($n=2000$, 400 replications, known logging "
            r"probabilities). Left: per-batch Monte Carlo mean of the uncapped DR score minus $V$ (Lemma~\ref{lem:mds} predicts 0; "
            r"Monte Carlo standard errors 0.007--0.07). Right: per-batch mean of the clipped part of the capped score "
            r"(Lemma~\ref{lem:drift} predicts a non-zero drift under reward-model misspecification).}" "\n"
            r"\label{tab:S-fwdcheck}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\begin{tabular}{lll" + "r" * 10 + "}\n" r"\toprule" "\n"
            r" & & & \multicolumn{5}{c}{Uncapped DR score $-V$, batch} & \multicolumn{5}{c}{Clipped part, batch} \\"
            r"\cmidrule(lr){4-8}\cmidrule(lr){9-13}" "\n"
            r"Noise & Reward model & Fit & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5 \\" "\n" r"\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "fwdcheck.tex", body)


def generic_by_n(mt_df, regimes, scen, pols, ns, meths, caption, label, out_path, reps):
    rows = []
    for rg, rlab in regimes:
        for sc in scen:
            for pol in pols:
                for i, mt in enumerate(meths):
                    cells = []
                    for n in ns:
                        r = mt_df[(mt_df.regime == rg) & (mt_df.scenario == sc) & (mt_df.policy == pol) &
                                  (mt_df.n == n) & (mt_df.method == mt)]
                        cells.append(cell(r.iloc[0]) if len(r) else "--")
                    lead = f"{rlab} & {NOISE[sc]} & {pol}" if i == 0 else " & & "
                    rows.append(f"{lead} & {LAB[mt]} & " + " & ".join(cells) + r" \\")
                rows.append(r"\midrule")
    rows = rows[:-1]
    head = "Reg. & Noise & Overlap & Estimator & " + " & ".join(f"$n={n}$" for n in ns)
    body = (r"{\spacingset{1}\scriptsize\setlength{\tabcolsep}{3pt}" "\n" r"\begin{longtable}{llll" + "c" * len(ns) + "}\n"
            rf"\caption{{{caption} Cell: bias / RMSE / coverage, {reps} replications.}}\label{{{label}}}\\" "\n"
            r"\toprule" "\n" + head + r" \\ \midrule" "\n" r"\endfirsthead" "\n" r"\toprule" "\n" + head +
            r" \\ \midrule" "\n" r"\endhead" "\n" r"\bottomrule" "\n" r"\endfoot" "\n" + "\n".join(rows) + "\n"
            r"\end{longtable}}" "\n")
    write(out_path, body)


def contamination_table(out):
    c = pd.read_csv(TAB / "final_contamination.csv")
    meths = ["RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-n13", "RCF-DR-VR-SN"]
    rows = []
    for rg, rl in [("A-huber-oracle", "A"), ("C-huber-oracle", "C")]:
        for n in (1000, 2000, 5000, 10000, 20000):
            v = c[(c.regime == rg) & (c.n == n)].set_index("method")
            cells = [f"{f3(v.loc[k, 'bias'], True)} / {f2(v.loc[k, 'cover'])}" for k in meths]
            pred = [f"{0.02 * v.loc[k, 'tau']:.3f}" for k in ["RCF-DR-VR", "RCF-DR-VR-n13"]]
            rows.append(f"{rl if n == 1000 else ''} & {n} & " + " & ".join(cells) + " & " + " & ".join(pred) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{One-sided gross contamination (2\% of rewards shifted by $U(20,40)$), poor overlap, known logging "
            r"probabilities, 200 replications. Cells: bias / coverage. The last two columns give the Proposition~\ref{prop:contam} "
            r"approximation $\epsilon\,\bar\tau_n$ ($\epsilon=0.02$, $\bar\tau_n$ the average data-driven threshold).}" "\n"
            r"\label{tab:S-contam}" "\n" r"\footnotesize" "\n" r"\begin{tabular}{llcccccc}" "\n" r"\toprule" "\n"
            r"Reg. & $n$ & RCF-DR-os & VR ($n^{1/4}$) & VR ($n^{1/3}$) & VR-SN ($n^{1/4}$) & $\epsilon\bar\tau_n$ ($n^{1/4}$)"
            r" & $\epsilon\bar\tau_n$ ($n^{1/3}$) \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n"
            r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "contamination.tex", body)


def n10k_table(out):
    g = pd.read_csv(TAB / "final_grid10k_value_metrics.csv")
    meths = ["DM", "DR", "DR-clip", "RCF-DR", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RM-huber"]
    rows = []
    for rg, rl in [("A-huber", "A"), ("C-huber", "C")]:
        for sc in ["gauss", "t3", "contam05", "gross_pos"]:
            v = g[(g.regime == rg) & (g.scenario == sc)].set_index("method")
            rows.append(f"{rl if sc == 'gauss' else ''} & {NOISE[sc]} & " +
                        " & ".join(f"{f3(v.loc[k, 'bias'], True)} / {f2(v.loc[k, 'cover'])}" for k in meths) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = (r"\begin{table}[htbp]" "\n" r"\centering\spacingset{1}" "\n"
            r"\caption{$n=10{,}000$, poor overlap, estimated (correctly specified) propensity model, 200 replications. "
            r"Cells: bias / coverage.}" "\n" r"\label{tab:S-10k}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2.5pt}"
            "\n" r"\begin{tabular}{ll" + "c" * len(meths) + "}\n" r"\toprule" "\n"
            "Reg. & Noise & " + " & ".join(LAB[k].replace(" (no rescale)", "$^\\dagger$") for k in meths) + r" \\" "\n"
            r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write(out / "supp_tables" / "n10k.tex", body)


def main(out):
    out = Path(out)
    m = pd.read_csv(TAB / "final_grid_value_metrics.csv")
    m["ratio_se"] = m.avg_se / m.sd
    t1_representative(m, out)
    s = pd.read_csv(TAB / "final_selection.csv")
    selection_table(s, 20, "C-huber", "tab:selection",
                    r"Policy selection among $M=20$ candidate exploration policies with policy-dependent reward-model "
                    r"misspecification (regime C: reward model misspecified, propensity model correct), $n=1000$, "
                    r"500 replications. P(best): probability of selecting the true best policy; Top-3: true best among "
                    r"the three highest estimates; regret $=V(\pi^\ast)-V(\hat\pi)$; Spearman: rank correlation between "
                    r"estimated and true values; winner's curse: $\hat V(\hat\pi)-V(\hat\pi)$; Sel.\ cover: coverage of "
                    r"the naive 95\% Wald interval for the selected policy.",
                    out / "tables" / "tab_selection.tex")
    a = pd.read_csv(TAB / "final_adaptive_value_metrics.csv")
    adaptive_table(a, "C-huber-oracle", "tab:adaptive",
                   r"Batched adaptive logging (five batches; the logger is refitted after each batch), poor overlap, "
                   r"recorded logging probabilities, K-fold cross-fitted misspecified reward model (regime C), 300 "
                   r"replications. RCF-DR-VR-tail is the optional adaptive-logging variant (Section~\ref{sec:adaptive}).",
                   out / "tables" / "tab_adaptive.tex")
    # supplement
    full_grid_tables(m, out)
    relative_efficiency(m, out)
    overlap_table(m, out)
    diagnostic_alpha_table(out)
    selection_table(s, 10, "C-huber", "tab:S-sel10",
                    r"As Table~\ref{tab:selection} with the first $M=10$ candidates (several near-ties).",
                    out / "supp_tables" / "sel_M10.tex")
    selection_table(s, 20, "A-huber", "tab:S-sel20A",
                    r"As Table~\ref{tab:selection} but in regime A (reward model correctly specified), $M=20$.",
                    out / "supp_tables" / "sel_M20_A.tex")
    adaptive_table(a, "A-huber-oracle", "tab:S-adaptA",
                   r"As Table~\ref{tab:adaptive}, regime A (reward model correctly specified).",
                   out / "supp_tables" / "adapt_A.tex", placement="htbp")
    estprop_table(a, out)
    forward_table(a, out)
    fwdcheck_table(out)
    h = pd.read_csv(TAB / "final_appendix_hlog.csv")
    generic_by_n(h, [("A-h", "A"), ("B-h", "B"), ("C-h", "C"), ("D-h", "D")], ["gauss_hlog", "t3_hlog", "gross_hlog"],
                 ["poor"], (500, 1000, 2000, 5000), KEY5,
                 r"Strong propensity misspecification: logger that exploits the nonlinear reward terms (h-logger), poor "
                 r"overlap. The misspecified propensity model uses only $x_1,\dots,x_4$.", "tab:S-hlog",
                 out / "supp_tables" / "hlog.tex", 300)
    r = pd.read_csv(TAB / "final_appendix_ar1.csv")
    generic_by_n(r, [("A-huber", "A"), ("C-huber", "C")], ["gauss", "t3", "contam05"], ["moderate", "poor"],
                 (1000, 2000), KEY5, r"AR(1) covariates, $\rho=0.5$.", "tab:S-ar1", out / "supp_tables" / "ar1.tex", 300)
    contamination_table(out)
    n10k_table(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
