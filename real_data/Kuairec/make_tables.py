"""KuaiRec tables and supplementary figures"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_data.kuairec.paths import RAW as RAWDIR, ROOT, TAB

HAVE_RAW = (RAWDIR / "cell_moderate_250.pkl.gz").exists()   # Table S39 and Figure S4 need the saved replications

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
for sub in ("supp_tables", "figures", "tables"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
S = pd.read_csv(TAB / "kuai_target_summary.csv"); S = S[S.dataset == "kuairec"]
SEL = pd.read_csv(TAB / "kuai_selection_summary.csv"); SEL = SEL[SEL.dataset == "kuairec"]
D = json.load(open(TAB / "design_kuairec.json"))
A = json.load(open(TAB / "kuairec_audit.json"))
M8 = ["DM", "IPW", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
OVL = {"good": "Uniform", "moderate": "Moderate", "poor": "Poor"}
NS = [250, 500, 1000, 2000, 4000]
PRIM, OLS, RAW = "log1p_wr|huber", "log1p_wr|ols", "wr|huber"


def f(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def write(name, txt):
    (OUT / "supp_tables" / name).write_text(txt)


rows = []
for ov in ["good", "moderate", "poor"]:
    L = D["logging"][ov]
    rows.append(f"{OVL[ov]} & {L['kappa']:.2f} & {L['pop_ess_train']:.4f} & {L['pop_ess_eval']:.4f} & "
                f"{L['min_pb']:.2e} & {L['max_ratio']:.1f} \\\\")
write("kuairec_kappa.tex", r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{KuaiRec overlap calibration. $\kappa$ is the smallest value on the grid $\{0,0.05,\dots,40\}$ whose population ESS$/n=1/\{n_u^{-1}\sum_u\sum_c\pi_e(c\mid u)^2/\pi_b(c\mid u)\}$ on the policy-training users is at most the target (uniform logging for the first row; 0.05 and 0.01 otherwise). The calibration uses only the two policies, not outcomes. The last three columns are computed on the 988 evaluation users.}\label{tab:S-kuairec-kappa}
\footnotesize
\begin{tabular}{lccccc}
\toprule
Overlap & $\kappa$ & ESS$/n$ (training users) & ESS$/n$ (evaluation users) & $\min\pi_b$ & $\max\pi_e/\pi_b$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

def rrow(lab, key):
    q = A[f"{key}_q"]
    return (f"{lab} & {100*A[key+'_frac_zero']:.2f} & {A[key+'_median']:.3f} & {A[key+'_mad']:.3f} & {A[key+'_skew']:.2f} & "
            f"{A[key+'_exkurt']:.1f} & {q['1']:.3f} & {q['10']:.3f} & {q['90']:.3f} & {q['99']:.3f} & {A[key+'_max']:.2f} \\\\")
write("kuairec_reward.tex", r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{KuaiRec reward distributions over all 4,676,570 observed interactions of the small matrix (1,411 users $\times$ 3,327 videos). Zeros are exact zeros (\%). The primary reward is $\log(1+\text{watch ratio})$. The last panel gives the residuals of an additive (user + video) fit, which approximate the residual structure faced by a reward model.}\label{tab:S-kuairec-reward}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{lcccccccccc}
\toprule
Reward & Zeros (\%) & Median & MAD & Skewness & Ex.\ kurtosis & $q_{0.01}$ & $q_{0.10}$ & $q_{0.90}$ & $q_{0.99}$ & Max \\
\midrule
""" + "\n".join([rrow("Watch ratio", "watch_ratio"), rrow(r"$\log(1+\text{watch ratio})$ (primary)", "log1p_watch_ratio"),
                 rrow("Play time (s)", "play_time_s"), rrow(r"$\log(1+\text{play time})$", "log1p_play_time_s")]) + r"""
\midrule
\multicolumn{11}{l}{Residuals of the additive user + video fit} \\
\midrule
 & MAD & SD & Bowley & Skewness & Ex.\ kurtosis & \multicolumn{5}{l}{Variance share: user / video effects} \\
$\log(1+\text{watch ratio})$ & """ + f"{A['additive_resid_mad']:.3f} & {A['additive_resid_sd']:.3f} & {f(A['additive_resid_bowley'],2)} & {A['additive_resid_skew']:.2f} & {A['additive_resid_exkurt']:.1f} & \\multicolumn{{5}}{{l}}{{{A['var_share_user']:.3f} / {A['var_share_item']:.3f}}}" + r""" \\
Watch ratio & """ + f"{A['raw_additive_resid_mad']:.3f} & -- & {f(A['raw_additive_resid_bowley'],2)} & {A['raw_additive_resid_skew']:.1f} & {A['raw_additive_resid_exkurt']:.0f} & \\multicolumn{{5}}{{l}}{{}}" + r""" \\
\bottomrule
\end{tabular}}
\end{table}
""")

order = ["Best-single", "Best-single-2", "Uniform", "Legacy-greedy", "Ridge-sm1", "Ridge-sm3 (target)", "Ridge-sm10", "Ridge-greedy", "RF-greedy"]
desc = {"Best-single": "single category with best training mean (cat28)", "Best-single-2": "second-best single category (cat7)",
        "Uniform": "uniform over the 18 categories", "Legacy-greedy": r"$\arg\max$ of a ridge model on PCs 1--3",
        "Ridge-sm1": r"$\mathrm{softmax}(1\cdot z(\hat u))$", "Ridge-sm3 (target)": r"$\mathrm{softmax}(3\cdot z(\hat u))$, the target",
        "Ridge-sm10": r"$\mathrm{softmax}(10\cdot z(\hat u))$", "Ridge-greedy": r"$\arg\max\hat u$", "RF-greedy": r"$\arg\max$ of a random forest"}
rows = []
for p in order:
    e = D["cand_pop_ess"][p]
    rows.append(f"{p.replace(' (target)','')} & {desc[p]} & {D['truth']['log1p_wr'][p]:.4f} & {D['truth']['wr'][p]:.4f} & "
                f"{e['good']:.3f} & {e['moderate']:.3f} & {e['poor']:.4f} \\\\")
rows.append(r"\midrule" + "\n" + f"Per-user oracle & best category per user (reference only) & {D['vbs']['log1p_wr']:.4f} & {D['vbs']['wr']:.4f} & -- & -- & -- \\\\")
write("kuairec_candidates.tex", r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{Exact values of the nine KuaiRec candidate policies on the 988 evaluation users, under the primary reward and the raw watch ratio, and their population ESS$/n$ under the three behavior policies. $\hat u$ is the per-category ridge prediction of the user--category mean reward, fitted on the training users; $z(\cdot)$ standardizes across categories within a user. Best-single and RF-greedy coincide on every evaluation user.}\label{tab:S-kuairec-cands}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{2}{c}{Exact value} & \multicolumn{3}{c}{Population ESS$/n$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
Policy & Definition & $\log(1+\text{watch ratio})$ & Watch ratio & Uniform & Moderate & Poor \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

def full_table(variant, label, cap):
    x = S[S.variant == variant].set_index(["overlap", "n", "method"])
    lines = []
    for ov in ["good", "moderate", "poor"]:
        for n in NS:
            for i, m in enumerate(M8):
                r = x.loc[(ov, n, m)]
                lead = f"{OVL[ov]} & {n}" if i == 0 else " & "
                lines.append(f"{lead} & {m} & {f(r.bias, 4)} & {r.sd:.4f} & {r.rmse:.4f} & {r.q99_abs_err:.3f} & {r.coverage:.3f} & {r.ci_width:.3f} \\\\")
            if not (ov == "poor" and n == 4000):
                lines.append(r"\midrule" if n == 4000 else r"\addlinespace[2pt]")
    return (r"""{\spacingset{1}\footnotesize\setlength{\LTcapwidth}{\textwidth}
\begin{longtable}{llccccccc}
\caption{""" + cap + r"""}\label{""" + label + r"""}\\
\toprule
Overlap & $n$ & Estimator & Bias & SD & RMSE & $q_{0.99}|\text{err}|$ & Coverage & CI width \\
\midrule
\endfirsthead
\toprule
Overlap & $n$ & Estimator & Bias & SD & RMSE & $q_{0.99}|\text{err}|$ & Coverage & CI width \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(lines) + r"""
\end{longtable}}
""")

write("kuairec_primary_full.tex", full_table(PRIM, "tab:S-kuairec-full",
      r"KuaiRec, primary reward $\log(1+\text{watch ratio})$, Huber reward model: target-policy results for all overlap levels, sample sizes and estimators (500 replications per cell; errors relative to the exact policy value 0.6342). $q_{0.99}|\text{err}|$ is the 99th percentile of the absolute error; coverage and width refer to the estimator's own 95\% Wald interval. DR and DR-clip coincide under uniform logging because the Ionides cap never binds."))
write("kuairec_ols_full.tex", full_table(OLS, "tab:S-kuairec-ols",
      r"KuaiRec sensitivity: least-squares reward model (frozen OLS routine) in place of Huber regression; primary reward, same logged samples and folds as Table~\ref{tab:S-kuairec-full}."))
write("kuairec_raw_full.tex", full_table(RAW, "tab:S-kuairec-raw",
      r"KuaiRec sensitivity: untransformed watch ratio as the reward (exact target value 1.0089), Huber reward model, same logged samples and folds as Table~\ref{tab:S-kuairec-full}. The residuals are strongly right-skewed (Table~\ref{tab:S-kuairec-reward})."))

x = S[(S.variant == PRIM) & (S.method == "RCF-DR-VR-SN")].set_index(["overlap", "n"])
xos = S[(S.variant == PRIM) & (S.method == "RCF-DR-os")].set_index(["overlap", "n"])
lines = []
for ov in ["good", "moderate", "poor"]:
    for i, n in enumerate(NS):
        r, o = x.loc[(ov, n)], xos.loc[(ov, n)]
        lines.append(f"{OVL[ov] if i == 0 else ''} & {n} & {r.ess_frac:.3f} & {r.max_w:.1f} & {r.alpha_hat:.2f} & {r.frac_nonrootn:.3f} & "
                     f"{r.frac_capped_vrsn:.4f} & {r.frac_huber:.4f} & {r.resid_mad:.3f} & {r.tau:.3f} & {o.frac_huber:.3f} & {o.tau:.3f} \\\\")
    if ov != "poor":
        lines.append(r"\midrule")
write("kuairec_weights.tex", r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{KuaiRec weight and truncation diagnostics for the target policy (primary reward, Huber reward model; averages over 500 replications, medians for ESS$/n$, max $w$ and $\hat\alpha$). ESS$/n$ is the sample effective sample size fraction; $\hat\alpha$ is the Hill estimate of the weight tail with $k=\lceil2\sqrt n\rceil$; ``$\hat\alpha\le2$'' is the share of replications flagged non-root-$n$ by the overlap diagnostic; ``capped'' is the fraction of weights above $c_n=\bar w\sqrt{n/\log n}$; ``Huber'' is the fraction of residuals with $|r|>\tau_n$; the residual MAD is the normal-consistent MAD of the cross-fitted residuals. The last two columns give the corresponding truncation fraction and threshold of the fixed-threshold RCF-DR-os.}\label{tab:S-kuairec-weights}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{llcccccccccc}
\toprule
 & & & & & & \multicolumn{4}{c}{RCF-DR-VR-SN} & \multicolumn{2}{c}{RCF-DR-os} \\
\cmidrule(lr){7-10}\cmidrule(lr){11-12}
Overlap & $n$ & ESS$/n$ & Max $w$ & $\hat\alpha$ & $\hat\alpha\le2$ & Capped & Huber & Resid.\ MAD & $\tau_n$ & Huber & $\tau$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

if HAVE_RAW:
    ems = ["DM", "SNIPW", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
    lines = []
    errs = {}
    for ov in ["moderate", "poor"]:
        for i, n in enumerate(NS):
            d = pd.read_pickle(RAWDIR / f"cell_{ov}_{n}.pkl.gz")
            d = d[(d.variant == PRIM) & (d.policy == "Ridge-sm3 (target)")]
            cells = []
            for m in ems:
                e = np.abs((d[d.method == m].est - d[d.method == m].truth).values)
                errs[(ov, n, m)] = e
                cells.append(f"{np.median(e):.4f} & {e.max():.3f}")
            lines.append(f"{OVL[ov] if i == 0 else ''} & {n} & " + " & ".join(cells) + r" \\")
        if ov == "moderate":
            lines.append(r"\midrule")
    write("kuairec_errors.tex", r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{KuaiRec absolute-error distribution for the target policy under moderate and poor overlap (primary reward, Huber reward model): median and maximum of $|\hat V-V|$ over 500 replications. Occasional large errors of the model-based estimators come from the per-category reward model extrapolating for rarely logged categories; they are shared by DM and the DR-type estimators and do not affect SNIPW.}\label{tab:S-kuairec-errors}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{ll""" + "cc" * len(ems) + r"""}
\toprule
 & & """ + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in ems) + r""" \\
""" + "".join(f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(ems))) + r"""
Overlap & $n$ & """ + " & ".join(["Median & Max"] * len(ems)) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
else:
    print("results/kuairec/raw/ not found: skipping kuairec_errors.tex (Table S39)")

y = SEL[SEL.variant == PRIM]
rg = y.pivot_table(index=["overlap", "n"], columns="method", values="regret")
pe = y.pivot_table(index=["overlap", "n"], columns="method", values="p_eps_best")
wc = y.pivot_table(index=["overlap", "n"], columns="method", values="winners_curse")
lines = []
for ov in ["good", "moderate", "poor"]:
    for i, n in enumerate(NS):
        k = (ov, n)
        lines.append(f"{OVL[ov] if i == 0 else ''} & {n} & " + " & ".join(f"{1000*rg.loc[k, m]:.1f}" for m in M8) + " & "
                     + " & ".join(f"{pe.loc[k, m]:.2f}" for m in ["SNIPW", "DR", "RCF-DR-VR-SN"]) + " & "
                     + " & ".join(f(1000 * wc.loc[k, m], 1) for m in ["SNIPW", "DR", "RCF-DR-VR-SN"]) + r" \\")
    if ov != "poor":
        lines.append(r"\midrule")
write("kuairec_selection.tex", r"""\begin{table}[htbp]
\centering\spacingset{1}
\caption{KuaiRec policy selection among the nine candidates of Table~\ref{tab:S-kuairec-cands} (primary reward, Huber reward model, 500 replications). Each estimator selects the candidate with the largest estimated value. Regret $=V(\pi^\ast)-V(\hat\pi)$ and the winner's curse $\hat V(\hat\pi)-V(\hat\pi)$ are multiplied by 1000. Because the top candidates are tied or within $5\times10^{-4}$, ``P($\le$0.001)'' reports the probability that the regret is at most 0.001 instead of the probability of selecting a unique best policy.}\label{tab:S-kuairec-selection}
\footnotesize
\adjustbox{max width=\textwidth}{%
\begin{tabular}{ll""" + "c" * 8 + "ccc" + "ccc" + r"""}
\toprule
 & & \multicolumn{8}{c}{Mean regret $\times1000$} & \multicolumn{3}{c}{P(regret $\le0.001$)} & \multicolumn{3}{c}{Winner's curse $\times1000$} \\
\cmidrule(lr){3-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}
Overlap & $n$ & """ + " & ".join(M8) + r""" & SNIPW & DR & VR-SN & SNIPW & DR & VR-SN \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")

SHOW = ["DM", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
STY = {"DM": ("tab:gray", "s"), "SNIPW": ("tab:olive", "^"), "DR": ("tab:red", "o"), "DR-clip": ("tab:orange", "v"),
       "RCF-DR-os": ("tab:purple", "D"), "RCF-DR-VR-SN": ("tab:blue", "*")}
COLS = [("good", "Uniform logging (ESS/$n$ = 0.30)"), ("moderate", "Moderate overlap (ESS/$n$ = 0.046)"), ("poor", "Poor overlap (ESS/$n$ = 0.009)")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def line_panel(ax, g, col, m):
    h = g[g.method == m].sort_values("n")
    c, mk = STY[m]
    vs = m == "RCF-DR-VR-SN"
    ax.plot(h.n, h[col], marker=mk, color=c, lw=2.2 if vs else 1.2, ms=8 if vs else 5, label=m, zorder=3 if vs else 2)


# raw watch-ratio sensitivity: RMSE, bias, coverage
fig, axes = plt.subplots(3, 3, figsize=(10.5, 8), sharex=True)
for j, (ov, title) in enumerate(COLS):
    g = S[(S.variant == RAW) & (S.overlap == ov)]
    for i, (col, lab) in enumerate([("rmse", "RMSE"), ("bias", "Bias"), ("coverage", "95% Wald coverage")]):
        ax = axes[i, j]
        for m in SHOW:
            line_panel(ax, g, col, m)
        ax.set_xscale("log")
        if col == "rmse":
            ax.set_yscale("log"); ax.set_title(title, fontsize=9.5)
        if col == "bias":
            ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-0.2, 0.15)
        if col == "coverage":
            ax.axhline(0.95, color="k", ls=":", lw=1); ax.set_ylim(0, 1); ax.set_xlabel("$n$")
        if j == 0:
            ax.set_ylabel(lab)
        ax.set_xticks(NS, [str(v) for v in NS]); ax.minorticks_off()
axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "figures/kuairec_raw_sensitivity.pdf", bbox_inches="tight"); plt.close(fig)

# selection regret
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for j, (ov, title) in enumerate(COLS):
    g = y[y.overlap == ov]
    for m in SHOW:
        h = g[g.method == m].sort_values("n")
        c, mk = STY[m]
        vs = m == "RCF-DR-VR-SN"
        axes[j].plot(h.n, 1000 * h.regret, marker=mk, color=c, lw=2.2 if vs else 1.2, ms=8 if vs else 5, label=m)
    axes[j].set_xscale("log"); axes[j].set_title(title, fontsize=9.5); axes[j].set_xlabel("$n$")
    axes[j].set_xticks(NS, [str(v) for v in NS]); axes[j].minorticks_off()
axes[0].set_ylabel("Mean regret $\\times 1000$")
axes[0].legend(fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "figures/kuairec_regret.pdf", bbox_inches="tight"); plt.close(fig)

# error distributions at n = 4000
if HAVE_RAW:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    for ax, ov in zip(axes, ["moderate", "poor"]):
        data = [errs[(ov, 4000, m)] for m in ems]
        ax.boxplot(data, whis=(1, 99), flierprops=dict(markersize=2.5))
        ax.set_yscale("log"); ax.set_xticks(range(1, len(ems) + 1), ems, fontsize=8)
        ax.set_title(f"{OVL[ov]} overlap, $n=4000$", fontsize=9.5)
    axes[0].set_ylabel("$|\\hat V - V|$ (log scale; whiskers 1%--99%)")
    fig.tight_layout(); fig.savefig(OUT / "figures/kuairec_error_distribution.pdf", bbox_inches="tight"); plt.close(fig)
else:
    print("results/kuairec/raw/ not found: skipping kuairec_error_distribution.pdf (Figure S4)")

r = S[S.variant == PRIM].pivot_table(index=["overlap", "n"], columns="method", values="rmse")
c = S[S.variant == PRIM].pivot_table(index=["overlap", "n"], columns="method", values="coverage")
ms = ["DM", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
L = [r"""\begin{table}[t]
\centering\spacingset{1}
\caption{KuaiRec target-policy results for the primary reward $\log(1+\text{watch ratio})$ (500 replications per cell; exact policy values from the fully observed user--video matrix). RMSE is multiplied by 100. Coverage is for the estimator's 95\% Wald interval. Population ESS$/n$ of the behavior policy is 0.30 (uniform), 0.046 (moderate) and 0.009 (poor).}
\label{tab:kuairec}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccccc}
\toprule
 & & \multicolumn{6}{c}{RMSE $\times 100$} & \multicolumn{3}{c}{Coverage} \\
\cmidrule(lr){3-8}\cmidrule(lr){9-11}
Overlap & $n$ & DM & SNIPW & DR & DR-clip & RCF-DR-os & RCF-DR-VR-SN & SNIPW & RCF-DR-os & RCF-DR-VR-SN \\
\midrule"""]
for ov in ["good", "moderate", "poor"]:
    for i, n in enumerate([250, 1000, 4000]):
        k = (ov, n)
        L.append(f"{OVL[ov] if i == 0 else ''} & {n} & " + " & ".join(f"{100 * r.loc[k, m]:.2f}" for m in ms) + " & "
                 + " & ".join(f"{c.loc[k, m]:.2f}" for m in ["SNIPW", "RCF-DR-os", "RCF-DR-VR-SN"]) + r" \\")
    if ov != "poor":
        L.append(r"\midrule")
L.append(r"""\bottomrule
\end{tabular}}
\end{table}""")
(OUT / "tables" / "tab_kuairec.tex").write_text("\n".join(L) + "\n")
print("ok")
