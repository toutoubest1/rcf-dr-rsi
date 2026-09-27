"""Simulation figures (Figures 1-5, S1) and summary tables for the paper.

usage: python -m experiments.analyze_final [grid] [select] [adapt] [contam] [extra]
Reads results/simulation/raw/*_final*.csv.gz (+ tables/adaptdiag_r3_by_batch.csv for Figure 4, right panel);
writes results/simulation/tables/final_*.csv|md and figures/fig*.png.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import FIG, GRID, INK2, RAW, SERIES, TAB, trim_png
from src.metrics import selection_metrics, value_metrics
from src.policies import candidate_policies_v3

# Paper figures are written to figures/ without in-image titles (the manuscript captions replace them) and
# then cropped to their content with a 12-pixel white border (common.trim_png).  Set PAPER_TITLES=1 to keep
# the in-image titles.
import os
PAPER = FIG
if os.environ.get("PAPER_TITLES") != "1":
    plt.Figure.suptitle = lambda self, *a, **k: None
_savefig = plt.Figure.savefig


def _save_and_trim(self, fname, *a, **k):
    _savefig(self, fname, *a, **k)
    if str(fname).endswith(".png") and os.environ.get("PAPER_TITLES") != "1":
        trim_png(fname)


plt.Figure.savefig = _save_and_trim
GRAY = "#b9b8b3"
MAIN = ["DM", "DR", "DR-clip", "RCF-DR", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RM-huber"]
KEY4 = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
NOISE = ["gauss", "t3", "t2", "contam05", "gross", "gross_pos"]
REG = {"A-huber": "A", "B-strong": "B", "C-huber": "C", "D-strong": "D"}


def _clean(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def load_grid():
    fs = [RAW / f"grid_final_n{n}.csv.gz" for n in (500, 1000, 2000, 5000)]
    d = pd.concat([pd.read_csv(f) for f in fs if f.exists()])
    return d


def grid_metrics(d):
    m = value_metrics(d, ["scenario", "policy", "regime", "n", "method"])
    ov = d.groupby(["scenario", "policy", "regime", "n"]).agg(
        ess_med=("ess_frac", "median"), alpha_med=("alpha_hat", "median")).reset_index()
    m = m.merge(ov, on=["scenario", "policy", "regime", "n"])
    m["reg"] = m.regime.map(REG)
    m.to_csv(TAB / "final_grid_value_metrics.csv", index=False)
    return m


# ------------------------------------------------------------------ tables
def table_representative(m):
    rows = []
    for rg in ["A", "B", "C", "D"]:
        for sc in ["gauss", "t3", "gross"]:
            for n in (1000, 5000):
                mm = m[(m.reg == rg) & (m.scenario == sc) & (m.n == n) & (m.policy == "poor")]
                for meth in MAIN[:7]:
                    r = mm[mm.method == meth].iloc[0]
                    rows.append(dict(regime=rg, noise=sc, n=n, method=meth,
                                     cell=f"{r.bias:+.3f} / {r.rmse:.3f} / {r.cover:.2f}"))
    t = pd.DataFrame(rows).pivot_table(index="method", columns=["regime", "noise", "n"], values="cell",
                                       aggfunc="first").loc[MAIN[:7]]
    (TAB / "final_T1_representative.md").write_text(
        "Poor overlap. Each cell: bias / RMSE / 95% coverage (500 replications).\n\n" + t.to_markdown())
    return t


def appendix_tables(m):
    for rg in ["A", "B", "C", "D"]:
        for v in ["bias", "rmse", "sd", "avg_se", "cover", "ci_len"]:
            t = m[m.reg == rg].pivot_table(index="method", columns=["scenario", "policy", "n"], values=v).loc[MAIN]
            (TAB / f"final_appendix_{rg}_{v}.md").write_text(t.to_markdown(floatfmt=".3f"))
    ov = m.groupby(["policy", "regime", "n"])[["ess_med", "alpha_med"]].median().reset_index()
    (TAB / "final_appendix_overlap.md").write_text(ov.to_markdown(index=False, floatfmt=".3f"))


def relative_efficiency(m):
    mm = m[m.method.isin(MAIN)].copy()
    mm["best"] = mm.groupby(["scenario", "policy", "regime", "n"]).rmse.transform("min")
    mm["ratio"] = mm.rmse / mm.best
    out = mm.groupby(["reg", "method"]).agg(
        geo_ratio=("ratio", lambda x: float(np.exp(np.log(x).mean()))), max_ratio=("ratio", "max"),
        mean_cover=("cover", "mean"), min_cover=("cover", "min")).reset_index()
    sym = mm[mm.scenario != "gross_pos"].groupby(["reg", "method"]).agg(
        geo_ratio_sym=("ratio", lambda x: float(np.exp(np.log(x).mean()))),
        min_cover_sym=("cover", "min")).reset_index()
    out = out.merge(sym, on=["reg", "method"])
    (TAB / "final_relative_efficiency.md").write_text(out.to_markdown(index=False, floatfmt=".3f"))
    return out


# ------------------------------------------------------------------ figures
def fig1_rmse_vs_ess(m, n=1000):
    scen = ["gauss", "t3", "contam05", "gross"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for i, rg in enumerate(["A", "C"]):
        for j, sc in enumerate(scen):
            ax = axes[i, j]
            mm = m[(m.reg == rg) & (m.scenario == sc) & (m.n == n)]
            for k, meth in enumerate(KEY4):
                s = mm[mm.method == meth].sort_values("ess_med")
                ax.plot(s.ess_med, s.rmse, color=SERIES[k], lw=2, marker="o", ms=4, label=meth)
            ax.set_xscale("log"); ax.set_title(f"{sc}, regime {rg}", loc="left"); _clean(ax)
            ax.set_xticks([0.15, 0.25, 0.4, 0.6], ["0.15", "0.25", "0.4", "0.6"]); ax.minorticks_off()
            if i == 1: ax.set_xlabel("median ESS / n")
            if j == 0: ax.set_ylabel("RMSE")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"Figure 1. RMSE vs overlap (n = {n}; good / moderate / poor target policies)",
                 x=0.01, y=0.995, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(PAPER / "fig1_rmse_vs_ess.png", dpi=170); plt.close(fig)


def fig2_bias_vs_n(m, m10=None):
    scen = ["gauss", "t3", "contam05", "gross"]
    meths = ["RCF-DR", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
    mm = m[(m.reg == "C") & (m.policy == "poor")]
    if m10 is not None:
        mm = pd.concat([mm, m10[(m10.regime == "C-huber")]])
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), sharex=True)
    for j, sc in enumerate(scen):
        for k, meth in enumerate(meths):
            s = mm[(mm.scenario == sc) & (mm.method == meth)].sort_values("n")
            axes[0, j].plot(s.n, s.bias, color=SERIES[k], lw=2, marker="o", ms=4, label=meth)
            axes[1, j].plot(s.n, s.cover, color=SERIES[k], lw=2, marker="o", ms=4)
        axes[0, j].axhline(0, color=INK2, lw=0.8); axes[1, j].axhline(0.95, color=INK2, lw=0.8, ls="--")
        axes[0, j].set_title(sc, loc="left")
        for i in range(2):
            ax = axes[i, j]; ax.set_xscale("log")
            ax.set_xticks([500, 1000, 2000, 5000, 10000], ["500", "1k", "2k", "5k", "10k"]); ax.minorticks_off()
            _clean(ax, "y")
        axes[1, j].set_xlabel("n")
    axes[0, 0].set_ylabel("bias"); axes[1, 0].set_ylabel("95% CI coverage")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Figure 2. Reward model misspecified, propensity correct (regime C), poor overlap",
                 x=0.01, y=0.995, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(PAPER / "fig2_bias_vs_n.png", dpi=170); plt.close(fig)


def selection_summary():
    d = pd.read_csv(RAW / "select_final.csv.gz")
    out = []
    for M in (10, 20):
        names = [c.name for c in candidate_policies_v3(M)]
        tv = np.array([d[d.policy == nm].truth.iloc[0] for nm in names])
        for (sc, rg, meth), g in d[d.policy.isin(names)].groupby(["scenario", "regime", "method"], sort=False):
            est = g.pivot(index="rep", columns="policy", values="est")[names].values
            se = g.pivot(index="rep", columns="policy", values="se")[names].values
            out.append(dict(M=M, scenario=sc, regime=rg, method=meth, **selection_metrics(est, tv, se)))
    s = pd.DataFrame(out)
    s.to_csv(TAB / "final_selection.csv", index=False)
    meths = ["DM", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
    cols = ["p_best", "best_in_est_top3", "regret", "regret_median", "spearman", "winners_curse", "sel_cover"]
    for M in (10, 20):
        t = s[(s.M == M) & (s.regime == "C-huber")].set_index(["scenario", "method"])[cols]
        t = t.loc[[(sc, mt) for sc in ["gauss", "t3", "contam05", "gross"] for mt in meths]]
        (TAB / f"final_T2_selection_M{M}.md").write_text(
            f"Regime C-huber (reward model wrong, propensity right), n = 1000, 500 replications, M = {M}.\n\n"
            + t.to_markdown(floatfmt=".3f"))
    return s


def fig3_selection(s, M=20):
    meths = ["DM", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
    scen = ["gauss", "t3", "contam05", "gross"]
    mets = [("p_best", "P(select true best)"), ("regret", "mean regret"),
            ("winners_curse", "winner's-curse bias"), ("sel_cover", "coverage of selected-policy CI")]
    ss = s[(s.M == M) & (s.regime == "C-huber")]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5), sharey=True)
    for ax, (v, lab) in zip(axes, mets):
        for k, sc in enumerate(scen):
            mm = ss[ss.scenario == sc].set_index("method").reindex(meths)
            y = np.arange(len(meths)) + (k - 1.5) * 0.15
            ax.scatter(mm[v], y, s=26, color=SERIES[k], label=sc, zorder=3, edgecolor="white", linewidth=0.8)
        if v == "winners_curse": ax.axvline(0, color=INK2, lw=0.8)
        if v == "sel_cover": ax.axvline(0.95, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(meths)), meths); ax.invert_yaxis(); ax.set_title(lab, loc="left"); _clean(ax, "x")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(f"Figure 3. Choosing which exploration policy to redeploy (M = {M} candidates, "
                 "policy-dependent reward misspecification)", x=0.01, y=0.99, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(PAPER / "fig3_selection.png", dpi=170); plt.close(fig)


def adaptive_summary():
    d = pd.concat([pd.read_csv(RAW / f"adapt_final_n{n}.csv.gz") for n in (1000, 2000, 5000)])
    m = value_metrics(d, ["scenario", "regime", "n", "method"])
    m.to_csv(TAB / "final_adaptive_value_metrics.csv", index=False)
    meths = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RCF-DR-VR-tail"]
    rows = []
    for rg in ["C-huber-oracle", "A-huber-oracle"]:
        for sc in ["gauss_adapt", "t3_adapt", "gross_adapt"]:
            for n in (1000, 2000, 5000):
                mm = m[(m.regime == rg) & (m.scenario == sc) & (m.n == n)].set_index("method")
                for meth in meths:
                    r = mm.loc[meth]
                    rows.append(dict(regime=rg.split("-")[0], noise=sc.replace("_adapt", ""), n=n, method=meth,
                                     cell=f"{r.bias:+.3f} / {r.rmse:.3f} / {r.cover:.2f}"))
    t = pd.DataFrame(rows).pivot_table(index="method", columns=["regime", "noise", "n"], values="cell",
                                       aggfunc="first").loc[meths]
    (TAB / "final_T3_adaptive.md").write_text(
        "Batched adaptive logging, known logging probabilities, poor overlap, K-fold reward model. "
        "Cell: bias / RMSE / coverage (300 replications).\n\n" + t.to_markdown())
    # forward vs k-fold
    f = m[m.method.isin(["DR", "RCF-DR-VR-SN"])].copy()
    f["fit"] = f.regime.map(lambda r: "forward" if "fwd-" in r else ("forward-strict" if "fwd0" in r else
                                      ("kfold, est. propensity" if r == "C-huber" else "kfold")))
    f["reward"] = f.regime.str[0]
    t2 = f.pivot_table(index=["reward", "method", "fit"], columns=["scenario", "n"], values=["bias", "rmse", "cover"])
    (TAB / "final_appendix_forward_vs_kfold.md").write_text(t2.round(3).to_markdown())
    return m


def fig4_adaptive(m, n=2000):
    meths = ["DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN", "RCF-DR-VR-tail"]
    col = {"DR-clip": SERIES[3], "RCF-DR-os": GRAY, "RCF-DR-VR": SERIES[0], "RCF-DR-VR-SN": SERIES[1],
           "RCF-DR-VR-tail": SERIES[2], "DR": INK2}
    batch = pd.read_csv(TAB / "adaptdiag_r3_by_batch.csv")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
    for ax, sc in zip(axes[:3], ["gauss_adapt", "t3_adapt", "gross_adapt"]):
        mm = m[(m.regime == "C-huber-oracle") & (m.scenario == sc) & (m.n == n)].set_index("method")
        xmax = max(mm.loc[[x for x in meths if x != "DR"], "sd"]) * 1.25
        for meth in meths:
            r = mm.loc[meth]
            x = min(r.sd, xmax)
            ax.scatter(x, abs(r.bias), s=40, color=col[meth], zorder=3, edgecolor="white", linewidth=0.8,
                       marker=">" if r.sd > xmax else "o")
            lab = meth + (f" (SD {r.sd:.2f}, off scale)" if r.sd > xmax else "")
            ax.annotate(lab, (x, abs(r.bias)), xytext={"DR-clip": (4, -10), "RCF-DR-VR-tail": (-5, -10)}.get(meth, (4, 3)),
                        ha="right" if meth == "RCF-DR-VR-tail" else "left",
                        textcoords="offset points", fontsize=7, color=INK2)
        ax.set_xlim(0, xmax * 1.35); ax.set_ylim(bottom=0)
        ax.set_title(sc.replace("_adapt", "") + f", n = {n}", loc="left"); ax.set_xlabel("Monte Carlo SD"); _clean(ax)
    axes[0].set_ylabel("|bias|")
    ax = axes[3]
    for k, (sc, lab) in enumerate([("t3", "IID logging"), ("t3_adapt", "adaptive logging")]):
        v = batch[(batch.scenario == sc) & (batch.n == 1000)].sort_values("batch")
        ax.bar(v.batch + (k - 0.5) * 0.38, v.clip_part_noisefree, width=0.36, color=SERIES[k], label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("logging batch"); ax.set_title("t3: clipping bias by batch", loc="left")
    ax.set_ylabel("contribution to bias"); ax.legend(frameon=False, fontsize=7, loc="lower left"); ax.set_ylim(top=0.001, bottom=1.35 * ax.get_ylim()[0]); _clean(ax, "y")
    fig.suptitle("Figure 4. Adaptive logging with a misspecified reward model: bias–variance frontier and "
                 "where the clipping drift comes from", x=0.01, y=0.99, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(PAPER / "fig4_adaptive.png", dpi=170); plt.close(fig)


def fig5_contamination():
    d = pd.concat([pd.read_csv(RAW / f"contam_final_n{n}.csv.gz") for n in (1000, 2000, 5000, 10000, 20000)])
    m = value_metrics(d, ["regime", "n", "method"])
    tau = d[d.method != "RCF-DR-os"].groupby(["regime", "n", "method"]).tau.mean().reset_index()
    m = m.merge(tau, on=["regime", "n", "method"], how="left")
    m.to_csv(TAB / "final_contamination.csv", index=False)
    meths = ["RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-n13", "RCF-DR-VR-SN"]
    labels = {"RCF-DR-os": "RCF-DR-os (fixed τ)", "RCF-DR-VR": "VR, τ ∝ n^1/4", "RCF-DR-VR-n13": "VR, τ ∝ n^1/3",
              "RCF-DR-VR-SN": "VR-SN, τ ∝ n^1/4"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rg, title in [(axes[0], "A-huber-oracle", "contamination only (reward model correct)"),
                          (axes[1], "C-huber-oracle", "contamination + misspecified reward model")]:
        mm = m[m.regime == rg]
        for k, meth in enumerate(meths):
            s = mm[mm.method == meth].sort_values("n")
            ax.plot(s.n, s.bias, color=SERIES[k], lw=2, marker="o", ms=4, label=labels[meth])
            if rg == "A-huber-oracle" and meth != "RCF-DR-os":
                ax.plot(s.n, 0.02 * s.tau, color=SERIES[k], lw=1, ls="--")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log"); ax.set_xticks([1000, 2000, 5000, 10000, 20000], ["1k", "2k", "5k", "10k", "20k"])
        ax.minorticks_off(); ax.set_xlabel("n"); ax.set_title(title, loc="left"); _clean(ax, "y")
    axes[0].set_ylabel("bias (2% one-sided outliers)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 5. Robustness–consistency trade-off: one-sided contamination bias grows with τ_n "
                 "(dashed: ε·τ_n, Proposition 1)", x=0.01, y=0.99, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(PAPER / "fig5_contamination.png", dpi=170); plt.close(fig)
    return m


def fig6_diagnostic(d):
    """Coverage and error vs estimated tail index (overlap diagnostic)."""
    dd = d[d.method.isin(["DR", "RCF-DR-VR-SN"]) & d.regime.isin(["A-huber", "C-huber"]) &
           d.scenario.isin(["gauss", "t3", "contam05", "gross"])].copy()
    dd["cover"] = (np.abs(dd.est - dd.truth) <= 1.96 * dd.se)
    dd["err2"] = (dd.est - dd.truth) ** 2
    bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0, np.inf]
    dd["abin"] = pd.cut(dd.alpha_hat, bins)
    g = dd.groupby(["method", "abin"], observed=True).agg(cover=("cover", "mean"), rmse=("err2", lambda x: np.sqrt(x.mean())),
                                                         count=("cover", "size")).reset_index()
    g.to_csv(TAB / "final_diagnostic_by_alpha.csv", index=False)
    cat = d[d.method == "DR"].groupby(["policy", "n"]).overlap_cat.value_counts(normalize=True).unstack().fillna(0)
    cat.to_csv(TAB / "final_diagnostic_categories.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    mids = {iv: (iv.left + min(iv.right, 5.0)) / 2 for iv in g.abin.unique()}
    for k, meth in enumerate(["DR", "RCF-DR-VR-SN"]):
        s = g[g.method == meth].copy(); s["x"] = s.abin.map(mids)
        axes[0].plot(s.x, s.cover, color=SERIES[k], lw=2, marker="o", ms=5, label=meth)
        axes[1].plot(s.x, s.rmse, color=SERIES[k], lw=2, marker="o", ms=5, label=meth)
    for ax in axes:
        ax.axvspan(0, 2.0, color="#f1c9c0", alpha=0.45, lw=0); ax.axvspan(2.0, 2.5, color="#f4e3b5", alpha=0.45, lw=0)
        ax.set_xlabel("estimated tail index α̂ of the importance weights (Hill, k = 2√n)"); _clean(ax)
        ax.text(1.0, ax.get_ylim()[1], "non-root-n", fontsize=7, va="top", color=INK2)
        ax.text(2.02, ax.get_ylim()[1], "borderline", fontsize=7, va="top", color=INK2)
    axes[0].axhline(0.95, color=INK2, lw=0.8, ls="--"); axes[0].set_ylabel("95% Wald CI coverage")
    axes[1].set_ylabel("RMSE"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 6 (diagnostic). Coverage and error by estimated weight tail index "
                 "(all n, overlaps, regimes A and C; symmetric noise)", x=0.01, y=0.99, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(PAPER / "fig6_overlap_diagnostic.png", dpi=170); plt.close(fig)
    return g, cat


def appendix_extra(tag, fname, regimes, title):
    f = RAW / f"{tag}.csv.gz"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    m = value_metrics(d, ["scenario", "policy", "regime", "n", "method"])
    m.to_csv(TAB / f"{fname}.csv", index=False)
    m["cell"] = m.apply(lambda r: f"{r.bias:+.3f} / {r.rmse:.3f} / {r.cover:.2f}", axis=1)
    out = [title, ""]
    for rg in regimes:
        t = m[m.regime == rg].pivot_table(index="method", columns=["scenario", "policy", "n"], values="cell",
                                          aggfunc="first").reindex(MAIN)
        out += [f"### {rg}", "", t.to_markdown(), ""]
    (TAB / f"{fname}.md").write_text("\n".join(out))
    return m


if __name__ == "__main__":
    parts = sys.argv[1:] or ["grid", "select", "adapt", "contam"]
    if "grid" in parts:
        d = load_grid(); m = grid_metrics(d)
        m10 = None
        if (RAW / "grid10k_final.csv.gz").exists():
            m10 = value_metrics(pd.read_csv(RAW / "grid10k_final.csv.gz"), ["scenario", "policy", "regime", "n", "method"])
            m10.to_csv(TAB / "final_grid10k_value_metrics.csv", index=False)
        table_representative(m); appendix_tables(m); relative_efficiency(m)
        fig1_rmse_vs_ess(m); fig2_bias_vs_n(m, m10); fig6_diagnostic(d)
    if "select" in parts:
        fig3_selection(selection_summary())
    if "adapt" in parts:
        fig4_adaptive(adaptive_summary())
    if "contam" in parts:
        fig5_contamination()
    if "extra" in parts:
        appendix_extra("hlog_final", "final_appendix_hlog", ["A-h", "B-h", "C-h", "D-h"],
                       "h-logger (strong propensity misspecification). Cell: bias / RMSE / coverage (300 reps).")
        appendix_extra("ar1_final", "final_appendix_ar1", ["A-huber", "C-huber"],
                       "AR(1) rho = 0.5 covariates. Cell: bias / RMSE / coverage (300 reps).")
