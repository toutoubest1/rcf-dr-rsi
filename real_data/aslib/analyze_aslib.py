"""ASlib summaries, figures (Figure 6, S2, S3) and evaluation of the pre-registered decision criteria.

usage: python -m real_data.aslib.analyze_aslib        -> results/aslib/tables/*.csv, figures/aslib_*.png
       python -m real_data.aslib.analyze_aslib md     -> results/aslib/tables/aslib_md_tables.md
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from real_data.aslib.paths import FIG, RAW, TAB  # noqa: E402

FIG.mkdir(exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
SCEN = ["SAT03-16_INDU", "ASP-POTASSCO"]
PRIMARY = {"SAT03-16_INDU": "logPAR10", "ASP-POTASSCO": "logPAR1"}
METHODS = ["DM", "IPW", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
TARGET = "Ridge-sm3 (target)"
OV = ["good", "moderate", "poor"]


def load(scen):
    fs = sorted((RAW / scen).glob("cell_*.pkl.gz"))
    return pd.concat([pd.read_pickle(f) for f in fs], ignore_index=True)


def variant_label(pop, reward, rmodel, scen):
    if pop == "allsolved":
        return "all-solved subpop. (logPAR1)"
    base = {"logPAR10": "logPAR10", "logPAR1": "logPAR1", "speed": "speed"}[reward]
    tag = " [primary]" if reward == PRIMARY[scen] and rmodel == "huber" else ""
    return base + (" / OLS reward model" if rmodel == "ols" else "") + tag


def summarise(df):
    e = df["est"] - df["truth"]
    lo, hi = df["est"] - 1.96 * df["se"], df["est"] + 1.96 * df["se"]
    return pd.Series(dict(
        reps=len(df), bias=e.mean(), sd=df["est"].std(), rmse=np.sqrt((e ** 2).mean()),
        q99_abs_err=np.quantile(np.abs(e), 0.99), max_abs_err=np.abs(e).max(),
        coverage=((lo <= df["truth"]) & (df["truth"] <= hi)).mean(), ci_width=(hi - lo).mean(),
        frac_huber=df["frac_huber"].mean(), frac_clipped=df["frac_clipped"].mean(), tau=df["tau"].mean()))


def selection(df):
    """df: one (overlap, n, variant) block with all candidates; per rep and method choose argmax est."""
    out = []
    truths = df.groupby("policy")["truth"].first()
    best_p, vbest = truths.idxmax(), truths.max()
    for m, g in df.groupby("method"):
        piv = g.pivot_table(index="rep", columns="policy", values="est")
        chosen = piv.idxmax(axis=1)
        vch = truths.loc[chosen].values
        est_ch = piv.max(axis=1).values
        # Kendall-type pairwise ordering accuracy
        pols = list(truths.index)
        T = truths.values
        E = piv[pols].values
        iu = np.triu_indices(len(pols), 1)
        agree = (np.sign(E[:, iu[0]] - E[:, iu[1]]) == np.sign(T[iu[0]] - T[iu[1]])).mean()
        out.append(dict(method=m, p_best=np.mean(chosen.values == best_p), regret=np.mean(vbest - vch),
                        regret_q90=np.quantile(vbest - vch, 0.9), winners_curse=np.mean(est_ch - vch),
                        pair_order_acc=agree))
    return pd.DataFrame(out)


def main():
    all_sum, all_sel, all_diag = [], [], []
    for scen in SCEN:
        if not (RAW / scen).exists():
            continue
        df = load(scen)
        df["variant"] = [variant_label(p, r, m, scen) for p, r, m in zip(df["pop"], df["reward"], df["rmodel"])]
        # weight diagnostics (target policy, one row per rep x pop)
        d = df[(df.method == "DR") & (df.policy == TARGET) & (df.rmodel == "huber")]
        d = d.drop_duplicates(["overlap", "n", "pop", "rep"])
        dg = d.groupby(["overlap", "n", "pop"]).agg(ess_frac=("ess_frac", "median"), max_w=("max_w", "median"),
                                                    max_w_q90=("max_w", lambda x: np.quantile(x, .9)),
                                                    alpha_hat=("alpha_hat", "median"),
                                                    frac_nonrootn=("alpha_hat", lambda x: np.mean(x <= 2)),
                                                    frac_capped_vrsn=("frac_capped_vrsn", "mean")).reset_index()
        dg.insert(0, "scenario", scen)
        all_diag.append(dg)
        t = df[df.policy == TARGET]
        s = t.groupby(["variant", "overlap", "n", "method"]).apply(summarise, include_groups=False).reset_index()
        s.insert(0, "scenario", scen)
        all_sum.append(s)
        sel = df.groupby(["variant", "overlap", "n"]).apply(selection, include_groups=False).reset_index()
        sel = sel.drop(columns=[c for c in sel.columns if c.startswith("level_")])
        sel.insert(0, "scenario", scen)
        all_sel.append(sel)
    S = pd.concat(all_sum)
    SEL = pd.concat(all_sel)
    DG = pd.concat(all_diag)
    S.to_csv(TAB / "aslib_target_summary.csv", index=False)
    SEL.to_csv(TAB / "aslib_selection_summary.csv", index=False)
    DG.to_csv(TAB / "aslib_weight_diagnostics.csv", index=False)
    figures(S, SEL)
    criteria(S, SEL)
    return S, SEL, DG


def criteria(S, SEL):
    """Pre-registered decision criteria C1-C4 (docs/aslib/PREREGISTRATION.md, section 6)."""
    rows = []
    for (scen, var), x in S.groupby(["scenario", "variant"]):
        p = x.pivot_table(index=["overlap", "n"], columns="method", values="rmse")
        q = x.pivot_table(index=["overlap", "n"], columns="method", values="q99_abs_err")
        c = x.pivot_table(index=["overlap", "n"], columns="method", values="coverage")
        sel = SEL[(SEL.scenario == scen) & (SEL.variant == var)].pivot_table(index=["overlap", "n"], columns="method",
                                                                              values="regret")
        for ov in ["moderate", "poor"]:
            pp, qq, ss = p.loc[ov], q.loc[ov], sel.loc[ov]
            c1 = ((pp["DR"] >= 1.5 * pp["RCF-DR-VR-SN"]) | (qq["DR"] >= 2 * qq["RCF-DR-VR-SN"]))
            c2a = (pp["RCF-DR-VR-SN"] < pp["DR"]) & (pp["RCF-DR-VR-SN"] < pp["DR-clip"])
            c2b = ss["RCF-DR-VR-SN"] <= ss["DR"]
            rows.append(dict(scenario=scen, variant=var, overlap=ov, C1_cells=f"{c1.sum()}/{len(c1)}",
                             C2a_cells=f"{c2a.sum()}/{len(c2a)}", C2b_cells=f"{c2b.sum()}/{len(c2b)}",
                             rmse_ratio_DR=f"{(pp['DR'] / pp['RCF-DR-VR-SN']).min():.2f}-{(pp['DR'] / pp['RCF-DR-VR-SN']).max():.2f}",
                             rmse_ratio_DRclip=f"{(pp['DR-clip'] / pp['RCF-DR-VR-SN']).min():.2f}-{(pp['DR-clip'] / pp['RCF-DR-VR-SN']).max():.2f}",
                             rmse_ratio_os=f"{(pp['RCF-DR-os'] / pp['RCF-DR-VR-SN']).min():.2f}-{(pp['RCF-DR-os'] / pp['RCF-DR-VR-SN']).max():.2f}",
                             rmse_ratio_SNIPW=f"{(pp['SNIPW'] / pp['RCF-DR-VR-SN']).min():.2f}-{(pp['SNIPW'] / pp['RCF-DR-VR-SN']).max():.2f}",
                             cov_VRSN=" / ".join(f"{v:.2f}" for v in c.loc[ov]["RCF-DR-VR-SN"].values)))
        cg = c.loc["good"]["RCF-DR-VR-SN"]
        cm = c.loc["moderate"]["RCF-DR-VR-SN"]
        rows.append(dict(scenario=scen, variant=var, overlap="C3",
                         C3_good_n2000plus=f"{cg.loc[[2000, 4000]].min():.3f}",
                         C3_moderate_n2000plus=f"{cm.loc[[2000, 4000]].min():.3f}",
                         C3_pass=bool(cg.loc[[2000, 4000]].min() >= 0.90 and cm.loc[[2000, 4000]].min() >= 0.90),
                         cov_VRSN="good: " + " / ".join(f"{v:.2f}" for v in cg.values)))
    pd.DataFrame(rows).to_csv(TAB / "aslib_criteria.csv", index=False)


SHOW = ["DM", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
STY = {"DM": ("tab:gray", "s"), "SNIPW": ("tab:olive", "^"), "DR": ("tab:red", "o"), "DR-clip": ("tab:orange", "v"),
       "RCF-DR-os": ("tab:purple", "D"), "RCF-DR-VR-SN": ("tab:blue", "*"), "IPW": ("tab:brown", "x"),
       "RCF-DR-VR": ("tab:cyan", "P")}


def _grid(S, col, fname, ylab, variant_key="primary", logy=True, hline=None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for i, scen in enumerate(SCEN):
        for j, ov in enumerate(OV):
            ax = axes[i, j]
            x = S[(S.scenario == scen) & (S.overlap == ov) & S.variant.str.contains(variant_key, regex=False)]
            for m in SHOW:
                g = x[x.method == m].sort_values("n")
                if len(g):
                    c, mk = STY[m]
                    ax.plot(g.n, g[col], marker=mk, color=c, label=m, lw=2.2 if m == "RCF-DR-VR-SN" else 1.2)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            if hline is not None:
                ax.axhline(hline, color="k", ls=":", lw=1)
            ax.set_title(f"{scen} - {ov} overlap", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("n")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def figures(S, SEL):
    _grid(S, "rmse", "aslib_rmse_primary.png", "RMSE (target policy)")
    _grid(S, "coverage", "aslib_coverage_primary.png", "95% Wald coverage", logy=False, hline=0.95)
    _grid(SEL.rename(columns={"regret": "rmse"}), "rmse", "aslib_regret_primary.png",
          "mean selection regret", logy=False)


def md_tables():
    """Markdown tables used in docs/aslib/ASLIB_RESULTS.md (written to results/aslib/tables/aslib_md_tables.md)."""
    S = pd.read_csv(TAB / "aslib_target_summary.csv")
    SEL = pd.read_csv(TAB / "aslib_selection_summary.csv")
    DG = pd.read_csv(TAB / "aslib_weight_diagnostics.csv")
    ms = ["DM", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
    out = []
    for scen in SCEN:
        x = S[(S.scenario == scen) & S.variant.str.contains("primary", regex=False)]
        r = x.pivot_table(index=["overlap", "n"], columns="method", values="rmse")[ms]
        c = x.pivot_table(index=["overlap", "n"], columns="method", values="coverage")[["DR", "RCF-DR-os", "RCF-DR-VR-SN", "SNIPW"]]
        b = x.pivot_table(index=["overlap", "n"], columns="method", values="bias")[["RCF-DR-os", "RCF-DR-VR-SN"]]
        d = DG[(DG.scenario == scen) & (DG["pop"] == "eval")].set_index(["overlap", "n"])
        out.append(f"\n### {scen}: target policy, primary reward ({PRIMARY[scen]})\n")
        out.append("| overlap | n | ESS/n (med) | Hill α̂ (med) | " + " | ".join(f"RMSE {m}" for m in ms) +
                   " | cov DR | cov os | cov VR-SN | cov SNIPW | bias os | bias VR-SN |")
        out.append("|" + "---|" * (4 + len(ms) + 6))
        for ov in OV:
            for n in [250, 500, 1000, 2000, 4000]:
                k = (ov, n)
                out.append(f"| {ov} | {n} | {d.loc[k, 'ess_frac']:.3f} | {d.loc[k, 'alpha_hat']:.2f} | " +
                           " | ".join(f"{r.loc[k, m]:.3f}" for m in ms) + " | " +
                           " | ".join(f"{c.loc[k, m]:.2f}" for m in ["DR", "RCF-DR-os", "RCF-DR-VR-SN", "SNIPW"]) + " | " +
                           " | ".join(f"{b.loc[k, m]:+.3f}" for m in ["RCF-DR-os", "RCF-DR-VR-SN"]) + " |")
        y = SEL[(SEL.scenario == scen) & SEL.variant.str.contains("primary", regex=False)]
        rg = y.pivot_table(index=["overlap", "n"], columns="method", values="regret")[ms]
        pb = y.pivot_table(index=["overlap", "n"], columns="method", values="p_best")[ms]
        out.append(f"\n### {scen}: policy selection over 9 candidates, primary reward — mean regret (P(best))\n")
        out.append("| overlap | n | " + " | ".join(ms) + " |")
        out.append("|" + "---|" * (2 + len(ms)))
        for ov in OV:
            for n in [250, 500, 1000, 2000, 4000]:
                k = (ov, n)
                out.append(f"| {ov} | {n} | " + " | ".join(f"{rg.loc[k, m]:.3f} ({pb.loc[k, m]:.2f})" for m in ms) + " |")
    (TAB / "aslib_md_tables.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "md":
        md_tables()
    else:
        main()
