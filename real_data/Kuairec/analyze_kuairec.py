"""KuaiRec summaries from the saved replications (no estimation)"""
import numpy as np
import pandas as pd

from real_data.kuairec.paths import RAW, TAB

TARGET = "Ridge-sm3 (target)"
METHODS = ["DM", "IPW", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR", "RCF-DR-VR-SN"]
PRIMARY = "log1p_wr|huber"


def load():
    return pd.concat([pd.read_pickle(f) for f in sorted(RAW.glob("cell_*.pkl.gz"))], ignore_index=True)


def summarise(g):
    e = g.est - g.truth
    lo, hi = g.est - 1.96 * g.se, g.est + 1.96 * g.se
    return pd.Series(dict(reps=len(g), bias=e.mean(), sd=g.est.std(), rmse=np.sqrt((e ** 2).mean()),
                          q99_abs_err=np.quantile(np.abs(e), .99), coverage=((lo <= g.truth) & (g.truth <= hi)).mean(),
                          ci_width=(hi - lo).mean(), frac_huber=g.frac_huber.mean(), frac_clipped=g.frac_clipped.mean(),
                          tau=g.tau.mean(), resid_mad=g.resid_mad.mean(), ess_frac=g.ess_frac.median(),
                          max_w=g.max_w.median(), alpha_hat=g.alpha_hat.median(),
                          frac_nonrootn=np.mean(g.alpha_hat <= 2), frac_capped_vrsn=g.frac_capped_vrsn.mean()))


def selection(g, eps=0.001):
    truths = g.groupby("policy").truth.first()
    vbest = truths.max()
    out = []
    for m, h in g.groupby("method"):
        piv = h.pivot_table(index="rep", columns="policy", values="est")
        ch = piv.idxmax(axis=1)
        reg = vbest - truths.loc[ch].values
        out.append(dict(method=m, regret=reg.mean(), p_eps_best=np.mean(reg <= eps),
                        winners_curse=np.mean(piv.max(axis=1).values - truths.loc[ch].values)))
    return pd.DataFrame(out)


def main():
    df = load()
    t = df[df.policy == TARGET]
    s = t.groupby(["variant", "overlap", "n", "method"]).apply(summarise, include_groups=False).reset_index()
    s.insert(0, "dataset", "kuairec")
    sel = df.groupby(["variant", "overlap", "n"]).apply(selection, include_groups=False).reset_index()
    sel = sel.drop(columns=[c for c in sel.columns if c.startswith("level_")])
    sel.insert(0, "dataset", "kuairec")
    TAB.mkdir(parents=True, exist_ok=True)
    s.to_csv(TAB / "kuai_target_summary.csv", index=False)
    sel.to_csv(TAB / "kuai_selection_summary.csv", index=False)
    return s, sel


if __name__ == "__main__":
    main()
