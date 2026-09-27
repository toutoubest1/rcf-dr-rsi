"""KuaiRec RMSE and 95% Wald coverage versus n for the target policy, primary reward
log(1 + watch ratio), Huber reward model; columns = uniform / moderate / poor overlap.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from real_data.kuairec.paths import FIG, TAB

out = Path(sys.argv[1]) if len(sys.argv) > 1 else FIG
out.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(TAB / "kuai_target_summary.csv")
S = S[(S.dataset == "kuairec") & (S.variant == "log1p_wr|huber")]
SHOW = ["DM", "SNIPW", "DR", "DR-clip", "RCF-DR-os", "RCF-DR-VR-SN"]
STY = {"DM": ("tab:gray", "s"), "SNIPW": ("tab:olive", "^"), "DR": ("tab:red", "o"), "DR-clip": ("tab:orange", "v"),
       "RCF-DR-os": ("tab:purple", "D"), "RCF-DR-VR-SN": ("tab:blue", "*")}
COLS = [("good", "Uniform logging (ESS/$n$ = 0.30)"), ("moderate", "Moderate overlap (ESS/$n$ = 0.046)"),
        ("poor", "Poor overlap (ESS/$n$ = 0.009)")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
for j, (ov, title) in enumerate(COLS):
    g = S[S.overlap == ov]
    for i, (col, lab) in enumerate([("rmse", "RMSE"), ("coverage", "95% Wald coverage")]):
        ax = axes[i, j]
        for m in SHOW:
            h = g[g.method == m].sort_values("n")
            c, mk = STY[m]
            vs = m == "RCF-DR-VR-SN"
            ax.plot(h.n, h[col], marker=mk, color=c, lw=2.2 if vs else 1.2, ms=8 if vs else 5, label=m, zorder=3 if vs else 2)
        ax.set_xscale("log")
        if col == "rmse":
            ax.set_yscale("log")
            ax.set_title(title, fontsize=9.5)
        else:
            ax.axhline(0.95, color="k", ls=":", lw=1)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("$n$")
        if j == 0:
            ax.set_ylabel(lab)
        ax.set_xticks([250, 500, 1000, 2000, 4000], ["250", "500", "1000", "2000", "4000"])
        ax.minorticks_off() if col == "coverage" else None
axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(out / "kuairec_rmse_coverage.pdf", bbox_inches="tight")
fig.savefig(out / "kuairec_rmse_coverage.png", dpi=200, bbox_inches="tight")
