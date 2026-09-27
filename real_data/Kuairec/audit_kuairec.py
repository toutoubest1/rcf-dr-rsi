"""KuaiRec data audit (no estimator): matrix density, missing cells, reward and residual diagnostics (Table S35)."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from real_data.kuairec.paths import DATA_RED as D, TAB


def mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def desc(x, name):
    q = np.quantile(x, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    return {f"{name}_frac_zero": float(np.mean(x == 0)), f"{name}_median": float(np.median(x)), f"{name}_mad": mad(x),
            f"{name}_mean": float(x.mean()), f"{name}_sd": float(x.std()), f"{name}_skew": float(skew(x)),
            f"{name}_exkurt": float(kurtosis(x)),
            f"{name}_q": dict(zip(["1", "5", "10", "25", "50", "75", "90", "95", "99", "99.9"], map(float, q))),
            f"{name}_max": float(x.max())}


if __name__ == "__main__":
    z = np.load(D / "small_matrix.npz")
    S = pd.DataFrame({k: z[k] for k in z.files})
    out = {"rows": len(S), "users": int(S.user_id.nunique()), "items": int(S.video_id.nunique())}
    out["cells"] = out["users"] * out["items"]
    out["duplicates_user_video"] = int(S.duplicated(["user_id", "video_id"]).sum())
    out["density"] = len(S.drop_duplicates(["user_id", "video_id"])) / out["cells"]
    out["missing_cells"] = out["cells"] - len(S.drop_duplicates(["user_id", "video_id"]))
    per_user_missing = S.groupby("user_id").video_id.nunique().rsub(out["items"])
    out["missing_per_user_q"] = dict(zip(["50", "90", "99", "max"], map(float, np.quantile(per_user_missing, [.5, .9, .99, 1]))))
    out["timestamp_missing"] = int(S.timestamp.isna().sum())
    wr = S.watch_ratio.values.astype(float)
    ps = S.play_duration.values / 1000.0
    out["video_duration_zero"] = int((S.video_duration <= 0).sum())
    out.update(desc(wr, "watch_ratio"))
    out.update(desc(np.log1p(wr), "log1p_watch_ratio"))
    out.update(desc(ps, "play_time_s"))
    out.update(desc(np.log1p(ps), "log1p_play_time_s"))
    # categories
    C = pd.read_csv(D / "item_categories.csv")
    C["feat"] = C.feat.apply(ast.literal_eval)
    C["first"] = C.feat.str[0]
    items = np.unique(S.video_id)
    Cs = C.set_index("video_id").loc[items]
    out["n_categories_multi"] = int(len(set(c for f in Cs.feat for c in f)))
    out["first_category_counts_small_items"] = {int(k): int(v) for k, v in Cs["first"].value_counts().items()}
    out["mean_labels_per_item"] = float(Cs.feat.str.len().mean())
    # user-level and item-level variance decomposition of log1p(watch_ratio) (two-way additive fit)
    y = np.log1p(wr)
    u = S.user_id.values
    v = S.video_id.values
    mu = y.mean()
    ue = pd.Series(y).groupby(u).transform("mean").values - mu
    r1 = y - mu - ue
    ve = pd.Series(r1).groupby(v).transform("mean").values
    res = r1 - ve
    out["additive_resid_mad"] = mad(res)
    out["additive_resid_sd"] = float(res.std())
    out["additive_resid_skew"] = float(skew(res))
    out["additive_resid_exkurt"] = float(kurtosis(res))
    q1, q2, q3 = np.quantile(res, [.25, .5, .75])
    out["additive_resid_bowley"] = float((q3 + q1 - 2 * q2) / (q3 - q1))
    out["var_share_user"] = float(ue.var() / y.var())
    out["var_share_item"] = float(ve.var() / y.var())
    # same for raw watch_ratio
    ue2 = pd.Series(wr).groupby(u).transform("mean").values - wr.mean()
    r2 = wr - wr.mean() - ue2
    res2 = r2 - pd.Series(r2).groupby(v).transform("mean").values
    q1, q2, q3 = np.quantile(res2, [.25, .5, .75])
    out.update(raw_additive_resid_mad=mad(res2), raw_additive_resid_skew=float(skew(res2)),
               raw_additive_resid_exkurt=float(kurtosis(res2)), raw_additive_resid_bowley=float((q3 + q1 - 2 * q2) / (q3 - q1)))
    zb = np.load(D / "big_matrix_smallusers.npz")
    out["big_rows_small_users"] = int(len(zb["user_id"]))
    out["big_overlap_pairs_with_small"] = int(len(pd.MultiIndex.from_arrays([zb["user_id"], zb["video_id"]]).intersection(
        pd.MultiIndex.from_arrays([S.user_id.values, S.video_id.values]))))
    U = pd.read_csv(D / "user_features.csv")
    out["user_feature_columns"] = list(U.columns)
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(TAB / "kuairec_audit.json", "w"), indent=1, default=str)
    for k, val in out.items():
        print(k, ":", val)
