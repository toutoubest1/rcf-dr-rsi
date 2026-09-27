"""KuaiRec design: data, rewards, category actions, missing-cell rule, user split, features, PCA, policies.

Fixed in docs/kuairec/PREREGISTRATION.md before any estimator was run.  No estimator is used here.
Input: data/kuairec/reduced/{small_matrix.npz, big_matrix_smallusers.npz, item_categories.csv,
user_features.csv}, produced by real_data/kuairec/reduce_kuairec.py from the official KuaiRec.zip.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"   # KuaiRec reduced files are in data/kuairec/reduced/
SPLIT_SEED = 20260925
N_PC, N_PC_LEGACY = 10, 3
TARGET_TEMP = 3.0
ALPHAS = np.logspace(-2, 4, 25)
KUAIREC_MIN_ITEMS = 50          # categories with >= 50 small-matrix videos kept, the rest pooled into "other"


# ------------------------------------------------------------------ shared helpers
def softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def zrow(u):
    return (u - u.mean(1, keepdims=True)) / (u.std(1, keepdims=True) + 1e-12)


def onehot(idx, K):
    P = np.zeros((len(idx), K))
    P[np.arange(len(idx)), idx] = 1.0
    return P


def ridge_scores(Xtr, Ytr, X):
    return np.column_stack([RidgeCV(alphas=ALPHAS).fit(Xtr, Ytr[:, a]).predict(X) for a in range(Ytr.shape[1])])


def pop_ess(pi_e, pi_b):
    return float(1.0 / np.mean((pi_e ** 2 / pi_b).sum(1)))


def calibrate_kappa(pi_e, g, target, grid=np.round(np.arange(0, 40.001, 0.05), 3)):
    for k in grid:
        v = pop_ess(pi_e, softmax(k * zrow(g)))
        if v <= target:
            return float(k), v
    return np.nan, np.nan


def split_users(n):
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    k = int(round(0.3 * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def user_static_features(U):
    """Numeric user features: signed log1p of counts/days, binary flags, one-hot of user_active_degree and of the
    encrypted categorical features with <= 50 levels (high-cardinality onehot_feat3/7/8 dropped). '_range' columns
    are dropped (redundant with the numeric counts)."""
    X = pd.DataFrame(index=U.index)
    for c in ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]:
        X[c] = np.log1p(U[c].astype(float).clip(lower=0))
    for c in ["is_lowactive_period", "is_live_streamer", "is_video_author"]:
        X[c] = U[c].astype(float)
    X = X.join(pd.get_dummies(U["user_active_degree"].astype(str), prefix="act", dtype=float))
    for j in range(18):
        c = f"onehot_feat{j}"
        v = U[c]
        if v.nunique() <= 50:
            X = X.join(pd.get_dummies(v.fillna(-1).astype(int).astype(str), prefix=c, dtype=float))
    return X.fillna(0.0)


def pcs_of(F, n_pc=N_PC):
    F = F.values.astype(float)
    sd = F.std(0)
    F = F[:, sd > 1e-10]
    Fs = (F - F.mean(0)) / F.std(0)
    pcs = PCA(n_components=n_pc, random_state=0).fit_transform(Fs)
    return Fs, pcs / pcs.std(0)


def candidates(u_hat, g_hat, rf_hat, sbs, sbs2):
    n, K = u_hat.shape
    zu = zrow(u_hat)
    return {
        "Best-single": onehot(np.full(n, sbs), K),
        "Best-single-2": onehot(np.full(n, sbs2), K),
        "Uniform": np.full((n, K), 1.0 / K),
        "Legacy-greedy": onehot(g_hat.argmax(1), K),
        "Ridge-sm1": softmax(1.0 * zu),
        "Ridge-sm3 (target)": softmax(3.0 * zu),
        "Ridge-sm10": softmax(10.0 * zu),
        "Ridge-greedy": onehot(u_hat.argmax(1), K),
        "RF-greedy": onehot(rf_hat.argmax(1), K),
    }


# ------------------------------------------------------------------ KuaiRec
def kuairec_reward(wr, reward):
    return {"log1p_wr": np.log1p(wr), "wr": wr}[reward]


class KuaiRec:
    """Semi-synthetic design on the (99.6%-observed) small matrix.
    Context = user; action = video category (first label; small categories pooled); when category c is chosen for
    user u, a video is drawn uniformly from the category-c videos OBSERVED for u (missing cells are never imputed:
    the within-category draw and the ground truth both use only observed videos)."""

    def __init__(self):
        z = np.load(DATA / "kuairec" / "reduced" / "small_matrix.npz")
        S = pd.DataFrame({"user_id": z["user_id"], "video_id": z["video_id"], "wr": z["watch_ratio"].astype(float)})
        C = pd.read_csv(DATA / "kuairec" / "reduced" / "item_categories.csv")
        C["first"] = C.feat.apply(ast.literal_eval).str[0]
        items = np.unique(S.video_id)
        first = C.set_index("video_id").loc[items, "first"]
        cnt = first.value_counts()
        keep = sorted(cnt[cnt >= KUAIREC_MIN_ITEMS].index)
        self.cat_names = [f"cat{c}" for c in keep] + ["other"]
        cmap = {c: i for i, c in enumerate(keep)}
        self.item_cat = first.map(lambda c: cmap.get(c, len(keep)))
        self.K = len(self.cat_names)
        S["cat"] = self.item_cat.loc[S.video_id].values
        self.users = np.unique(S.user_id)
        uidx = pd.Series(np.arange(len(self.users)), index=self.users)
        S["u"] = uidx.loc[S.user_id].values
        # per (user, category) index ranges into a sorted array of observed rewards
        S = S.sort_values(["u", "cat", "video_id"]).reset_index(drop=True)
        self.S = S
        key = S.u.values * self.K + S.cat.values
        self.start = np.searchsorted(key, np.arange(len(self.users) * self.K))
        self.stop = np.searchsorted(key, np.arange(len(self.users) * self.K), side="right")
        self.cellsize = (self.stop - self.start).reshape(len(self.users), self.K)
        assert self.cellsize.min() > 0
        self.wr = S.wr.values
        self.tr, self.ev = split_users(len(self.users))
        self._features()
        self.cell_mean = {r: self._cell_mean(r) for r in ("log1p_wr", "wr")}
        self._policies()

    def _cell_mean(self, reward):
        y = kuairec_reward(self.wr, reward)
        cs = np.concatenate([[0.0], np.cumsum(y)])
        return ((cs[self.stop] - cs[self.start]) / (self.stop - self.start)).reshape(len(self.users), self.K)

    def _features(self):
        U = pd.read_csv(DATA / "kuairec" / "reduced" / "user_features.csv").set_index("user_id").loc[self.users]
        X = user_static_features(U)
        zb = np.load(DATA / "kuairec" / "reduced" / "big_matrix_smallusers.npz")
        B = pd.DataFrame({"user_id": zb["user_id"], "video_id": zb["video_id"], "y": np.log1p(zb["watch_ratio"].astype(float))})
        C = pd.read_csv(DATA / "kuairec" / "reduced" / "item_categories.csv")
        C["first"] = C.feat.apply(ast.literal_eval).str[0]
        keep = [int(c[3:]) for c in self.cat_names[:-1]]
        cmap = {c: i for i, c in enumerate(keep)}
        B["cat"] = C.set_index("video_id").loc[B.video_id, "first"].map(lambda c: cmap.get(c, len(keep))).values
        g = B.groupby(["user_id", "cat"]).y
        mean = g.mean().unstack().reindex(index=self.users, columns=range(self.K))
        share = g.size().unstack().reindex(index=self.users, columns=range(self.K)).fillna(0)
        share = share.div(share.sum(1).replace(0, 1), axis=0)
        overall = B.groupby("user_id").y.mean().reindex(self.users)
        mean = mean.apply(lambda col: col.fillna(overall)).fillna(overall.mean())
        H = pd.concat([mean.add_prefix("hist_mean_"), share.add_prefix("hist_share_")], axis=1)
        H["hist_overall"] = overall.fillna(overall.mean()).values
        H["hist_logn"] = np.log1p(B.groupby("user_id").size().reindex(self.users).fillna(0)).values
        self.hist_share = share.values          # production-recommender exposure share per (user, category), big matrix
        H.index = X.index
        self.Fs, self.pcs = pcs_of(pd.concat([X, H], axis=1))

    def _policies(self):
        Ytr = self.cell_mean["log1p_wr"][self.tr]
        self.u_hat = ridge_scores(self.Fs[self.tr], Ytr, self.Fs)
        self.g_hat = ridge_scores(self.pcs[self.tr, :N_PC_LEGACY], Ytr, self.pcs[:, :N_PC_LEGACY])
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33, random_state=0,
                                   n_jobs=2).fit(self.Fs[self.tr], Ytr)
        self.rf_hat = rf.predict(self.Fs)
        order = np.argsort(-Ytr.mean(0))
        self.sbs, self.sbs2 = int(order[0]), int(order[1])
        self.target = softmax(TARGET_TEMP * zrow(self.u_hat))
        self.cands = candidates(self.u_hat, self.g_hat, self.rf_hat, self.sbs, self.sbs2)

    def logging(self, kappa):
        """Legacy 'exposure-following' recommender: softmax(kappa * z(log(share + 0.01))), where share is the
        user's category exposure share under the production recommender (big matrix, disjoint from the small
        matrix). kappa = 0 is uniform logging. Outcome-free."""
        return softmax(kappa * zrow(np.log(self.hist_share + 0.01)))

    def truth(self, P, reward):
        return float((P[self.ev] * self.cell_mean[reward][self.ev]).sum(1).mean())

    def draw_rewards(self, users, cats, rng, reward):
        """Uniform draw of an observed video within (user, category); returns the reward."""
        idx = users * self.K + cats
        lo, hi = self.start[idx], self.stop[idx]
        j = lo + np.floor(rng.random(len(users)) * (hi - lo)).astype(int)
        return kuairec_reward(self.wr[j], reward)
