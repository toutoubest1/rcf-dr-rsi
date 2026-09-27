"""ASlib design: data preparation, rewards, timeout conventions, policies, outcome-free overlap calibration.

Everything in this module was fixed in docs/aslib/PREREGISTRATION.md before any estimator was run on ASlib.
No estimator is imported here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from real_data.aslib.aslib_io import load_scenario  # noqa: E402

SPLIT_SEED = 20260925          # 30/70 policy-training / evaluation split
RT_FLOOR = 0.01                # seconds; timer resolution (SAT03 records 0.001 s steps)
N_PC = 10                      # PCs used by the reward models (estimation side)
N_PC_LEGACY = 3                # PCs used by the legacy (logging) selector
TARGET_TEMP = 3.0              # target policy softmax(3 * z(u_hat))
ALPHAS = np.logspace(-2, 4, 25)
PRIMARY_CONVENTION = {"SAT03-16_INDU": "PAR10",   # description.txt: performance_measures [PAR10]
                      "ASP-POTASSCO": "PAR1"}      # description.txt: performance_measures [runtime];
                                                   # timeouts are recorded at the cutoff (600 s)
ESS_TARGETS_PLANNED = {"good": 0.60, "moderate": 0.25, "poor": 0.10}   # not attainable, see docs/aslib/PREREGISTRATION.md
# final: good = uniform logging (kappa = 0); moderate / poor = smallest kappa with population ESS/n <= target
ESS_TARGETS_FINAL = {"good": None, "moderate": 0.05, "poor": 0.01}
SCEN_ID = {"SAT03-16_INDU": 1, "ASP-POTASSCO": 2}


# ------------------------------------------------------------------ data
def load(scenario):
    desc, runs, feats = load_scenario(scenario)
    col = "runtime" if "runtime" in runs.columns else "PAR10"
    runs = runs.rename(columns={col: "runtime"})
    runs = runs[runs["repetition"] == 1]
    cutoff = float(desc["algorithm_cutoff_time"])
    algs = sorted(runs["algorithm"].unique())
    insts = sorted(runs["instance_id"].unique())
    M = runs.pivot_table(index="instance_id", columns="algorithm", values="runtime", aggfunc="first").reindex(
        index=insts, columns=algs)
    S = runs.pivot_table(index="instance_id", columns="algorithm", values="runstatus", aggfunc="first").reindex(
        index=insts, columns=algs)
    ok = (S == "ok").values
    rt = M.values.astype(float)
    fcols = [c for c in feats.columns if c not in ("instance_id", "repetition")]
    F = feats[feats["repetition"] == feats["repetition"].min()].set_index("instance_id").reindex(insts)[fcols]
    return dict(scenario=scenario, algs=algs, insts=np.array(insts), cutoff=cutoff, rt=rt, ok=ok,
                F=F.values.astype(float))


def runtime_matrix(d, convention):
    """PAR1: unsolved -> cutoff; PAR10: unsolved -> 10 * cutoff. Solved runtimes floored at RT_FLOOR."""
    fac = {"PAR1": 1.0, "PAR10": 10.0}[convention]
    solved = np.maximum(d["rt"], RT_FLOOR)
    return np.where(d["ok"], solved, fac * d["cutoff"])


def reward_matrix(d, reward):
    """reward in {'logPAR10', 'logPAR1', 'speed'}; larger is better."""
    if reward == "logPAR10":
        return -np.log(runtime_matrix(d, "PAR10"))
    if reward == "logPAR1":
        return -np.log(runtime_matrix(d, "PAR1"))
    if reward == "speed":   # VBS-normalised speed: rt_VBS / rt_a if solved, 0 if unsolved
        rt = np.where(d["ok"], np.maximum(d["rt"], RT_FLOOR), np.inf)
        vbs = rt.min(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            return np.where(d["ok"], vbs / rt, 0.0)
    raise ValueError(reward)


def features(d):
    """Unsupervised: signed log1p, median imputation, drop constant columns, standardise, PCA on all
    instances; the returned PC scores are standardised to unit variance. No outcome information is used."""
    F = d["F"].copy()
    F = np.sign(F) * np.log1p(np.abs(F))
    med = np.nanmedian(F, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    F = np.where(np.isnan(F), med, F)
    sd = F.std(0)
    F = F[:, sd > 1e-10]
    Fs = (F - F.mean(0)) / F.std(0)
    pcs = PCA(n_components=N_PC, random_state=0).fit_transform(Fs)
    pcs = pcs / pcs.std(0)
    return Fs, pcs


def split(n):
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n)
    k = int(round(0.3 * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


# ------------------------------------------------------------------ policies
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


class Design:
    """All policies of the pre-registered design for one scenario. Policies are trained once, on the
    policy-training split with the scenario's primary reward, and are then fixed functions of x."""

    def __init__(self, scenario):
        self.d = load(scenario)
        self.scenario = scenario
        self.K = len(self.d["algs"])
        self.Fs, self.pcs = features(self.d)
        self.tr, self.ev = split(len(self.d["insts"]))
        self.primary = "logPAR10" if PRIMARY_CONVENTION[scenario] == "PAR10" else "logPAR1"
        Y = reward_matrix(self.d, self.primary)
        Ytr = Y[self.tr]
        # rich selector: ridge on all transformed features (target + candidates)
        self.u_hat = ridge_scores(self.Fs[self.tr], Ytr, self.Fs)
        # legacy selector: ridge on the first 3 PCs (logging policy)
        self.g_hat = ridge_scores(self.pcs[self.tr, :N_PC_LEGACY], Ytr, self.pcs[:, :N_PC_LEGACY])
        # random-forest greedy selector
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33, random_state=0,
                                   n_jobs=2).fit(self.Fs[self.tr], Ytr)
        self.rf_hat = rf.predict(self.Fs)
        mean_tr = Ytr.mean(0)
        order = np.argsort(-mean_tr)
        self.sbs, self.sbs2 = int(order[0]), int(order[1])

    # target policy
    def target(self):
        return softmax(TARGET_TEMP * zrow(self.u_hat))

    def logging(self, kappa):
        return softmax(kappa * zrow(self.g_hat))

    def candidates(self):
        n, K = self.u_hat.shape
        zu = zrow(self.u_hat)
        return {
            "SBS": onehot(np.full(n, self.sbs), K),
            "SBS-2": onehot(np.full(n, self.sbs2), K),
            "Uniform": np.full((n, K), 1.0 / K),
            "Legacy-greedy": onehot(self.g_hat.argmax(1), K),
            "Ridge-sm1": softmax(1.0 * zu),
            "Ridge-sm3 (target)": softmax(3.0 * zu),
            "Ridge-sm10": softmax(10.0 * zu),
            "Ridge-greedy": onehot(self.u_hat.argmax(1), K),
            "RF-greedy": onehot(self.rf_hat.argmax(1), K),
        }


def pop_ess(pi_e, pi_b):
    """Population ESS/n of the importance weight for pi_e under pi_b over the rows given:
    (E w)^2 / E w^2 with E w = 1 and E w^2 = mean_x sum_a pi_e^2 / pi_b. Outcome-free."""
    return float(1.0 / np.mean((pi_e ** 2 / pi_b).sum(1)))


def calibrate_kappa(pi_e, g, target, grid=np.round(np.arange(0, 40.001, 0.05), 3)):
    """Smallest kappa >= 0 whose population ESS/n on the policy-training split is <= target (outcome-free)."""
    vals = [pop_ess(pi_e, softmax(k * zrow(g))) for k in grid]
    for k, v in zip(grid, vals):
        if v <= target:
            return float(k), v, (grid, np.array(vals))
    return np.nan, np.nan, (grid, np.array(vals))
