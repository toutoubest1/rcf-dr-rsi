"""Nuisance estimation: outcome regression m(x, a) and behaviour propensity pi_b(a|x).

All nuisances can be produced either
  * cross-fitted (K-fold): predictions for fold k come from models trained on
    the other K-1 folds, so the model used for observation i never saw i;
  * in-sample: a single model fit on all data (ablation for cross-fitting).

Specifications
--------------
outcome 'correct'   : per-action regression on the exact feature map of mu_a
outcome 'mis'       : per-action linear regression on x only (omits all
                      nonlinear terms: sin, interaction, action-specific h_a)
outcome 'correct_huber' / 'mis_huber' : same features, Huber-loss IRLS fit
                      (the reward models used in the paper)
propensity 'correct': multinomial logistic on [x, x0*x1, x2^2, sin(2 x8)]
                      (exactly the behaviour logit family)
propensity 'mis'    : multinomial logistic on x only
propensity 'oracle' : the true logging probabilities (in RSI replay the
                      exploration policy is code, so pi_b is often known)
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold

from .dgp import K

PROP_FLOOR = 1e-3
PROP_C = None                      # None -> choose by CV
PROP_CS = [0.03, 0.1, 0.3, 1.0, 3.0, 30.0]


# ----------------------------------------------------------------- features
def outcome_features(X: np.ndarray, spec: str) -> np.ndarray:
    base = spec.replace("_huber", "")
    if base == "correct":
        return np.column_stack([
            X, np.sin(X[:, 0]), X[:, 1] * X[:, 2], np.cos(X[:, 3]), X[:, 4] ** 2,
            np.tanh(X[:, 5] * X[:, 6]), np.abs(X[:, 7])])
    if base == "mis":
        return X
    raise ValueError(spec)


def propensity_features(X: np.ndarray, spec: str) -> np.ndarray:
    if spec == "correct":
        return np.column_stack([X, X[:, 0] * X[:, 1], X[:, 2] ** 2, np.sin(2 * X[:, 8])])
    if spec == "mis":
        return X
    if spec == "correct_h":
        # correct for the h-logger: behaviour features + action-specific reward nonlinearities
        return np.column_stack([X, X[:, 0] * X[:, 1], X[:, 2] ** 2, np.sin(2 * X[:, 8]),
                                np.cos(X[:, 3]), X[:, 4] ** 2, np.tanh(X[:, 5] * X[:, 6]), np.abs(X[:, 7])])
    if spec == "mis_strong":
        # severe misspecification (strong propensity misspecification supplement) -- only the first 4 covariates, no nonlinear terms
        return X[:, :4]
    raise ValueError(spec)


# ------------------------------------------------------------ regressions
def _ols(Z: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    G = Z1.T @ Z1 + ridge * np.eye(Z1.shape[1])
    return np.linalg.solve(G, Z1.T @ y)


def _huber_irls(Z: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 50) -> np.ndarray:
    """Huber M-regression by IRLS with MAD-rescaled residuals (scale re-estimated each step)."""
    b = _ols(Z, y)
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    for _ in range(iters):
        r = y - Z1 @ b
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        u = np.abs(r) / (k * s)
        wts = np.where(u <= 1, 1.0, 1.0 / u)
        G = Z1.T @ (Z1 * wts[:, None]) + 1e-6 * np.eye(Z1.shape[1])
        b_new = np.linalg.solve(G, Z1.T @ (wts * y))
        if np.max(np.abs(b_new - b)) < 1e-7:
            b = b_new
            break
        b = b_new
    return b


def _predict(Z, b):
    return b[0] + Z @ b[1:]


def fit_outcome(X, A, Y, spec):
    Z = outcome_features(X, spec)
    robust = spec.endswith("_huber")
    coefs = []
    for a in range(K):
        idx = A == a
        if idx.sum() < Z.shape[1] + 2:  # too few observations -> pooled fit
            idx = np.ones_like(A, dtype=bool)
        coefs.append(_huber_irls(Z[idx], Y[idx]) if robust else _ols(Z[idx], Y[idx]))
    return coefs


def predict_outcome(coefs, X, spec):
    Z = outcome_features(X, spec)
    return np.column_stack([_predict(Z, coefs[a]) for a in range(K)])


def fit_propensity(X, A, spec):
    Z = propensity_features(X, spec)
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    # NOTE (log #1): an (almost) unpenalised fit (C=1e4) gave over-confident
    # out-of-fold probabilities and mean importance weights 5-20% above 1
    # -> upward IS bias.  The ridge penalty is now chosen by 3-fold CV on
    # log-loss inside each training fold.
    if PROP_C is None:
        clf = LogisticRegressionCV(Cs=PROP_CS, cv=3, scoring="neg_log_loss", max_iter=2000)
    else:
        clf = LogisticRegression(C=PROP_C, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit((Z - mu) / sd, A)
    return clf, mu, sd


def predict_propensity(model, X, spec):
    clf, mu, sd = model
    Z = propensity_features(X, spec)
    p = np.zeros((len(X), K))
    p[:, clf.classes_] = clf.predict_proba((Z - mu) / sd)
    p = np.maximum(p, PROP_FLOOR)
    return p / p.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ cross-fitting
def forward_outcome(X, A, Y, spec, batch, first="kfold2", seed=0):
    """Chronological ('forward') fitting for batched adaptive logs.

    For batch b >= 1 the reward model is fit on batches 0..b-1 only, so the prediction used for an
    observation in batch b is measurable w.r.t. the past history H_{b-1} (Section 3.7 of the paper).
    Batch 0 has no past:  first='zero'   -> mhat = 0 (strictly predictable; the DR score is then IS);
                          first='kfold2' -> 2-fold cross-fitting within batch 0 (not predictable, but
                                            only for the first batch).
    """
    n = len(Y)
    pred = np.zeros((n, K))
    ub = np.unique(batch)
    for j, b in enumerate(ub):
        te = np.where(batch == b)[0]
        if j == 0:
            if first == "kfold2":
                kf = KFold(n_splits=2, shuffle=True, random_state=seed)
                for tr_i, te_i in kf.split(te):
                    tr, tt = te[tr_i], te[te_i]
                    pred[tt] = predict_outcome(fit_outcome(X[tr], A[tr], Y[tr], spec), X[tt], spec)
            continue
        tr = np.where(np.isin(batch, ub[:j]))[0]
        pred[te] = predict_outcome(fit_outcome(X[tr], A[tr], Y[tr], spec), X[te], spec)
    return pred


def crossfit_nuisances(X, A, Y, outcome_specs=("correct", "mis"), prop_specs=("correct", "mis"),
                       n_folds: int = 5, seed: int = 0, insample: bool = True, pb_true=None,
                       forward=(), batch=None):
    """Return dict with keys ('m', spec, cf) -> n x K and ('pb', spec, cf) -> n x K.

    cf=True: out-of-fold predictions; cf=False: in-sample fit.
    """
    n = len(Y)
    out = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(X))
    for spec in outcome_specs:
        pred = np.empty((n, K))
        for tr, te in folds:
            pred[te] = predict_outcome(fit_outcome(X[tr], A[tr], Y[tr], spec), X[te], spec)
        out[("m", spec, True)] = pred
        if insample:
            out[("m", spec, False)] = predict_outcome(fit_outcome(X, A, Y, spec), X, spec)
    for spec in prop_specs:
        pred = np.empty((n, K))
        for tr, te in folds:
            pred[te] = predict_propensity(fit_propensity(X[tr], A[tr], spec), X[te], spec)
        out[("pb", spec, True)] = pred
        if insample:
            out[("pb", spec, False)] = predict_propensity(fit_propensity(X, A, spec), X, spec)
    for scheme in forward:
        first = "zero" if scheme == "fwd0" else "kfold2"
        for spec in outcome_specs:
            out[("m", spec, scheme)] = forward_outcome(X, A, Y, spec, batch, first=first, seed=seed)
    if pb_true is not None:
        out[("pb", "oracle", True)] = out[("pb", "oracle", False)] = pb_true
    return out
