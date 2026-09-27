"""Overlap diagnostic for replay-based OPE (Section 2.5 of the paper).

The theory (Section 3.5 of the paper) shows that the tail index alpha of the importance
weights, P(w > t) ~ t^-alpha, determines whether root-n Wald inference is defensible:
  alpha > 2   : E[w^2] < inf, so the DR score can have finite variance (root-n possible);
  alpha = 2   : log-divergent variance; the best rate is sqrt(log n / n) and bias/SE stays O(1)
                for any cap;
  alpha < 2   : infinite variance; slower than root-n.
The categories below are PRACTICAL WARNING LEVELS, not theorems. alpha_hat is noisy
(sd ~ alpha/sqrt(k)), so a buffer above 2 is used for "stable".

  'stable'      alpha_hat > 2.5
  'borderline'  2 < alpha_hat <= 2.5
  'non-root-n'  alpha_hat <= 2   -> Wald intervals may be unreliable (infinite/borderline variance)
"""
from __future__ import annotations

import numpy as np


def hill_alpha(w: np.ndarray, k: int | None = None) -> tuple[float, float, int]:
    """Hill estimator of the upper tail index of w from the top-k order statistics.
    Default k = ceil(2 sqrt(n)). Returns (alpha_hat, approximate SE = alpha_hat / sqrt(k), k)."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    if k is None:
        k = int(np.ceil(2 * np.sqrt(n)))
    k = max(10, min(k, n - 1))
    ws = np.sort(w)[::-1]
    thr = ws[k]
    if thr <= 0:
        return np.inf, np.inf, k
    logs = np.log(ws[:k] / thr)
    m = logs.mean()
    a = 1.0 / m if m > 0 else np.inf
    return float(a), float(a / np.sqrt(k)), k


def overlap_diagnostic(w: np.ndarray, k: int | None = None) -> dict:
    """ESS/n, max weight (absolute and relative to sum), Hill tail index, and a warning category."""
    w = np.asarray(w, dtype=float)
    n = len(w)
    ess = w.sum() ** 2 / (w ** 2).sum()
    a, se, kk = hill_alpha(w, k)
    if a > 2.5:
        cat = "stable"
    elif a > 2.0:
        cat = "borderline"
    else:
        cat = "non-root-n"
    msg = {
        "stable": "weight tail looks light enough for root-n Wald inference",
        "borderline": "weight tail index near 2: Wald coverage may be below nominal; "
                      "report a full-pipeline bootstrap interval",
        "non-root-n": "estimated weight tail index <= 2: the DR score may have infinite or borderline "
                      "variance; ordinary root-n Wald inference may be unreliable",
    }[cat]
    return dict(ess_frac=float(ess / n), max_w=float(w.max()), max_w_share=float(w.max() / w.sum()),
                alpha_hat=a, alpha_se=se, hill_k=kk, category=cat, message=msg)
