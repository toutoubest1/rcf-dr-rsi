"""Target exploration policies with controllable distance from pi_b.

A target policy tilts the behaviour logits by a score direction s_a(x):

    pi_{lam, eta, s}(a | x) = softmax_a( (1 - eta) g_a(x) + lam * s_a(x) ).

lam = eta = 0 recovers pi_b.  Larger lam moves the policy further from pi_b
along s; eta > 0 additionally "un-learns" the behaviour preference and,
for eta > 1, actively favours actions pi_b rarely takes (poor overlap,
extreme importance weights).  The score direction mixes
  * u_a(x): an "improvement" signal (v1: linear part of mu_a; v2: linear
    part + action-specific nonlinear term h_a),
    mimicking an RSI system that proposes greedier exploration strategies;
  * v_a(x): a random linear direction (fixed per policy id),
    mimicking arbitrary proposed changes of exploration behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import numpy as np

from .dgp import BETA, INTERCEPT, K, P, action_specific, behavior_logits, softmax

# DGP_VERSION is part of the truth-cache key.
#   v1: u_a(x) = b_a + beta_a'x  (linear improvement direction)
#   v2: u_a(x) = b_a + beta_a'x + h_a(x)  (target policies also exploit the
#       nonlinear action-specific reward terms)
DGP_VERSION = "v2"


def improvement_score(X: np.ndarray) -> np.ndarray:
    u = INTERCEPT[None, :] + X @ BETA.T
    if DGP_VERSION == "v2":
        u = u + action_specific(X)
    return u


def random_score(X: np.ndarray, pid: int) -> np.ndarray:
    r = np.random.default_rng(10_000 + pid)
    G = r.normal(0.0, 0.3, size=(K, P))
    c = r.normal(0.0, 0.3, size=K)
    return c[None, :] + X @ G.T


def tilted_policy(X: np.ndarray, lam: float, alpha: float = 1.0, pid: int = 0,
                  eta: float = 0.0) -> np.ndarray:
    s = alpha * improvement_score(X)
    if alpha < 1.0:
        s = s + (1.0 - alpha) * random_score(X, pid)
    return softmax((1.0 - eta) * behavior_logits(X) + lam * s)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    lam: float
    alpha: float = 1.0
    pid: int = 0
    eta: float = 0.0

    def fn(self):
        return partial(tilted_policy, lam=self.lam, alpha=self.alpha, pid=self.pid, eta=self.eta)

    def __call__(self, X):
        return tilted_policy(X, self.lam, self.alpha, self.pid, self.eta)


# Overlap regimes. lam and eta were calibrated during development
# to give normalised ESS 1/E[w^2] of roughly
# ~0.66 (good), ~0.22 (moderate), ~0.03 (poor) under the indep design.
OVERLAP_REGIMES = {
    "good": PolicySpec("good", lam=1.0, eta=0.0),
    "moderate": PolicySpec("moderate", lam=2.0, eta=0.5),
    "poor": PolicySpec("poor", lam=3.0, eta=2.0),
}


# ============================================================================
# Candidate policies with POLICY-DEPENDENT reward-model misspecification
# ============================================================================
# Each candidate tilts the behaviour logits along the linear improvement
# direction plus a candidate-specific subset of the omitted nonlinear reward
# components.  A linear reward model misses exactly those components, so its
# value bias differs across candidates. 

def linear_score(X: np.ndarray) -> np.ndarray:
    return INTERCEPT[None, :] + X @ BETA.T


def component_score(X: np.ndarray, comps: tuple) -> np.ndarray:
    """Action-specific nonlinear reward components selected by name.

    'h0'..'h3' : the true action-specific terms h_a (cos x4, x5^2, tanh(x6 x7), |x8|),
                 added to their own action only;
    'x23'      : a DECOY -- tilts action 1 by the shared term x2*x3, which does not
                 change the reward difference between actions (no true gain).
    """
    out = np.zeros((X.shape[0], K))
    h = action_specific(X)
    for c in comps:
        if c.startswith("h"):
            a = int(c[1])
            out[:, a] += h[:, a]
        elif c == "x23":
            out[:, 1] += 0.5 * X[:, 1] * X[:, 2]
    return out


@dataclass(frozen=True)
class CompPolicySpec:
    name: str
    lam: float
    eta: float = 0.0
    gamma: float = 0.0
    comps: tuple = ()

    def __call__(self, X):
        s = linear_score(X) + self.gamma * component_score(X, self.comps)
        return softmax((1.0 - self.eta) * behavior_logits(X) + self.lam * s)

    # for the truth cache key (simulation._key uses these attributes)
    @property
    def alpha(self):
        return f"g{self.gamma}"

    @property
    def pid(self):
        return "+".join(self.comps) or "lin"


def candidate_policies_v3(M: int = 10) -> list:
    """Hand-designed candidate sets (M = 10 or 20) (Table S2 of the supplement)."""
    base = [
        CompPolicySpec("lin_a", lam=2.0, eta=0.5),
        CompPolicySpec("lin_b", lam=3.5, eta=1.0),
        CompPolicySpec("h1", lam=2.0, eta=0.5, gamma=3.0, comps=("h1",)),
        CompPolicySpec("h2", lam=2.0, eta=0.5, gamma=3.0, comps=("h2",)),
        CompPolicySpec("h3", lam=2.0, eta=0.5, gamma=3.0, comps=("h3",)),
        CompPolicySpec("h0", lam=2.0, eta=0.5, gamma=3.0, comps=("h0",)),
        CompPolicySpec("hall", lam=2.0, eta=0.5, gamma=1.0, comps=("h0", "h1", "h2", "h3")),
        CompPolicySpec("h12", lam=3.0, eta=1.0, gamma=2.0, comps=("h1", "h2")),
        CompPolicySpec("decoy_x23", lam=2.0, eta=0.5, gamma=3.0, comps=("x23",)),
        CompPolicySpec("lin_c", lam=1.5, eta=0.0),
    ]
    if M <= 10:
        return base[:M]
    extra = [
        CompPolicySpec("h1_strong", lam=3.0, eta=1.0, gamma=4.0, comps=("h1",)),
        CompPolicySpec("h2_strong", lam=3.0, eta=1.0, gamma=4.0, comps=("h2",)),
        CompPolicySpec("h3_weak", lam=2.0, eta=0.3, gamma=1.5, comps=("h3",)),
        CompPolicySpec("h0_weak", lam=2.0, eta=0.3, gamma=1.5, comps=("h0",)),
        CompPolicySpec("h03", lam=2.5, eta=0.8, gamma=2.0, comps=("h0", "h3")),
        CompPolicySpec("hall_strong", lam=3.0, eta=1.2, gamma=1.5, comps=("h0", "h1", "h2", "h3")),
        CompPolicySpec("lin_d", lam=5.0, eta=1.5),
        CompPolicySpec("lin_e", lam=2.5, eta=0.2),
        CompPolicySpec("decoy_strong", lam=3.0, eta=1.0, gamma=5.0, comps=("x23",)),
        CompPolicySpec("h23", lam=2.5, eta=0.8, gamma=2.5, comps=("h2", "h3")),
    ]
    return (base + extra)[:M]
