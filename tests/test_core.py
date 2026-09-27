import numpy as np
import pytest

from src.dgp import (K, P, behavior_probs, draw_noise, generate_replay, mu_all, true_value)
from src.estimators import dm, ips, robust_dr, snips
from src.registry import REAL_DATA_ESTIMATORS, SIM_ESTIMATORS
from src.nuisance import crossfit_nuisances
from src.policies import OVERLAP_REGIMES, PolicySpec, tilted_policy
from src.tuning import CLIP_RULES, tau_mad


def test_policies_are_distributions():
    X = np.random.default_rng(0).standard_normal((500, P))
    for spec in OVERLAP_REGIMES.values():
        pi = spec(X)
        assert np.allclose(pi.sum(1), 1) and (pi > 0).all()
    assert np.allclose(tilted_policy(X, 0.0, eta=0.0), behavior_probs(X))


def test_truth_mc_accuracy_and_seed_stability():
    s = OVERLAP_REGIMES["moderate"]
    v1, se1 = true_value(s, n_mc=400_000, seed=1)
    v2, se2 = true_value(s, n_mc=400_000, seed=2)
    assert se1 < 0.003
    assert abs(v1 - v2) < 4 * np.hypot(se1, se2)


def test_truth_for_behavior_equals_mean_reward():
    # value of pi_b equals E[Y] under pi_b-logged data
    b = PolicySpec("b", lam=0.0)
    v, _ = true_value(b, n_mc=400_000)
    rng = np.random.default_rng(3)
    d = generate_replay(rng, 400_000, noise="gauss")
    assert abs(d.Y.mean() - v) < 0.01


def test_noise_laws():
    rng = np.random.default_rng(1)
    e, f = draw_noise(rng, 200_000, "contam", rate=0.1, sigma_out=10)
    assert abs(f.mean() - 0.1) < 0.005
    assert abs(np.median(e)) < 0.02
    e, f = draw_noise(rng, 200_000, "gross", rate=0.02)
    assert abs(e.mean()) < 0.05


def test_dr_exact_when_nuisance_exact():
    rng = np.random.default_rng(5)
    d = generate_replay(rng, 2000)
    pi = OVERLAP_REGIMES["moderate"](d.X)
    Y0 = d.mu[np.arange(len(d.A)), d.A]  # noiseless
    v_dm, _ = dm(pi, d.A, Y0, d.mu, d.pb)
    v_dr, _ = robust_dr(pi, d.A, Y0, d.mu, d.pb)
    assert abs(v_dm - v_dr) < 1e-10  # residuals are exactly zero


def test_robust_dr_reduces_to_dr():
    rng = np.random.default_rng(6)
    d = generate_replay(rng, 1000, noise="t3")
    pi = OVERLAP_REGIMES["poor"](d.X)
    mhat = d.mu + 0.3
    a = robust_dr(pi, d.A, d.Y, mhat, d.pb, clip="none", tau="none")
    idx = np.arange(len(d.A))
    w = pi[idx, d.A] / d.pb[idx, d.A]
    manual = np.mean((pi * mhat).sum(1) + w * (d.Y - mhat[idx, d.A]))
    assert abs(a[0] - manual) < 1e-12


def test_is_unbiased_with_oracle_propensity():
    s = OVERLAP_REGIMES["good"]
    truth, _ = true_value(s, n_mc=500_000)
    ests = []
    for r in range(300):
        d = generate_replay(np.random.default_rng(100 + r), 500)
        pi = s(d.X)
        ests.append(ips(pi, d.A, d.Y, d.mu, d.pb)[0])
    ests = np.array(ests)
    assert abs(ests.mean() - truth) < 4 * ests.std() / np.sqrt(len(ests))


def test_crossfit_out_of_fold_and_correct_spec():
    rng = np.random.default_rng(7)
    d = generate_replay(rng, 1000)
    Y0 = d.mu[np.arange(1000), d.A]
    nu = crossfit_nuisances(d.X, d.A, Y0, outcome_specs=("correct", "mis"), prop_specs=("correct", "mis"),
                            n_folds=5, seed=0)
    # correct outcome spec recovers noiseless mu out of fold
    assert np.max(np.abs(nu[("m", "correct", True)] - d.mu)) < 1e-4
    # misspecified spec does not
    assert np.mean(np.abs(nu[("m", "mis", True)] - d.mu)) > 0.1
    # propensities are valid distributions
    for key in [("pb", "correct", True), ("pb", "mis", True)]:
        assert np.allclose(nu[key].sum(1), 1)
    # correct propensity close to truth
    assert np.mean(np.abs(nu[("pb", "correct", True)] - d.pb)) < 0.05


def test_adaptive_logging_weights_have_unit_conditional_mean():
    d = generate_replay(np.random.default_rng(22), 4000, logging="batched")
    pi = OVERLAP_REGIMES["moderate"](d.X)
    # E[w | X, H] = sum_a pi(a|X) = 1 exactly for the known logging probabilities
    assert np.allclose((d.pb * (pi / d.pb)).sum(1), 1.0)
    assert set(np.unique(d.meta["batch_id"])) == {0, 1, 2, 3, 4}


def test_overlap_diagnostic_hill():
    from src.diagnostics import hill_alpha, overlap_diagnostic
    rng = np.random.default_rng(31)
    for a in (1.5, 2.0, 3.0):
        w = rng.pareto(a, 200_000) + 1.0          # exact Pareto(alpha) tail
        ah, se, k = hill_alpha(w)
        assert abs(ah - a) < 4 * se + 0.05
    d = overlap_diagnostic(rng.pareto(1.5, 5000) + 1.0)
    assert d["category"] == "non-root-n"
    d = overlap_diagnostic(np.exp(rng.normal(0, 0.3, 5000)))   # light tail
    assert d["category"] == "stable"


def test_forward_fit_is_predictable():
    from src.nuisance import forward_outcome
    d = generate_replay(np.random.default_rng(32), 1000, logging="batched")
    b = d.meta["batch_id"]
    p1 = forward_outcome(d.X, d.A, d.Y, "correct_huber", b, first="zero")
    # changing the outcomes of the LAST batch must not change any prediction (all use the past only)
    Y2 = d.Y.copy(); Y2[b == 4] += 100.0
    p2 = forward_outcome(d.X, d.A, Y2, "correct_huber", b, first="zero")
    assert np.allclose(p1, p2)
    assert np.allclose(p1[b == 0], 0.0)
    # changing batch-2 outcomes changes batches 3, 4 but not 0..2
    Y3 = d.Y.copy(); Y3[b == 2] += 5.0
    p3 = forward_outcome(d.X, d.A, Y3, "correct_huber", b, first="zero")
    assert np.allclose(p1[b <= 2], p3[b <= 2]) and not np.allclose(p1[b == 3], p3[b == 3])


def test_tuning_rules():
    rng = np.random.default_rng(8)
    r = rng.normal(0, 2, 100_000)
    assert abs(tau_mad(r, kappa=1.0) - 2) < 0.05
    w = np.ones(1000)
    assert abs(CLIP_RULES["ionides"](w) - np.sqrt(1000)) < 1e-9
    assert CLIP_RULES["none"](w) == np.inf


def test_all_registered_estimators_run():
    d = generate_replay(np.random.default_rng(9), 800, noise="contam", rate=0.1, logging="batched")
    pi = OVERLAP_REGIMES["poor"](d.X)
    for reg in (SIM_ESTIMATORS, REAL_DATA_ESTIMATORS):
        for name, (f, kw) in reg.items():
            v, se = f(pi, d.A, d.Y, d.mu + 0.2, d.pb, batch=d.meta["batch_id"], **kw)
            assert np.isfinite(v) and np.isfinite(se), name


def test_vr_without_threshold_equals_capped_dr():
    from src.vr import vr_dr
    d = generate_replay(np.random.default_rng(21), 1000, noise="t3")
    pi = OVERLAP_REGIMES["poor"](d.X)
    mh = d.mu + 0.2
    n = len(d.A)
    v0, _ = vr_dr(pi, d.A, d.Y, mh, d.pb, kappa=np.inf)
    vd, _ = robust_dr(pi, d.A, d.Y, mh, d.pb, clip=lambda w, psi, mpi: w.mean() * np.sqrt(n / np.log(n)))
    assert abs(v0 - vd) < 1e-12


def test_rm_huber_near_mean_for_large_n():
    from src.robust_mean import huber_location, tau_n
    rng = np.random.default_rng(13)
    g = np.exp(rng.normal(0, 1, 200_000))  # skewed, finite moments
    assert abs(huber_location(g)[0] - np.exp(0.5)) < 0.02
    assert tau_n(rng.normal(size=4000)) > tau_n(rng.normal(size=1000))
