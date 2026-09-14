"""Phase 5 Step 23 proof #14: market-only vs market+model comparisons stay temporally
valid (fit on one season, evaluated on the other - never fit-and-eval on the same rows)."""

from __future__ import annotations

import numpy as np

from nfl_predict.market.incremental_information import run_incremental_information_test


def test_fit_and_eval_seasons_are_always_disjoint():
    rng = np.random.default_rng(0)
    n = 60
    seasons = np.array([2024] * (n // 2) + [2025] * (n // 2))
    market = rng.normal(0, 7, n)
    model_pred = market + rng.normal(0, 1, n)
    actual = market + rng.normal(0, 10, n)

    results = run_incremental_information_test("test_model", market, model_pred, actual, seasons)
    assert len(results) == 2
    for r in results:
        assert r.fit_season != r.eval_season
        assert {r.fit_season, r.eval_season} == {2024, 2025}
        assert r.n_fit == n // 2
        assert r.n_eval == n // 2


def test_a_model_perfectly_correlated_with_the_outcome_improves_on_market_only():
    rng = np.random.default_rng(1)
    n = 200
    seasons = np.array([2024] * (n // 2) + [2025] * (n // 2))
    actual = rng.normal(0, 14, n)
    market = actual + rng.normal(0, 12, n)  # noisy proxy for actual
    model_pred = actual + rng.normal(0, 0.01, n)  # near-perfect predictor of actual

    results = run_incremental_information_test("near_perfect_model", market, model_pred, actual, seasons)
    for r in results:
        assert r.improvement_mae > 0  # combining with a near-perfect signal must help
        assert r.market_plus_model_mae < r.market_only_mae


def test_an_uninformative_random_model_does_not_reliably_improve_on_market_only():
    rng = np.random.default_rng(2)
    n = 200
    seasons = np.array([2024] * (n // 2) + [2025] * (n // 2))
    actual = rng.normal(0, 14, n)
    market = actual + rng.normal(0, 12, n)
    model_pred = rng.normal(0, 14, n)  # pure noise, unrelated to the outcome

    results = run_incremental_information_test("noise_model", market, model_pred, actual, seasons)
    # A noise feature can occasionally overfit a small training season by chance, but the
    # improvement should be small in magnitude either way - nowhere near the near-perfect
    # model's improvement in the previous test.
    for r in results:
        assert abs(r.improvement_mae) < 1.0
