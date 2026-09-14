"""Phase 5 Step 5 checks: the market benchmark uses the same metrics functions as Phase 4,
on the same 570-game holdout, with no market data leaking into the outcome join."""

from __future__ import annotations

import pytest

from nfl_predict.backtesting.ledger import ledger_paths


def _skip_if_no_ledger():
    ledger_path, _ = ledger_paths("phase4_v1")
    if not ledger_path.is_file():
        pytest.skip("phase4_v1 ledger not present in this environment")


def test_market_benchmark_covers_all_570_holdout_games():
    _skip_if_no_ledger()
    from nfl_predict.market.benchmark import run_market_benchmark

    result = run_market_benchmark("phase4_v1")
    assert result["n_games"] == 570


def test_market_benchmark_bookmaker_hold_is_small_and_positive():
    _skip_if_no_ledger()
    from nfl_predict.market.benchmark import run_market_benchmark

    result = run_market_benchmark("phase4_v1")
    # A standard two-sided market has a few percent of hold (e.g. -110/-110 -> ~4.76%) -
    # anything wildly outside a plausible range would indicate a units/sign bug upstream.
    assert 0.0 < result["average_bookmaker_hold"] < 0.15


def test_market_margin_metrics_have_the_same_shape_as_a_phase4_model_entry():
    _skip_if_no_ledger()
    from nfl_predict.market.benchmark import run_market_benchmark

    result = run_market_benchmark("phase4_v1")
    margin = result["market_spread"]["home_margin"]["overall_2024_2025"]
    assert set(margin) == {"n", "mae", "rmse", "bias", "corr"}
    assert margin["n"] == 570
    assert margin["mae"] > 0


def test_market_win_metrics_exclude_ties_the_same_way_phase4_does():
    _skip_if_no_ledger()
    from nfl_predict.market.benchmark import run_market_benchmark

    result = run_market_benchmark("phase4_v1")
    win = result["market_moneyline"]["home_win"]["overall_2024_2025"]
    assert win["n"] == 569
    assert win["n_ties_excluded"] == 1
