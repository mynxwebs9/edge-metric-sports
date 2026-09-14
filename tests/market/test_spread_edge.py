"""Phase 5 Step 10/18: bucketed ATS results, split by season so a losing season can't hide
inside a profitable aggregate."""

from __future__ import annotations

from nfl_predict.market.ats import ATSBetRecord
from nfl_predict.market.spread_edge import spread_edge_bucket_report


def _bet(game_id, season, edge_bucket, grade, side="home") -> ATSBetRecord:
    return ATSBetRecord(
        game_id=game_id, season=season, week=1, season_type="REG", model_id="test",
        side=side, edge_points=1.5, edge_bucket=edge_bucket, home_spread_traditional=-3.0,
        actual_margin=0.0, grade=grade, real_price=-110, real_profit_units=(0.909 if grade == "home_covers" else -1.0),
        assumed_price=-110, assumed_profit_units=(0.909 if grade == "home_covers" else -1.0),
    )


def test_bucket_report_separates_every_predefined_bucket():
    bets = [_bet("g1", 2024, "0.00-0.99", "home_covers"), _bet("g2", 2025, "4.00+", "away_covers")]
    report = spread_edge_bucket_report(bets)
    assert set(report) >= {"0.00-0.99", "1.00-1.99", "2.00-2.99", "3.00-3.99", "4.00+", "all_bets_any_disagreement"}
    assert report["0.00-0.99"]["combined_2024_2025"]["n_bets"] == 1
    assert report["1.00-1.99"]["combined_2024_2025"] == {"n_bets": 0}


def test_bucket_report_splits_by_season_without_double_counting():
    bets = [_bet("g1", 2024, "1.00-1.99", "home_covers"), _bet("g2", 2025, "1.00-1.99", "away_covers")]
    report = spread_edge_bucket_report(bets)
    bucket = report["1.00-1.99"]
    assert bucket["combined_2024_2025"]["n_bets"] == 2
    assert bucket["season_2024"]["n_bets"] == 1
    assert bucket["season_2025"]["n_bets"] == 1
