"""Phase 5 Step 19: representative case selection (categories A/B/C)."""

from __future__ import annotations

from nfl_predict.market.ats import ATSBetRecord
from nfl_predict.market.error_review import select_review_cases


def _bet(game_id, edge, grade, side, actual_margin, home_spread=-3.0) -> ATSBetRecord:
    return ATSBetRecord(
        game_id=game_id, season=2024, week=1, season_type="REG", model_id="test",
        side=side, edge_points=edge, edge_bucket="4.00+", home_spread_traditional=home_spread,
        actual_margin=actual_margin, grade=grade, real_price=-110, real_profit_units=0.0,
        assumed_price=-110, assumed_profit_units=0.0,
    )


def test_case_a_model_disagreed_and_was_right():
    # Model bet home (big positive edge), and home covered.
    bets = [_bet("g1", edge=5.0, grade="home_covers", side="home", actual_margin=10.0)]
    cases = select_review_cases(bets)
    assert len(cases["A_model_disagreed_and_was_right"]) == 1
    assert cases["A_model_disagreed_and_was_right"][0].game_id == "g1"
    assert cases["B_model_disagreed_and_market_was_right"] == []


def test_case_b_model_disagreed_and_market_was_right():
    # Model bet away (edge favors away), but home covered - model was wrong, market right.
    bets = [_bet("g2", edge=-5.0, grade="home_covers", side="away", actual_margin=10.0)]
    cases = select_review_cases(bets)
    assert len(cases["B_model_disagreed_and_market_was_right"]) == 1
    assert cases["A_model_disagreed_and_was_right"] == []


def test_case_c_agreed_and_both_wrong():
    # Small edge (agreement), but actual margin is far from the shared implied line.
    bets = [_bet("g3", edge=0.1, grade="home_covers", side="home", actual_margin=30.0, home_spread=-3.0)]
    cases = select_review_cases(bets, big_miss_threshold=14.0)
    assert len(cases["C_agreed_and_both_wrong"]) == 1


def test_small_miss_agreement_does_not_qualify_as_case_c():
    bets = [_bet("g4", edge=0.1, grade="home_covers", side="home", actual_margin=4.0, home_spread=-3.0)]
    cases = select_review_cases(bets, big_miss_threshold=14.0)
    assert cases["C_agreed_and_both_wrong"] == []


def test_pushes_are_excluded_from_categories_a_and_b():
    bets = [_bet("g5", edge=5.0, grade="push", side="home", actual_margin=3.0)]
    cases = select_review_cases(bets)
    assert cases["A_model_disagreed_and_was_right"] == []
    assert cases["B_model_disagreed_and_market_was_right"] == []


def test_n_per_category_limits_results():
    bets = [_bet(f"g{i}", edge=5.0 + i, grade="home_covers", side="home", actual_margin=20.0) for i in range(10)]
    cases = select_review_cases(bets, n_per_category=3)
    assert len(cases["A_model_disagreed_and_was_right"]) == 3
    # Largest-edge games should be selected first.
    assert cases["A_model_disagreed_and_was_right"][0].game_id == "g9"
