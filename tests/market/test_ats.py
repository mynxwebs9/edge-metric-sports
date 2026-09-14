"""Phase 5 Step 23 proofs #3 (pushes handled correctly at the aggregate level), #11/#12
(no silently-assumed prices; assumed-price analysis explicitly labeled)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nfl_predict.market.ats import ASSUMED_STANDARD_PRICE, build_ats_bets, summarize_ats_bets


def _synthetic_frame() -> pl.DataFrame:
    # market_implied_home_margin = -home_spread_traditional, per game:
    #   g1: -(-3.0) = 3.0   g2: -(-3.0) = 3.0   g3: -(3.0) = -3.0   g4: -(-7.0) = 7.0
    return pl.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "season": [2024, 2024, 2024, 2024], "week": [1, 1, 1, 1], "season_type": ["REG"] * 4,
        "home_spread_traditional": [-3.0, -3.0, 3.0, -7.0],
        "home_spread_price": [-110, -110, -108, -112],
        "away_spread_price": [-110, -110, -112, -108],
        "home_margin": [10.0, 3.0, -2.0, 3.0],  # g1 home covers big, g2 pushes exactly, g3 home dog covers, g4 home favorite fails to cover
    })


NO_DISAGREEMENT_MARGIN = np.array([3.0, 3.0, -3.0, 7.0])  # exactly matches market_implied_home_margin everywhere


def test_a_bet_that_pushes_is_excluded_from_win_loss_but_counted_and_given_zero_profit():
    frame = _synthetic_frame()
    # g2: home_spread=-3.0, home_margin=3.0 -> exact push. Force a real disagreement here
    # (model predicts a much bigger home margin than the market) so a bet IS placed on g2,
    # and its outcome still lands exactly on the market's line -> push.
    predicted_margin = np.array([10.0, 9.0, -2.0, -7.0])
    bets = build_ats_bets("test_model", predicted_margin, frame)
    g2_bet = next(b for b in bets if b.game_id == "g2")
    assert g2_bet.grade == "push"
    assert g2_bet.real_profit_units == 0.0
    assert g2_bet.assumed_profit_units == 0.0

    summary = summarize_ats_bets(bets)
    assert summary["pushes"] >= 1
    assert summary["wins"] + summary["losses"] + summary["pushes"] == summary["n_bets"]


def test_no_disagreement_games_produce_no_bet():
    frame = _synthetic_frame()
    bets = build_ats_bets("test_model", NO_DISAGREEMENT_MARGIN, frame)
    assert bets == []


def test_a_real_disagreement_produces_exactly_one_graded_bet():
    frame = _synthetic_frame()
    predicted_margin = np.array([3.0, 3.0, -3.0, 0.0])  # disagrees only on g4 (market implies home margin +7, model says 0)
    bets = build_ats_bets("test_model", predicted_margin, frame)
    assert len(bets) == 1
    bet = bets[0]
    assert bet.game_id == "g4"
    assert bet.side == "away"  # model favors away more than market (0 < 7)
    assert bet.real_price == -108  # away_spread_price for g4
    assert bet.assumed_price == ASSUMED_STANDARD_PRICE


def test_summarize_ats_bets_keeps_real_and_assumed_price_summaries_fully_separate():
    frame = _synthetic_frame()
    # Force disagreement on every game so all 4 produce a graded bet.
    predicted_margin = np.array([20.0, 13.0, -12.0, 3.0])
    bets = build_ats_bets("test_model", predicted_margin, frame)
    assert len(bets) == 4
    summary = summarize_ats_bets(bets)

    assert summary["real_price_summary"]["price_source"] != summary["assumed_price_summary"]["price_source"]
    assert summary["assumed_price_summary"]["average_price"] == ASSUMED_STANDARD_PRICE
    # At least one bet's real recorded price differs from the assumed constant -110 - proof
    # the real-price path isn't secretly falling back to the assumption.
    assert any(b.real_price != ASSUMED_STANDARD_PRICE for b in bets)


def test_summarize_ats_bets_on_zero_bets_does_not_crash():
    summary = summarize_ats_bets([])
    assert summary == {"n_bets": 0}


def test_wilson_ci_widens_for_smaller_samples():
    from nfl_predict.market.ats import wilson_confidence_interval

    small_lo, small_hi = wilson_confidence_interval(wins=5, losses=5)
    large_lo, large_hi = wilson_confidence_interval(wins=500, losses=500)
    assert (small_hi - small_lo) > (large_hi - large_lo)
