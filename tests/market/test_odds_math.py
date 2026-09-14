"""Phase 5 Step 23 proofs #1-2, #4-7: American-odds conversion, no-vig probability, ATS
sign convention, spread-disagreement-adjacent sign convention, moneyline P/L, and the
-110 break-even reference point."""

from __future__ import annotations

import pytest

from nfl_predict.market.odds_math import (
    american_to_implied_probability,
    ats_profit_units,
    ats_won,
    break_even_probability,
    grade_ats,
    implied_probability_to_american,
    moneyline_clv_probability,
    moneyline_profit_units,
    nflverse_spread_to_traditional_home_spread,
    no_vig_two_way,
    spread_clv_points,
)


# ---------------------------------------------------------------------------
# Proof #1: American odds -> implied probability
# ---------------------------------------------------------------------------

def test_negative_odds_implied_probability():
    assert american_to_implied_probability(-110) == pytest.approx(110 / 210)
    assert american_to_implied_probability(-200) == pytest.approx(200 / 300)


def test_positive_odds_implied_probability():
    assert american_to_implied_probability(150) == pytest.approx(100 / 250)
    assert american_to_implied_probability(100) == pytest.approx(0.5)


def test_implied_probability_round_trip():
    for odds in (-110, -200, -105, 120, 150, 300):
        prob = american_to_implied_probability(odds)
        recovered = implied_probability_to_american(prob)
        assert recovered == odds


def test_zero_odds_is_rejected():
    with pytest.raises(ValueError):
        american_to_implied_probability(0)


# ---------------------------------------------------------------------------
# Proof #2: no-vig probability calculation
# ---------------------------------------------------------------------------

def test_no_vig_two_way_normalizes_to_one():
    result = no_vig_two_way(-120, 105)
    assert result.no_vig_prob_a + result.no_vig_prob_b == pytest.approx(1.0)


def test_no_vig_two_way_symmetric_market_has_no_hold_correction():
    # -110/-110 is a standard symmetric market: raw implied probs already sum to > 1
    # (that's the hold), and de-vigging a perfectly symmetric market keeps each side at 0.5.
    result = no_vig_two_way(-110, -110)
    assert result.no_vig_prob_a == pytest.approx(0.5)
    assert result.no_vig_prob_b == pytest.approx(0.5)
    assert result.bookmaker_hold > 0


def test_no_vig_hold_matches_manual_calculation():
    # Team A -120, Team B +105 - the exact worked example from the Phase 5 brief.
    result = no_vig_two_way(-120, 105)
    raw_a = 120 / 220
    raw_b = 100 / 205
    assert result.raw_implied_prob_a == pytest.approx(raw_a)
    assert result.raw_implied_prob_b == pytest.approx(raw_b)
    assert result.bookmaker_hold == pytest.approx(raw_a + raw_b - 1.0)
    assert result.no_vig_prob_a == pytest.approx(raw_a / (raw_a + raw_b))


# ---------------------------------------------------------------------------
# Proof #7: -110 break-even is approximately 52.38%
# ---------------------------------------------------------------------------

def test_minus_110_break_even_is_approximately_52_38_percent():
    assert break_even_probability(-110) == pytest.approx(0.5238, abs=1e-4)


# ---------------------------------------------------------------------------
# Proof #4: ATS result sign convention
# ---------------------------------------------------------------------------

def test_grade_ats_home_favorite_covers():
    # Home favored by 3.5 (traditional -3.5); home wins by 7 -> covers.
    assert grade_ats(actual_margin=7.0, home_spread_traditional=-3.5) == "home_covers"


def test_grade_ats_home_favorite_fails_to_cover():
    # Home favored by 3.5; home wins by only 1 -> does not cover, away covers.
    assert grade_ats(actual_margin=1.0, home_spread_traditional=-3.5) == "away_covers"


def test_grade_ats_home_favorite_loses_outright_away_covers():
    assert grade_ats(actual_margin=-10.0, home_spread_traditional=-3.5) == "away_covers"


def test_grade_ats_underdog_home_covers_by_losing_close():
    # Home is a 3.5-point underdog (traditional +3.5); home loses by only 2 -> covers.
    assert grade_ats(actual_margin=-2.0, home_spread_traditional=3.5) == "home_covers"


def test_grade_ats_push():
    assert grade_ats(actual_margin=3.5, home_spread_traditional=-3.5) == "push"
    assert grade_ats(actual_margin=0.0, home_spread_traditional=0.0) == "push"


def test_nflverse_sign_conversion_matches_documented_dictionary():
    # nflverse spread_line=+3.0 means home favored by 3 -> traditional home spread is -3.0.
    assert nflverse_spread_to_traditional_home_spread(3.0) == -3.0
    assert nflverse_spread_to_traditional_home_spread(-3.0) == 3.0


def test_ats_won_is_none_on_push():
    assert ats_won("home", "push") is None
    assert ats_won("away", "push") is None


def test_ats_won_matches_the_side_that_actually_covered():
    assert ats_won("home", "home_covers") is True
    assert ats_won("away", "home_covers") is False
    assert ats_won("home", "away_covers") is False
    assert ats_won("away", "away_covers") is True


# ---------------------------------------------------------------------------
# Proof #3: pushes are handled correctly (no profit/loss on a push)
# ---------------------------------------------------------------------------

def test_ats_profit_units_on_push_is_zero():
    assert ats_profit_units("home", "push", -110) == 0.0
    assert ats_profit_units("away", "push", -110) == 0.0


# ---------------------------------------------------------------------------
# Proof #6: moneyline P/L calculations
# ---------------------------------------------------------------------------

def test_moneyline_profit_units_favorite_win():
    assert moneyline_profit_units(-150, won=True) == pytest.approx(100 / 150)


def test_moneyline_profit_units_underdog_win():
    assert moneyline_profit_units(150, won=True) == pytest.approx(1.5)


def test_moneyline_profit_units_loss_is_always_minus_one():
    assert moneyline_profit_units(-150, won=False) == -1.0
    assert moneyline_profit_units(300, won=False) == -1.0


def test_ats_profit_units_wins_and_losses_use_the_ats_price_not_moneyline_price():
    # Home covers at a -108 ATS price - the win/loss math is identical to a moneyline bet
    # at that same price, so this exercises the ATS wrapper directly.
    assert ats_profit_units("home", "home_covers", -108) == pytest.approx(100 / 108)
    assert ats_profit_units("home", "away_covers", -108) == -1.0


# ---------------------------------------------------------------------------
# CLV formulas (Step 11), tested for correctness even though real historical data has only
# one snapshot per game (see test_clv.py for the "not fabricated" proof).
# ---------------------------------------------------------------------------

def test_spread_clv_points_favorable_example_from_brief():
    # Recommended side (favorite) at -2.5, closes at -3.5 -> favorable CLV of +1.0 point,
    # the exact worked example from the Phase 5 brief.
    clv = spread_clv_points(bet_time_home_spread_traditional=-2.5, closing_home_spread_traditional=-3.5, side="home")
    assert clv == pytest.approx(1.0)


def test_spread_clv_points_unfavorable_when_line_moves_the_other_way():
    clv = spread_clv_points(bet_time_home_spread_traditional=-3.5, closing_home_spread_traditional=-2.5, side="home")
    assert clv == pytest.approx(-1.0)


def test_spread_clv_points_for_the_away_side_uses_the_negated_line():
    # Same underlying market as the favorable example, but graded from the away
    # (underdog) side's perspective - the dog getting a bigger home-favorite by closing
    # actually means the dog's own number also got worse for it (it now has to lay fewer
    # points... wait, no: a bigger home favorite means the away underdog's own spread also
    # widens in the away team's favor). Verify the sign mechanically rather than by
    # intuition alone.
    clv = spread_clv_points(bet_time_home_spread_traditional=-2.5, closing_home_spread_traditional=-3.5, side="away")
    assert clv == pytest.approx(-1.0)


def test_moneyline_clv_probability_favorable_when_side_strengthens_after_bet():
    clv = moneyline_clv_probability(bet_time_no_vig_prob_for_side=0.40, closing_no_vig_prob_for_side=0.4545)
    assert clv == pytest.approx(0.0545, abs=1e-4)
