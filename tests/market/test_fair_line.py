"""Phase 5 Step 4 correctness checks: fair-line derivation from frozen model outputs and
the one Phase 3/4-validated uncertainty estimate - no new fitting happens here."""

from __future__ import annotations

import numpy as np
import pytest

from nfl_predict.market.fair_line import (
    DEFAULT_MARGIN_RESIDUAL_STD,
    PHASE3_FROZEN_MARGIN_RESIDUAL_STD,
    fair_home_cover_probability,
    fair_home_spread_traditional,
    fair_home_win_probability,
    fair_moneyline_price,
)
from nfl_predict.market.odds_math import american_to_implied_probability


def test_default_sigma_is_the_documented_phase3_frozen_value():
    assert DEFAULT_MARGIN_RESIDUAL_STD == PHASE3_FROZEN_MARGIN_RESIDUAL_STD


def test_fair_win_probability_is_50_50_at_zero_predicted_margin():
    assert fair_home_win_probability(np.array([0.0]))[0] == pytest.approx(0.5)


def test_fair_win_probability_increases_with_predicted_margin():
    probs = fair_home_win_probability(np.array([-10.0, -3.0, 0.0, 3.0, 10.0]))
    assert list(probs) == sorted(probs)


def test_fair_win_probability_approaches_bounds_for_large_margins():
    assert fair_home_win_probability(np.array([50.0]))[0] > 0.99
    assert fair_home_win_probability(np.array([-50.0]))[0] < 0.01


def test_fair_cover_probability_is_50_50_when_margin_exactly_meets_the_spread():
    # Home favored by 3.5 (traditional -3.5), model predicts exactly a 3.5-point win.
    prob = fair_home_cover_probability(np.array([3.5]), np.array([-3.5]))
    assert prob[0] == pytest.approx(0.5)


def test_fair_cover_probability_favors_home_when_model_predicts_more_than_the_spread():
    prob = fair_home_cover_probability(np.array([10.0]), np.array([-3.5]))
    assert prob[0] > 0.5


def test_fair_cover_probability_favors_away_when_model_predicts_less_than_the_spread():
    prob = fair_home_cover_probability(np.array([1.0]), np.array([-3.5]))
    assert prob[0] < 0.5


def test_fair_home_spread_traditional_is_the_negated_predicted_margin():
    assert fair_home_spread_traditional(np.array([7.0]))[0] == -7.0
    assert fair_home_spread_traditional(np.array([-4.0]))[0] == 4.0


def test_fair_cover_probability_at_the_models_own_fair_spread_is_always_50_50():
    margins = np.array([-14.0, -3.0, 0.0, 6.5, 21.0])
    own_fair_spread = fair_home_spread_traditional(margins)
    probs = fair_home_cover_probability(margins, own_fair_spread)
    assert np.allclose(probs, 0.5)


def test_fair_moneyline_price_round_trips_through_implied_probability():
    prices = fair_moneyline_price(np.array([0.5, 0.7, 0.3]))
    recovered = [american_to_implied_probability(p) for p in prices]
    assert recovered == pytest.approx([0.5, 0.7, 0.3], abs=0.01)
