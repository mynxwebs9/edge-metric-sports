"""Phase 5 Step 23: moneyline edge bet construction and tie-as-push handling."""

from __future__ import annotations

import numpy as np
import polars as pl

from nfl_predict.market.moneyline_edge import build_moneyline_bets, summarize_moneyline_bets


def _frame() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "season": [2024, 2024, 2024], "week": [1, 1, 1], "season_type": ["REG"] * 3,
        "home_moneyline": [-150, 120, -110], "away_moneyline": [130, -140, -110],
        "home_win": [1, 0, None], "is_tie": [False, False, True],
    })


def test_no_edge_produces_no_bet():
    frame = _frame()
    # Model matches the market's own no-vig probability exactly for every game.
    from nfl_predict.market.odds_math import no_vig_two_way
    matched_probs = [no_vig_two_way(h, a).no_vig_prob_a for h, a in zip(frame["home_moneyline"], frame["away_moneyline"])]
    bets = build_moneyline_bets("test_model", np.array(matched_probs), frame)
    assert bets == []


def test_a_tie_is_graded_as_a_push_with_zero_profit():
    frame = _frame()
    model_probs = np.array([0.9, 0.9, 0.9])  # forces a bet on every game, including the tie
    bets = build_moneyline_bets("test_model", model_probs, frame)
    tie_bet = next(b for b in bets if b.game_id == "g3")
    assert tie_bet.won is None
    assert tie_bet.profit_units == 0.0


def test_side_selection_picks_whichever_side_has_positive_edge():
    frame = _frame()
    # g1: home strongly favored by market; model agrees even more heavily on home.
    model_probs = np.array([0.95, 0.5, 0.5])
    bets = build_moneyline_bets("test_model", model_probs, frame)
    g1_bet = next(b for b in bets if b.game_id == "g1")
    assert g1_bet.side == "home"
    assert g1_bet.edge_probability > 0


def test_summarize_moneyline_bets_reports_calibration_gap():
    frame = _frame()
    model_probs = np.array([0.95, 0.5, 0.5])
    bets = build_moneyline_bets("test_model", model_probs, frame)
    summary = summarize_moneyline_bets(bets)
    assert "calibration_gap" in summary
    assert summary["wins"] + summary["losses"] + summary["pushes"] == summary["n_bets"]


def test_summarize_moneyline_bets_on_empty_list():
    assert summarize_moneyline_bets([]) == {"n_bets": 0}
