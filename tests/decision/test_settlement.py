"""Phase 7 Step 24 proofs #8-9: settlement handles spread win/loss/push and moneyline
correctly."""

from __future__ import annotations

import pytest

from nfl_predict.decision.pick_ledger import SettlementResult
from nfl_predict.decision.settlement import (
    UnsupportedMarketTypeError,
    settle_moneyline_pick,
    settle_pick,
    settle_spread_pick,
)


def test_spread_home_favorite_covers():
    assert settle_spread_pick("home", -3.5, actual_home_margin=7.0) == SettlementResult.WIN


def test_spread_home_favorite_fails_to_cover():
    assert settle_spread_pick("home", -3.5, actual_home_margin=1.0) == SettlementResult.LOSS


def test_spread_away_side_wins_when_home_favorite_fails_to_cover():
    assert settle_spread_pick("away", -3.5, actual_home_margin=1.0) == SettlementResult.WIN


def test_spread_push():
    assert settle_spread_pick("home", -3.5, actual_home_margin=3.5) == SettlementResult.PUSH
    assert settle_spread_pick("away", -3.5, actual_home_margin=3.5) == SettlementResult.PUSH


def test_moneyline_home_win():
    assert settle_moneyline_pick("home", actual_home_win=True) == SettlementResult.WIN
    assert settle_moneyline_pick("away", actual_home_win=True) == SettlementResult.LOSS


def test_moneyline_away_win():
    assert settle_moneyline_pick("away", actual_home_win=False) == SettlementResult.WIN
    assert settle_moneyline_pick("home", actual_home_win=False) == SettlementResult.LOSS


def test_moneyline_tie_is_a_push():
    assert settle_moneyline_pick("home", actual_home_win=None) == SettlementResult.PUSH
    assert settle_moneyline_pick("away", actual_home_win=None) == SettlementResult.PUSH


def test_settle_pick_dispatches_by_market_type():
    assert settle_pick("spread", "home", line=-3.5, actual_home_margin=7.0) == SettlementResult.WIN
    assert settle_pick("moneyline", "home", actual_home_win=True) == SettlementResult.WIN


def test_totals_market_type_is_rejected():
    with pytest.raises(UnsupportedMarketTypeError):
        settle_pick("total", "over", line=45.0, actual_home_margin=10.0)


def test_spread_settlement_requires_line_and_actual_margin():
    with pytest.raises(ValueError):
        settle_pick("spread", "home")
