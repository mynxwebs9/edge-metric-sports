"""Phase 7 Step 24 proofs #10-11, #28, #30: units use actual prices, missing price never
silently assumes -110, unsettled picks excluded from settled-record denominators,
categories remain strictly separate."""

from __future__ import annotations

import pytest

from nfl_predict.decision.records import (
    ASSUMED_HYPOTHETICAL_PRICE,
    compute_category_record,
    compute_hypothetical_assumed_price_record,
)


def _settled(category, settlement, price, pick_id="p") -> dict:
    return {"category": category, "status": "SETTLED", "settlement": settlement, "price": price, "pick_id": pick_id}


def _unsettled(category, pick_id="u") -> dict:
    return {"category": category, "status": "PUBLISHED", "settlement": None, "price": -110, "pick_id": pick_id}


def test_units_use_the_actual_recorded_price_not_a_default():
    picks = [_settled("BEST_BETS", "WIN", -200), _settled("BEST_BETS", "LOSS", -110, pick_id="p2")]
    record = compute_category_record(picks, "BEST_BETS")
    # WIN at -200 pays 0.5 units; LOSS costs 1.0 unit -> net -0.5, NOT what -110/-110 would give.
    assert record.total_units == pytest.approx(0.5 - 1.0)
    assert record.average_price == pytest.approx(-155.0)


def test_missing_price_key_raises_rather_than_assuming_110():
    picks = [{"category": "BEST_BETS", "status": "SETTLED", "settlement": "WIN", "pick_id": "p"}]  # no "price" key at all
    with pytest.raises(KeyError):
        compute_category_record(picks, "BEST_BETS")


def test_hypothetical_record_is_separate_and_labeled():
    picks = [_settled("BEST_BETS", "WIN", -200)]
    official = compute_category_record(picks, "BEST_BETS")
    hypothetical = compute_hypothetical_assumed_price_record(picks, "BEST_BETS")
    assert official.is_hypothetical is False
    assert hypothetical.is_hypothetical is True
    assert hypothetical.average_price == float(ASSUMED_HYPOTHETICAL_PRICE)
    assert official.average_price != hypothetical.average_price


def test_unsettled_picks_are_excluded_from_the_record():
    picks = [_settled("BEST_BETS", "WIN", -110), _unsettled("BEST_BETS")]
    record = compute_category_record(picks, "BEST_BETS")
    assert record.n_settled == 1
    assert record.wins == 1


def test_categories_remain_strictly_separate():
    picks = [
        _settled("BEST_BETS", "WIN", -110, pick_id="b1"),
        _settled("LEANS", "LOSS", -110, pick_id="l1"),
        _settled("ALL_MODEL_PREDICTIONS", "WIN", -110, pick_id="a1"),
    ]
    best_bets = compute_category_record(picks, "BEST_BETS")
    leans = compute_category_record(picks, "LEANS")
    all_preds = compute_category_record(picks, "ALL_MODEL_PREDICTIONS")

    assert best_bets.n_settled == 1 and best_bets.wins == 1 and best_bets.losses == 0
    assert leans.n_settled == 1 and leans.wins == 0 and leans.losses == 1
    assert all_preds.n_settled == 1 and all_preds.wins == 1


def test_empty_category_returns_none_fields_not_zero_division_error():
    record = compute_category_record([], "BEST_BETS")
    assert record.n_settled == 0
    assert record.win_rate is None
    assert record.total_units is None


def test_pushes_do_not_count_as_wins_or_losses_in_win_rate():
    picks = [_settled("BEST_BETS", "PUSH", -110, pick_id="p1"), _settled("BEST_BETS", "WIN", -110, pick_id="p2")]
    record = compute_category_record(picks, "BEST_BETS")
    assert record.pushes == 1
    assert record.wins == 1
    assert record.win_rate == 1.0  # 1 win / (1 win + 0 losses), push excluded from denominator
