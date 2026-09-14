"""Phase 7 Step 24 proofs #3-7, #26-27: published picks cannot be deleted or have their
line/price altered; category (Best Bet vs Lean) is fixed at publish time; void/correction
events append rather than mutate; outcomes never touch the original snapshots."""

from __future__ import annotations

import pytest

from nfl_predict.decision.pick_ledger import (
    InvalidPickStateTransitionError,
    InvalidVoidReasonError,
    PickAlreadyExistsError,
    PickCategory,
    PickNotFoundError,
    PublishedPick,
    SettlementResult,
    publish_pick,
    read_current_picks,
    settle_pick,
    void_pick,
)


def _pick(pick_id="p1", category=PickCategory.BEST_BETS.value, line=-3.0, price=-110) -> PublishedPick:
    return PublishedPick(
        pick_id=pick_id, game_id="g1", published_at="2026-09-11T20:00:00+00:00", kickoff_at="2026-09-13T17:00:00+00:00",
        decision_id="d1", rule_version="v1", category=category, market_type="spread", selection="home",
        line=line, price=price, sportsbook_or_source="test_book", market_snapshot_id="m1",
        model_prediction_snapshot={"elo_margin": 10.0}, research_snapshot_id="r1", validation_status="PROSPECTIVE",
    )


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_publishing_then_reading_round_trips(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick())
    picks = read_current_picks()
    assert len(picks) == 1
    assert picks[0]["line"] == -3.0
    assert picks[0]["price"] == -110
    assert picks[0]["status"] == "PUBLISHED"


def test_a_pick_id_cannot_be_republished(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick())
    with pytest.raises(PickAlreadyExistsError):
        publish_pick(_pick())


def test_there_is_no_delete_function_and_settled_picks_still_appear(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    import nfl_predict.decision.pick_ledger as pl

    assert not hasattr(pl, "delete_pick")
    publish_pick(_pick())
    settle_pick("p1", SettlementResult.LOSS, "2026-09-14T00:00:00+00:00", "test_source")
    picks = read_current_picks()
    assert len(picks) == 1  # still present - a loss never removes a pick
    assert picks[0]["status"] == "SETTLED"
    assert picks[0]["settlement"] == "LOSS"


def test_settling_never_alters_the_original_line_or_price(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick(line=-3.5, price=-115))
    settle_pick("p1", SettlementResult.WIN, "2026-09-14T00:00:00+00:00", "test_source")
    picks = read_current_picks()
    assert picks[0]["line"] == -3.5
    assert picks[0]["price"] == -115


def test_voiding_never_alters_the_original_line_or_price(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick(line=-3.5, price=-115))
    void_pick("p1", "GAME_CANCELLED", "2026-09-14T00:00:00+00:00", "test_operator")
    picks = read_current_picks()
    assert picks[0]["line"] == -3.5
    assert picks[0]["price"] == -115
    assert picks[0]["status"] == "VOID"


def test_category_cannot_be_changed_after_publication_no_function_exists(tmp_path, monkeypatch):
    """There is no reclassify/promote/demote function anywhere in this module - a losing
    Best Bet cannot become a Lean and a winning Lean cannot become a Best Bet."""
    import nfl_predict.decision.pick_ledger as pl

    assert not hasattr(pl, "reclassify_pick")
    assert not hasattr(pl, "promote_pick")
    assert not hasattr(pl, "demote_pick")

    _patch(monkeypatch, tmp_path)
    publish_pick(_pick(category=PickCategory.LEANS.value))
    settle_pick("p1", SettlementResult.WIN, "2026-09-14T00:00:00+00:00", "test_source")
    picks = read_current_picks()
    assert picks[0]["category"] == PickCategory.LEANS.value  # still a Lean despite winning


def test_a_losing_best_bet_cannot_be_voided(tmp_path, monkeypatch):
    """Not literally blocked by category (the code can't know intent), but this test
    documents the policy check: void_reason must be one of the allowed enumerated reasons,
    none of which is "the pick lost" - only an allowed operational reason is accepted."""
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick())
    with pytest.raises(InvalidVoidReasonError):
        void_pick("p1", "MODEL_WAS_WRONG", "2026-09-14T00:00:00+00:00", "test_operator")


def test_cannot_settle_a_pick_twice(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick())
    settle_pick("p1", SettlementResult.WIN, "2026-09-14T00:00:00+00:00", "test_source")
    with pytest.raises(InvalidPickStateTransitionError):
        settle_pick("p1", SettlementResult.LOSS, "2026-09-14T00:00:00+00:00", "test_source")


def test_cannot_void_an_already_settled_pick(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick())
    settle_pick("p1", SettlementResult.WIN, "2026-09-14T00:00:00+00:00", "test_source")
    with pytest.raises(InvalidPickStateTransitionError):
        void_pick("p1", "GAME_CANCELLED", "2026-09-14T00:00:00+00:00", "test_operator")


def test_settling_a_nonexistent_pick_raises_not_found(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    with pytest.raises(PickNotFoundError):
        settle_pick("nonexistent", SettlementResult.WIN, "2026-09-14T00:00:00+00:00", "test_source")


def test_model_prediction_snapshot_is_never_touched_by_settlement(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(_pick())
    settle_pick("p1", SettlementResult.LOSS, "2026-09-14T00:00:00+00:00", "test_source")
    picks = read_current_picks()
    assert picks[0]["model_prediction_snapshot"] == {"elo_margin": 10.0}
