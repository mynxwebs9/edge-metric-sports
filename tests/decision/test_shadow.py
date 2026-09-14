"""Phase 7 Step 24 proof #24: shadow decisions never affect the official record."""

from __future__ import annotations

from nfl_predict.decision.pick_ledger import PickCategory, PublishedPick, publish_pick, read_current_picks
from nfl_predict.decision.shadow import (
    ShadowDecisionRecord,
    compare_shadow_to_official,
    read_shadow_decisions,
    record_shadow_decision,
)


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.decision.shadow.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_shadow_decisions_are_stored_separately_from_official_picks(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    record_shadow_decision(ShadowDecisionRecord(
        shadow_decision_id="s1", game_id="g1", market_type="spread", rule_set_version="v2_experimental",
        decision="QUALIFIED_BET", reason_codes=("MODEL_MARKET_DISAGREEMENT",),
        decision_timestamp="2026-09-11T20:00:00+00:00", input_packet_hash="hash1",
    ))
    official_picks = read_current_picks()
    assert official_picks == []  # the shadow decision never touched the official ledger


def test_shadow_decisions_are_readable_by_rule_set_version(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    record_shadow_decision(ShadowDecisionRecord(shadow_decision_id="s1", game_id="g1", market_type="spread", rule_set_version="v2_experimental", decision="QUALIFIED_BET", reason_codes=(), decision_timestamp="2026-09-11T20:00:00+00:00", input_packet_hash="h1"))
    record_shadow_decision(ShadowDecisionRecord(shadow_decision_id="s2", game_id="g2", market_type="spread", rule_set_version="v2_experimental", decision="NO_BET", reason_codes=(), decision_timestamp="2026-09-11T20:00:00+00:00", input_packet_hash="h2"))

    decisions = read_shadow_decisions("v2_experimental")
    assert len(decisions) == 2
    assert read_shadow_decisions("v3_nonexistent") == []


def test_comparing_shadow_to_official_is_read_only_and_does_not_create_picks(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_pick(PublishedPick(
        pick_id="p1", game_id="g1", published_at="2026-09-11T20:00:00+00:00", kickoff_at="2026-09-13T17:00:00+00:00",
        decision_id="d1", rule_version="v1", category=PickCategory.BEST_BETS.value, market_type="spread", selection="home",
        line=-3.0, price=-110, sportsbook_or_source="test", market_snapshot_id="m1", model_prediction_snapshot={},
        research_snapshot_id="r1", validation_status="PROSPECTIVE",
    ))
    shadow_decisions = [{"game_id": "g1", "market_type": "spread", "decision": "LEAN"}]
    comparison = compare_shadow_to_official(shadow_decisions, read_current_picks())
    assert comparison["n_matched_to_official_picks"] == 1
    assert read_current_picks()[0]["pick_id"] == "p1"  # unchanged by the comparison
    assert len(read_current_picks()) == 1  # comparison created no new picks
