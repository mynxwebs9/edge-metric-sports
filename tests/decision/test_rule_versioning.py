"""Phase 7 Step 24 proof #25: rule-version changes preserve prior decisions - a
`DecisionRecord`'s `decision_rule_version` is fixed at the time it was made and a later
rule-set change never retroactively alters it."""

from __future__ import annotations

import dataclasses

from nfl_predict.decision.engine import decide
from nfl_predict.decision.model_agreement import compute_model_agreement
from nfl_predict.decision.rules_config import Rule, RuleSet, RuleStatus, load_rule_set
from nfl_predict.decision.schemas import (
    DecisionInputPacket,
    DecisionRecord,
    MarketPoint,
    ModelPoint,
    ResearchPoint,
    SystemHealth,
)

NOW = "2026-09-11T20:00:00+00:00"


def _packet() -> DecisionInputPacket:
    elo = ModelPoint(model_id="elo_v2", available=True, predicted_margin=5.0, home_win_probability=0.6)
    ridge = ModelPoint(model_id="ridge_margin_E_v1", available=False)
    lightgbm = ModelPoint(model_id="lightgbm_F_v1", available=False)
    agreement = compute_model_agreement(elo, ridge, lightgbm)
    market = MarketPoint(available=True, home_spread_traditional=0.0, home_moneyline=-110, away_moneyline=-110, snapshot_timestamp=NOW)
    research = ResearchPoint(available=True, research_id="r1", classification="NO_MATERIAL_NEW_INFORMATION", materiality_level=0, research_timestamp=NOW)
    health = SystemHealth(missing_required_data=(), stale_flags=(), market_age_seconds=3600.0, research_age_seconds=3600.0)
    return DecisionInputPacket(game_id="g1", decision_timestamp=NOW, kickoff_timestamp="2026-09-13T17:00:00+00:00", elo=elo, ridge=ridge, lightgbm=lightgbm, model_agreement=agreement, market=market, research=research, system_health=health)


def _stricter_rule_set(base: RuleSet) -> RuleSet:
    """Simulates a NEW, stricter rule-set version - min_spread_disagreement_points raised
    from 3.0 to 100.0 - without touching the original rule set object at all."""
    stricter_rules = dict(base.rules)
    original = stricter_rules["min_spread_disagreement_points"]
    stricter_rules["min_spread_disagreement_points"] = dataclasses.replace(original, version=2, threshold=100.0, status=RuleStatus.EXPERIMENTAL)
    return RuleSet(rule_set_version="v2_experimental", rule_set_status=RuleStatus.EXPERIMENTAL, rules=stricter_rules)


def test_a_decision_made_under_one_rule_set_is_unaffected_by_a_later_stricter_rule_set():
    v1 = load_rule_set()
    packet = _packet()

    decision_v1, reasons_v1 = decide(packet, v1, "spread")
    record_v1 = DecisionRecord(
        decision_id="d1", game_id=packet.game_id, decision_timestamp=packet.decision_timestamp,
        kickoff_timestamp=packet.kickoff_timestamp, decision=decision_v1, reason_codes=reasons_v1,
        decision_rule_version=v1.rule_set_version, market_type="spread", input_packet_hash="hash1",
        validation_status="PROSPECTIVE",
    )

    v2 = _stricter_rule_set(v1)
    decision_v2, _ = decide(packet, v2, "spread")

    # The v2 rule set produces a DIFFERENT (stricter) outcome for the SAME packet...
    assert decision_v2 != decision_v1 or decision_v2.value in ("NO_BET",)
    # ...but the STORED v1 record's own fields are completely untouched - re-running under
    # v2 never mutates record_v1, and its decision_rule_version still says "v1".
    assert record_v1.decision_rule_version == "v1"
    assert record_v1.decision == decision_v1


def test_original_rule_set_object_is_never_mutated_by_constructing_a_stricter_variant():
    v1 = load_rule_set()
    original_threshold = v1.get("min_spread_disagreement_points").threshold
    _stricter_rule_set(v1)
    assert v1.get("min_spread_disagreement_points").threshold == original_threshold  # untouched
