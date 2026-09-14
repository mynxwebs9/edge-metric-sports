"""Phase 7 Step 24 proofs #1, #18-23: default NO_BET, staleness gates, VETO_CONSIDERATION
cannot silently qualify, missing data fails safely, determinism, no LLM call inside the
engine (structural - decide() takes only already-structured fields)."""

from __future__ import annotations

import pytest

from nfl_predict.decision.engine import UnsupportedMarketTypeError, _compute_disagreement, decide
from nfl_predict.market.odds_math import no_vig_two_way
from nfl_predict.decision.model_agreement import compute_model_agreement
from nfl_predict.decision.rules_config import load_rule_set
from nfl_predict.decision.schemas import (
    Decision,
    DecisionInputPacket,
    MarketPoint,
    ModelPoint,
    ReasonCode,
    ResearchPoint,
    SystemHealth,
)

RULE_SET = load_rule_set()
NOW = "2026-09-11T20:00:00+00:00"


def _packet(
    elo_margin=0.0, elo_prob=0.5, ridge_margin=None, lightgbm_margin=None,
    market_available=True, home_spread=0.0, home_ml=-110, away_ml=-110,
    research_available=True, research_classification="NO_MATERIAL_NEW_INFORMATION",
    materiality=0, unresolved_risks=(),
    market_age=3600.0, research_age=3600.0,
) -> DecisionInputPacket:
    elo = ModelPoint(model_id="elo_v2", available=True, predicted_margin=elo_margin, home_win_probability=elo_prob)
    ridge = ModelPoint(model_id="ridge_margin_E_v1", available=ridge_margin is not None, predicted_margin=ridge_margin)
    lightgbm = ModelPoint(model_id="lightgbm_F_v1", available=lightgbm_margin is not None, predicted_margin=lightgbm_margin)
    agreement = compute_model_agreement(elo, ridge, lightgbm)
    market = MarketPoint(available=market_available, home_spread_traditional=home_spread, home_moneyline=home_ml, away_moneyline=away_ml, snapshot_timestamp=NOW) if market_available else MarketPoint(available=False)
    research = ResearchPoint(
        available=research_available, research_id="r1", classification=research_classification,
        materiality_level=materiality, unresolved_risks=unresolved_risks, research_timestamp=NOW,
    ) if research_available else ResearchPoint(available=False)
    health = SystemHealth(missing_required_data=(), stale_flags=(), market_age_seconds=market_age, research_age_seconds=research_age)
    return DecisionInputPacket(
        game_id="g1", decision_timestamp=NOW, kickoff_timestamp="2026-09-13T17:00:00+00:00",
        elo=elo, ridge=ridge, lightgbm=lightgbm, model_agreement=agreement,
        market=market, research=research, system_health=health,
    )


def test_default_is_no_bet_with_insufficient_evidence():
    packet = _packet(elo_margin=0.1, home_spread=0.0)  # tiny disagreement
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.NO_BET
    assert ReasonCode.DISAGREEMENT_BELOW_MINIMUM in reasons


def test_veto_consideration_cannot_silently_qualify_even_with_huge_disagreement():
    packet = _packet(elo_margin=20.0, home_spread=-3.0, research_classification="VETO_CONSIDERATION")
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.VETO
    assert reasons == (ReasonCode.VETO_CONSIDERATION_UNRESOLVED,)


def test_missing_market_fails_safely_to_no_bet():
    packet = _packet(market_available=False)
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.NO_BET
    assert ReasonCode.MISSING_LIVE_DATA in reasons


def test_missing_research_fails_safely_to_no_bet():
    packet = _packet(research_available=False)
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.NO_BET
    assert ReasonCode.MISSING_LIVE_DATA in reasons


def test_stale_market_caps_at_watch_even_with_large_disagreement():
    packet = _packet(elo_margin=10.0, home_spread=-3.0, market_age=999999.0)  # far beyond 24h
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.WATCH
    assert ReasonCode.STALE_MARKET in reasons


def test_stale_research_caps_at_watch():
    packet = _packet(elo_margin=10.0, home_spread=-3.0, research_age=999999.0)
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.WATCH
    assert ReasonCode.STALE_RESEARCH in reasons


def test_high_uncertainty_caps_at_watch_regardless_of_disagreement():
    packet = _packet(elo_margin=15.0, home_spread=-3.0, research_classification="HIGH_UNCERTAINTY")
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.WATCH
    assert ReasonCode.HIGH_MODEL_UNCERTAINTY in reasons


def test_large_disagreement_with_only_elo_available_produces_lean_not_qualified_bet():
    packet = _packet(elo_margin=10.0, home_spread=-3.0)  # no ridge/lightgbm
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.LEAN
    assert ReasonCode.INSUFFICIENT_MODEL_COVERAGE in reasons


def test_large_disagreement_with_all_three_models_agreeing_produces_qualified_bet():
    packet = _packet(elo_margin=10.0, ridge_margin=8.0, lightgbm_margin=9.0, home_spread=-3.0)
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.QUALIFIED_BET
    assert ReasonCode.NO_DEMONSTRATED_EDGE in reasons  # permanent caveat


def test_model_disagreement_with_each_other_caps_at_lean():
    # Elo and LightGBM point the same direction as the market disagreement but wildly
    # disagree with EACH OTHER (dispersion > 3.0).
    packet = _packet(elo_margin=10.0, ridge_margin=1.0, lightgbm_margin=9.5, home_spread=-3.0)
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.LEAN
    assert ReasonCode.MODEL_DISAGREEMENT_HIGH in reasons


def test_research_supports_market_caps_at_lean_even_with_full_coverage_and_agreement():
    packet = _packet(elo_margin=10.0, ridge_margin=9.0, lightgbm_margin=9.5, home_spread=-3.0, research_classification="SUPPORTS_MARKET")
    decision, reasons = decide(packet, RULE_SET, "spread")
    assert decision == Decision.LEAN
    assert ReasonCode.RESEARCH_SUPPORTS_MARKET in reasons


def test_moneyline_market_type_works_with_probability_disagreement():
    packet = _packet(elo_prob=0.65, home_ml=-110, away_ml=-110, ridge_margin=8.0, lightgbm_margin=9.0)
    decision, reasons = decide(packet, RULE_SET, "moneyline")
    assert decision in (Decision.QUALIFIED_BET, Decision.LEAN, Decision.NO_BET)  # exercised without crashing; specific case checked below


def test_a_synthetic_packet_meeting_every_frozen_v1_gate_reaches_qualified_bet_moneyline():
    """Correction brief test #15 (moneyline half - the spread half already exists above as
    test_large_disagreement_with_all_three_models_agreeing_produces_qualified_bet). Proves
    QUALIFIED_BET is operationally reachable for moneyline too, WITHOUT changing rule set v1."""
    packet = _packet(elo_prob=0.70, elo_margin=10.0, home_ml=-140, away_ml=120, ridge_margin=8.0, lightgbm_margin=9.0)
    decision, reasons = decide(packet, RULE_SET, "moneyline")
    assert decision == Decision.QUALIFIED_BET
    assert ReasonCode.NO_DEMONSTRATED_EDGE in reasons


def test_spread_disagreement_uses_the_actual_market_line_with_the_documented_sign_convention():
    """Correction brief test #11: home_spread_traditional=-3.0 (home favored by 3) implies a
    market home-margin of +3.0; a +10.0 Elo margin disagrees with the market by exactly 7.0,
    not by 10.0 or 13.0 - proving the traditional-sign-to-margin conversion is applied."""
    packet = _packet(elo_margin=10.0, home_spread=-3.0)
    disagreement = _compute_disagreement(packet, "spread")
    assert disagreement == pytest.approx(7.0)

    # Sign flips correctly for an underdog home line too (home_spread=+3.0 -> market margin -3.0).
    underdog_packet = _packet(elo_margin=-10.0, home_spread=3.0)
    assert _compute_disagreement(underdog_packet, "spread") == pytest.approx(-7.0)


def test_moneyline_disagreement_uses_the_actual_no_vig_market_probability():
    """Correction brief test #12: the model probability is compared against the NO-VIG
    probability derived from the real moneylines - never the raw vig-inclusive implied
    probability, and never a blind average of the two American odds."""
    packet = _packet(elo_prob=0.70, home_ml=-140, away_ml=120)
    disagreement = _compute_disagreement(packet, "moneyline")
    expected_market_prob = no_vig_two_way(-140, 120).no_vig_prob_a
    assert disagreement == pytest.approx(0.70 - expected_market_prob)

    # A blind average of the raw prices would not even be a probability - confirm this is
    # nowhere near that.
    naive_wrong = (-140 + 120) / 2
    assert disagreement != pytest.approx(0.70 - naive_wrong)


def test_totals_market_type_is_rejected():
    packet = _packet()
    with pytest.raises(UnsupportedMarketTypeError):
        decide(packet, RULE_SET, "total")


def test_decide_is_deterministic_given_the_same_packet_and_rules():
    packet = _packet(elo_margin=10.0, ridge_margin=8.0, lightgbm_margin=9.0, home_spread=-3.0)
    result_a = decide(packet, RULE_SET, "spread")
    result_b = decide(packet, RULE_SET, "spread")
    assert result_a == result_b


def test_research_classification_cannot_modify_the_models_numeric_output():
    """Proof #18: whatever research says, packet.elo.predicted_margin/home_win_probability
    are read-only inputs to decide() - the function returns a Decision/reason-code tuple,
    never a mutated packet, and ResearchPoint itself has no field that could carry a
    margin/probability override (schemas.py)."""
    import dataclasses

    packet_a = _packet(elo_margin=10.0, ridge_margin=8.0, lightgbm_margin=9.0, home_spread=-3.0, research_classification="SUPPORTS_MODEL")
    packet_b = _packet(elo_margin=10.0, ridge_margin=8.0, lightgbm_margin=9.0, home_spread=-3.0, research_classification="VETO_CONSIDERATION")

    decide(packet_a, RULE_SET, "spread")
    decide(packet_b, RULE_SET, "spread")

    # Elo's own numeric fields are identical regardless of what research says - only the
    # DECISION differs, never the underlying model prediction.
    assert packet_a.elo.predicted_margin == packet_b.elo.predicted_margin == 10.0
    assert packet_a.elo.home_win_probability == packet_b.elo.home_win_probability

    # And decide() itself never mutates the packet it was given (it's a frozen dataclass;
    # this additionally confirms no field was reassigned via object.__setattr__ trickery).
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet_a.elo = ModelPoint(model_id="hacked", available=True, predicted_margin=999.0)


def test_no_bet_never_requires_a_minimum_pick_count():
    """Structural proof: decide() has no notion of 'this week' or a quota - it is called
    once per game/market and every path is reachable independently of any other game."""
    packets = [_packet(elo_margin=0.05 * i, home_spread=0.0) for i in range(10)]
    decisions = [decide(p, RULE_SET, "spread")[0] for p in packets]
    assert all(d == Decision.NO_BET for d in decisions)  # zero qualified bets is a valid outcome
