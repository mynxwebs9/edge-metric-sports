"""Phase 7 Step 1/23: the deterministic decision engine.

Given the same `DecisionInputPacket` and the same rule set, `decide()` ALWAYS returns the
same `Decision` and `reason_codes` - no LLM call happens inside this function, no random
number, no wall-clock read (the packet already carries every timestamp it needs). The LLM
research layer (Phase 6) runs strictly before this and enters only as already-parsed,
already-classified structured fields (`ResearchPoint`) - never as a fresh prompt call from
inside this module.

**Default is `NO_BET`** (Step 1) - every return path that isn't an explicit qualification
falls through to it. **No rule here requires a minimum number of weekly picks** (Step 2) -
`decide()` is called once per game/market and has no notion of "this week's quota."
"""

from __future__ import annotations

from nfl_predict.decision.rules_config import RuleSet
from nfl_predict.decision.schemas import Decision, DecisionInputPacket, ReasonCode
from nfl_predict.decision.staleness import is_stale
from nfl_predict.market.odds_math import no_vig_two_way

SUPPORTED_MARKET_TYPES = ("spread", "moneyline")


class UnsupportedMarketTypeError(Exception):
    """Raised for any market_type other than 'spread'/'moneyline' - totals are explicitly
    not supported yet (Step 16: Phase 4/5 showed insufficient independent totals signal)."""


def _compute_disagreement(packet: DecisionInputPacket, market_type: str) -> float | None:
    """Spread: points, home-margin space (Elo's predicted margin minus the market's
    implied margin - identical convention to `nfl_predict.market.disagreement.edge_points`).
    Moneyline: probability difference (Elo's home win probability minus the market's no-vig
    home win probability)."""
    if not packet.market.available or not packet.elo.available or packet.elo.predicted_margin is None:
        return None

    if market_type == "spread":
        if packet.market.home_spread_traditional is None:
            return None
        market_implied_margin = -packet.market.home_spread_traditional
        return packet.elo.predicted_margin - market_implied_margin

    if market_type == "moneyline":
        if packet.elo.home_win_probability is None:
            return None
        if packet.market.no_vig_home_win_probability is not None:
            market_prob = packet.market.no_vig_home_win_probability
        elif packet.market.home_moneyline is not None and packet.market.away_moneyline is not None:
            market_prob = no_vig_two_way(packet.market.home_moneyline, packet.market.away_moneyline).no_vig_prob_a
        else:
            return None
        return packet.elo.home_win_probability - market_prob

    raise UnsupportedMarketTypeError(f"market_type={market_type!r} is not supported - only {SUPPORTED_MARKET_TYPES} (Step 16: no totals yet)")


def decide(packet: DecisionInputPacket, rule_set: RuleSet, market_type: str) -> tuple[Decision, tuple[ReasonCode, ...]]:
    if market_type not in SUPPORTED_MARKET_TYPES:
        raise UnsupportedMarketTypeError(f"market_type={market_type!r} is not supported - only {SUPPORTED_MARKET_TYPES} (Step 16: no totals yet)")

    reasons: list[ReasonCode] = []

    # --- Hard block: an unresolved VETO_CONSIDERATION always wins, regardless of anything else. ---
    if packet.research.available and packet.research.classification == "VETO_CONSIDERATION":
        return Decision.VETO, (ReasonCode.VETO_CONSIDERATION_UNRESOLVED,)

    # --- Cannot assess at all without both a market snapshot and a research pass. ---
    if not packet.market.available or not packet.research.available:
        return Decision.NO_BET, (ReasonCode.MISSING_LIVE_DATA,)

    # --- Staleness caps at WATCH - data exists but may no longer be current. ---
    market_max_age_hours = rule_set.get("max_market_snapshot_age_hours").threshold
    research_max_age_hours = rule_set.get("max_research_age_hours").threshold
    market_stale = is_stale(packet.system_health.market_age_seconds, market_max_age_hours)
    research_stale = is_stale(packet.system_health.research_age_seconds, research_max_age_hours)
    if market_stale:
        reasons.append(ReasonCode.STALE_MARKET)
    if research_stale:
        reasons.append(ReasonCode.STALE_RESEARCH)
    if market_stale or research_stale:
        return Decision.WATCH, tuple(reasons)

    # --- HIGH_UNCERTAINTY research caps at WATCH, regardless of disagreement. ---
    if packet.research.classification == "HIGH_UNCERTAINTY":
        return Decision.WATCH, (ReasonCode.HIGH_MODEL_UNCERTAINTY,)

    disagreement = _compute_disagreement(packet, market_type)
    if disagreement is None:
        return Decision.NO_BET, (ReasonCode.MISSING_LIVE_DATA,)

    min_disagreement_rule_id = "min_spread_disagreement_points" if market_type == "spread" else "min_moneyline_disagreement_probability"
    min_disagreement = rule_set.get(min_disagreement_rule_id).threshold

    if abs(disagreement) < min_disagreement:
        return Decision.NO_BET, (ReasonCode.DISAGREEMENT_BELOW_MINIMUM,)

    # --- A genuine directional candidate. Now check what disqualifies full publication. ---
    reasons.append(ReasonCode.MODEL_MARKET_DISAGREEMENT)

    coverage_ok = packet.model_agreement.n_models_available == 3 and packet.model_agreement.all_agree_on_direction is True
    if not coverage_ok:
        reasons.append(ReasonCode.INSUFFICIENT_MODEL_COVERAGE)
        _append_research_context(packet, reasons)
        return Decision.LEAN, tuple(reasons)

    max_dispersion = rule_set.get("max_model_margin_dispersion_points").threshold
    if packet.model_agreement.margin_dispersion is not None and packet.model_agreement.margin_dispersion > max_dispersion:
        reasons.append(ReasonCode.MODEL_DISAGREEMENT_HIGH)
        _append_research_context(packet, reasons)
        return Decision.LEAN, tuple(reasons)

    reasons.append(ReasonCode.MODEL_AGREEMENT_HIGH)

    if packet.research.classification == "SUPPORTS_MARKET":
        reasons.append(ReasonCode.RESEARCH_SUPPORTS_MARKET)
        return Decision.LEAN, tuple(reasons)

    _append_research_context(packet, reasons)
    reasons.append(ReasonCode.NO_DEMONSTRATED_EDGE)  # permanent caveat baked into every QUALIFIED_BET
    return Decision.QUALIFIED_BET, tuple(reasons)


def _append_research_context(packet: DecisionInputPacket, reasons: list[ReasonCode]) -> None:
    classification = packet.research.classification
    if classification == "SUPPORTS_MODEL":
        reasons.append(ReasonCode.RESEARCH_SUPPORTS_MODEL)
    elif classification == "SUPPORTS_MARKET":
        reasons.append(ReasonCode.RESEARCH_SUPPORTS_MARKET)
    elif classification == "MIXED":
        reasons.append(ReasonCode.RESEARCH_MIXED)
    elif classification == "NO_MATERIAL_NEW_INFORMATION":
        reasons.append(ReasonCode.RESEARCH_NO_MATERIAL_INFORMATION)

    if packet.research.materiality_level is not None and packet.research.materiality_level >= 3:
        reasons.append(ReasonCode.MAJOR_INJURY_RISK)
    if any("qb" in risk.lower() for risk in packet.research.unresolved_risks):
        reasons.append(ReasonCode.QB_UNCERTAINTY)
