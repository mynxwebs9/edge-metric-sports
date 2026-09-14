"""Phase 7 Step 5: builds the immutable `DecisionInputPacket` from already-existing
structured data - Phase 6's stored research run (`input.json`/`findings.json`/
`evaluation.json`) - reusing those values verbatim rather than recomputing anything. No
decision may use information timestamped after `decision_timestamp`
(`DecisionTimestampViolationError` below enforces this explicitly, in addition to
`historical_guard`'s protection further upstream in Phase 6).
"""

from __future__ import annotations

from datetime import datetime, timezone

from nfl_predict.research.storage import read_research_run

from nfl_predict.decision.model_agreement import compute_model_agreement
from nfl_predict.decision.schemas import (
    DecisionInputPacket,
    MarketPoint,
    ModelPoint,
    ResearchPoint,
    SystemHealth,
)
from nfl_predict.decision.staleness import compute_age_seconds


class DecisionTimestampViolationError(Exception):
    """Raised when an input's own timestamp is after the decision_timestamp - a decision
    may never use information from the future relative to when it claims to be made."""


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_research_point_from_stored_run(stored: dict, now_iso: str) -> ResearchPoint:
    """Extracted so both `build_decision_packet_from_research_run` (Phase 7, reads a
    historical/pilot research run) and Phase 8A's live orchestrator (which persists a real
    research run just before building the packet) can turn a `read_research_run()` result
    into a `ResearchPoint` identically - one implementation, never two competing ones."""
    now_dt = _parse(now_iso)
    manifest = stored["manifest"]
    if manifest["findings_kind"] != "findings":
        return ResearchPoint(available=False)

    findings_dict = stored["findings"]
    evaluation_dict = stored.get("evaluation")
    research_ts = findings_dict["research_timestamp"]
    if _parse(research_ts) > now_dt:
        raise DecisionTimestampViolationError(f"Research timestamp {research_ts} is after decision_timestamp {now_iso}")

    classification = evaluation_dict["final_classification"] if evaluation_dict else findings_dict["research_classification"]
    materiality = evaluation_dict["overall_materiality"] if evaluation_dict else None
    source_quality_note = evaluation_dict["notes"] if evaluation_dict else None

    return ResearchPoint(
        available=True, research_id=findings_dict["research_id"], classification=classification,
        materiality_level=materiality, source_quality_note=source_quality_note,
        unresolved_risks=tuple(findings_dict.get("missing_information", [])), research_timestamp=research_ts,
    )


def build_decision_packet_from_research_run(
    season: int, week: int, game_id: str, run_id: str, decision_timestamp: str | None = None,
) -> DecisionInputPacket:
    stored = read_research_run(season, week, game_id, run_id)
    packet_dict = stored["input"]
    manifest = stored["manifest"]
    now_iso = decision_timestamp or datetime.now(timezone.utc).isoformat()
    now_dt = _parse(now_iso)

    elo_dict = packet_dict["elo"]
    elo = ModelPoint(model_id=elo_dict["model_id"], available=elo_dict["available"], predicted_margin=elo_dict["predicted_margin"], home_win_probability=elo_dict["home_win_probability"], uncertainty_note=elo_dict.get("uncertainty_note"))
    ridge_dict = packet_dict["ridge"]
    ridge = ModelPoint(model_id=ridge_dict["model_id"], available=ridge_dict["available"], predicted_margin=ridge_dict["predicted_margin"], home_win_probability=ridge_dict.get("home_win_probability"))
    lightgbm_dict = packet_dict["lightgbm"]
    lightgbm = ModelPoint(model_id=lightgbm_dict["model_id"], available=lightgbm_dict["available"], predicted_margin=lightgbm_dict["predicted_margin"], home_win_probability=lightgbm_dict.get("home_win_probability"))
    agreement = compute_model_agreement(elo, ridge, lightgbm)

    market_dict = packet_dict["market"]
    market = MarketPoint(
        available=market_dict["available"],
        source="nflverse-derived (see nfl_predict.market)" if market_dict["available"] else None,
        home_spread_traditional=market_dict.get("home_spread_traditional"),
        home_spread_price=market_dict.get("home_spread_price"), away_spread_price=market_dict.get("away_spread_price"),
        home_moneyline=market_dict.get("home_moneyline"), away_moneyline=market_dict.get("away_moneyline"),
        no_vig_home_win_probability=market_dict.get("no_vig_home_win_probability"),
        snapshot_timestamp=market_dict.get("snapshot_timestamp"),
    )
    if market.snapshot_timestamp and _parse(market.snapshot_timestamp) > now_dt:
        raise DecisionTimestampViolationError(f"Market snapshot timestamp {market.snapshot_timestamp} is after decision_timestamp {now_iso}")

    research = build_research_point_from_stored_run(stored, now_iso)

    missing = []
    if not market.available:
        missing.append("market")
    if not research.available:
        missing.append("research")
    if not ridge.available:
        missing.append("ridge")
    if not lightgbm.available:
        missing.append("lightgbm")

    market_age = compute_age_seconds(market.snapshot_timestamp, now_iso) if market.available else None
    research_age = compute_age_seconds(research.research_timestamp, now_iso) if research.available else None

    health = SystemHealth(missing_required_data=tuple(missing), stale_flags=(), market_age_seconds=market_age, research_age_seconds=research_age)

    return DecisionInputPacket(
        game_id=game_id, decision_timestamp=now_iso, kickoff_timestamp=packet_dict.get("kickoff_timestamp"),
        elo=elo, ridge=ridge, lightgbm=lightgbm, model_agreement=agreement,
        market=market, research=research, system_health=health,
    )
