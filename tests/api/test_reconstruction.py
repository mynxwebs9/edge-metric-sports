"""Phase 8B, item 6's explicit requirement: historical market reconstruction must NEVER
blend every snapshot batch persisted for a `provider_event_id` - it must use either the
single most recent batch ("current") or the EXACT batch a specific persisted decision
referenced ("historical"), never a mix. Also covers the real `latest_research_summary`
sorting bug this endpoint's own build caught (sorting by `run_id` string instead of each
run's real `research_timestamp`).
"""

from __future__ import annotations

from pathlib import Path

from nfl_predict.api.reconstruction import (
    current_best_bets,
    latest_decisions_for_game,
    latest_market_point,
    latest_model_prediction,
    latest_research_summary,
    market_point_for_decision_record,
    system_pick_for_game,
)
from nfl_predict.decision.decision_log import append_decision_record
from nfl_predict.decision.pick_ledger import PublishedPick, publish_pick
from nfl_predict.decision.schemas import Decision, DecisionRecord, ReasonCode
from nfl_predict.live.market_consensus import compute_market_consensus
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.live.prediction_publication import PublicModelPrediction, PublicationState, publish_model_prediction
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider
from nfl_predict.research.input_packet import MarketContext, ModelPrediction, ResearchInputPacket
from nfl_predict.research.llm_provider import FixtureLLMProvider
from nfl_predict.research.run_research import run_research_for_game

GAME_ID = "2026_01_DEN_KC"
SEASON, WEEK = 2026, 1
LLM_FIXTURE = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "llm_fixture.json"


class _FakeOddsProvider(OddsProvider):
    def __init__(self, events, markets_by_event):
        self._events = events
        self._markets = markets_by_event

    def get_events(self):
        return self._events

    def get_markets_for_sport(self):
        return [s for snaps in self._markets.values() for s in snaps]


def _snap(event_id, bookmaker, fetched_at, home_spread, home_ml, away_ml):
    return OddsMarketSnapshot(
        provider_event_id=event_id, fetched_at=fetched_at, bookmaker=bookmaker,
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread,
        home_spread_price=-110, away_spread_price=-110, home_moneyline=home_ml, away_moneyline=away_ml,
        total_line=44.5, over_price=-110, under_price=-110,
    )


def _fetch(fetched_at, home_spread, home_ml=-150, away_ml=130):
    events = [OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")]
    markets = {"evt_denkc": [_snap("evt_denkc", "book_a", fetched_at, home_spread, home_ml, away_ml)]}
    return fetch_and_snapshot_live_odds(_FakeOddsProvider(events, markets), season=SEASON)


def test_latest_market_point_uses_only_the_most_recent_batch_never_a_blend(api_data_dir):
    _fetch("2026-09-14T10:00:00+00:00", home_spread=-1.0)  # an older line
    _fetch("2026-09-14T17:00:00+00:00", home_spread=-3.0)  # the current line

    market = latest_market_point(SEASON, WEEK, GAME_ID)

    assert market.available is True
    assert market.home_spread_traditional == -3.0  # only the newest batch, not a median of [-1.0, -3.0]
    assert market.snapshot_timestamp == "2026-09-14T17:00:00+00:00"


def test_market_point_for_decision_record_uses_the_exact_referenced_batch_not_the_latest(api_data_dir):
    """The core Phase 8A lesson: a decision made against an OLDER line must reconstruct that
    OLDER line exactly, even though a NEWER batch now exists for the same provider_event_id."""
    early_result = _fetch("2026-09-14T10:00:00+00:00", home_spread=-1.0)
    early_market = early_result.games[GAME_ID].market
    _fetch("2026-09-14T17:00:00+00:00", home_spread=-3.0)  # a later refresh - must NOT leak into the reconstruction below

    record = {
        "market_provider_event_id": early_market.provider_event_id,
        "market_snapshot_timestamp": early_market.snapshot_timestamp,
        "market_snapshot_reference": early_market.market_snapshot_reference,
    }
    reconstructed = market_point_for_decision_record(record)

    assert reconstructed is not None
    assert reconstructed.home_spread_traditional == -1.0  # the OLD line, not the new -3.0
    assert reconstructed.snapshot_timestamp == "2026-09-14T10:00:00+00:00"
    assert reconstructed.market_snapshot_reference == early_market.market_snapshot_reference


def test_market_point_for_decision_record_returns_none_for_a_pre_correction_record_with_no_reference(api_data_dir):
    _fetch("2026-09-14T10:00:00+00:00", home_spread=-1.0)
    old_style_record = {"market_provider_event_id": None, "market_snapshot_timestamp": None}
    assert market_point_for_decision_record(old_style_record) is None


def test_market_point_for_decision_record_refuses_a_tampered_reference(api_data_dir):
    result = _fetch("2026-09-14T10:00:00+00:00", home_spread=-1.0)
    market = result.games[GAME_ID].market
    tampered = {
        "market_provider_event_id": market.provider_event_id,
        "market_snapshot_timestamp": market.snapshot_timestamp,
        "market_snapshot_reference": "not_the_real_reference",
    }
    assert market_point_for_decision_record(tampered) is None


def _research_packet() -> ResearchInputPacket:
    return ResearchInputPacket(
        game_id=GAME_ID, season=SEASON, week=WEEK, season_type="REG",
        home_team_id="2310", away_team_id="1400", kickoff_timestamp="2026-09-14T20:15:00+00:00",
        packet_generated_at="2026-09-11T10:00:00+00:00",
        elo=ModelPrediction(model_id="elo_v2", available=True, predicted_margin=-3.2, home_win_probability=0.42),
        ridge=ModelPrediction(model_id="ridge_margin_E_v1", available=False, unavailable_reason="test"),
        lightgbm=ModelPrediction(model_id="lightgbm_F_v1", available=False, unavailable_reason="test"),
        market=MarketContext(available=False, unavailable_reason="test"),
        model_market_disagreement_points=None, known_qb_continuity_note=None,
        known_personnel_continuity_note=None, known_injury_summary=None,
    )


def test_latest_research_summary_sorts_by_real_timestamp_not_by_run_id_string(api_data_dir):
    """Regression test for a real bug: a manually-named run_id (e.g. a pilot run) sorts
    lexicographically AFTER any ISO-timestamp run_id despite being chronologically older -
    `sorted(run_ids)[-1]` picked the stale pilot run over real, later runs."""
    run_research_for_game(
        packet=_research_packet(), provider=FixtureLLMProvider(LLM_FIXTURE),
        research_prompt_template_text="x", run_id="pilot_20260911", now="2026-09-11T10:05:00+00:00",
    )
    run_research_for_game(
        packet=_research_packet(), provider=FixtureLLMProvider(LLM_FIXTURE),
        research_prompt_template_text="x", run_id="2026-09-12T08-00-00.000000+00-00", now="2026-09-12T08:00:00+00:00",
    )

    summary = latest_research_summary(SEASON, WEEK, GAME_ID)

    assert summary["available"] is True
    assert summary["run_id"] == "2026-09-12T08-00-00.000000+00-00"  # the REAL later run, not "pilot_20260911"
    assert summary["research_timestamp"] == "2026-09-12T08:00:00+00:00"


def test_latest_model_prediction_prefers_the_most_recent_non_superseded_record(api_data_dir):
    older = PublicModelPrediction(
        prediction_id="2026-09-12T05:00:00+00:00", game_id=GAME_ID, season=SEASON, week=WEEK,
        generated_at="2026-09-12T05:00:00+00:00", elo_home_win_probability=0.4, elo_predicted_margin=-2.0,
        ridge_predicted_margin=None, lightgbm_predicted_margin=None, model_agreement_all_agree=None,
        model_agreement_dispersion=None, predicted_winner="away",
    )
    newer = PublicModelPrediction(
        prediction_id="2026-09-12T08:00:00+00:00", game_id=GAME_ID, season=SEASON, week=WEEK,
        generated_at="2026-09-12T08:00:00+00:00", elo_home_win_probability=0.39, elo_predicted_margin=-3.2,
        ridge_predicted_margin=-3.4, lightgbm_predicted_margin=-0.9, model_agreement_all_agree=True,
        model_agreement_dispersion=2.5, predicted_winner="away",
    )
    publish_model_prediction(older, PublicationState.PUBLISHED)
    publish_model_prediction(newer, PublicationState.PUBLISHED)

    result = latest_model_prediction(GAME_ID)
    assert result["prediction_id"] == "2026-09-12T08:00:00+00:00"
    assert result["elo_predicted_margin"] == -3.2


def test_latest_model_prediction_returns_none_when_nothing_published(api_data_dir):
    assert latest_model_prediction(GAME_ID) is None


def test_latest_decisions_for_game_returns_the_latest_per_market_type(api_data_dir):
    older = DecisionRecord(
        decision_id="a", game_id=GAME_ID, decision_timestamp="2026-09-12T05:00:00+00:00", kickoff_timestamp="2026-09-14T20:15:00",
        decision=Decision.NO_BET, reason_codes=(ReasonCode.MISSING_LIVE_DATA,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h1", validation_status="PROSPECTIVE",
    )
    newer = DecisionRecord(
        decision_id="b", game_id=GAME_ID, decision_timestamp="2026-09-12T08:00:00+00:00", kickoff_timestamp="2026-09-14T20:15:00",
        decision=Decision.QUALIFIED_BET, reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h2", validation_status="PROSPECTIVE",
    )
    append_decision_record(older, SEASON, WEEK)
    append_decision_record(newer, SEASON, WEEK)

    result = latest_decisions_for_game(SEASON, WEEK, GAME_ID)
    assert result["spread"]["decision"] == "QUALIFIED_BET"
    assert result["moneyline"] is None


def test_current_best_bets_is_empty_when_the_ledger_is_empty(api_data_dir):
    assert current_best_bets() == []


def test_current_best_bets_only_returns_published_best_bets_category(api_data_dir):
    published = PublishedPick(
        pick_id="p1", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00", kickoff_at="2026-09-14T20:15:00",
        decision_id="b", rule_version="v1", category="BEST_BETS", market_type="spread", selection="KC -2.5",
        line=-2.5, price=-110, sportsbook_or_source="consensus", market_snapshot_id=None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    )
    not_a_best_bet = PublishedPick(
        pick_id="p2", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00", kickoff_at="2026-09-14T20:15:00",
        decision_id="c", rule_version="v1", category="LEANS", market_type="moneyline", selection="KC ML",
        line=None, price=-142, sportsbook_or_source="consensus", market_snapshot_id=None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    )
    publish_pick(published)
    publish_pick(not_a_best_bet)

    picks = current_best_bets()
    assert len(picks) == 1
    assert picks[0]["pick_id"] == "p1"


def test_system_pick_for_game_is_none_when_nothing_published_yet(api_data_dir):
    assert system_pick_for_game(GAME_ID) is None


def test_system_pick_for_game_returns_only_the_all_model_predictions_category(api_data_dir):
    """A real Best Bet published for the same game must NOT come back as the system_pick -
    these are two deliberately separate records (see pick_publishing.py)."""
    publish_pick(PublishedPick(
        pick_id="best1", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00", kickoff_at="2026-09-14T20:15:00",
        decision_id="b", rule_version="v1", category="BEST_BETS", market_type="spread", selection="home",
        line=-2.5, price=-110, sportsbook_or_source="consensus", market_snapshot_id=None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    ))
    publish_pick(PublishedPick(
        pick_id=f"{GAME_ID}_all_model_predictions", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00",
        kickoff_at="2026-09-14T20:15:00", decision_id="c", rule_version="v1", category="ALL_MODEL_PREDICTIONS",
        market_type="moneyline", selection="away", line=None, price=120, sportsbook_or_source="consensus",
        market_snapshot_id=None, model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    ))

    pick = system_pick_for_game(GAME_ID)
    assert pick is not None
    assert pick["category"] == "ALL_MODEL_PREDICTIONS"
    assert pick["selection"] == "away"


def test_system_pick_for_game_excludes_a_voided_pick(api_data_dir):
    from nfl_predict.decision.pick_ledger import void_pick

    publish_pick(PublishedPick(
        pick_id=f"{GAME_ID}_all_model_predictions", game_id=GAME_ID, published_at="2026-09-12T09:00:00+00:00",
        kickoff_at="2026-09-14T20:15:00", decision_id="c", rule_version="v1", category="ALL_MODEL_PREDICTIONS",
        market_type="moneyline", selection="away", line=None, price=120, sportsbook_or_source="consensus",
        market_snapshot_id=None, model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE",
    ))
    void_pick(f"{GAME_ID}_all_model_predictions", void_reason="DUPLICATE_PUBLICATION", void_timestamp="2026-09-12T10:00:00+00:00", authorized_by="test")

    assert system_pick_for_game(GAME_ID) is None
