"""Phase 8A correction: orchestrator-level integration tests proving `run_slate()` builds
real per-game MarketPoint/ResearchPoint from actually-fetched odds and actually-invoked
research - never merely from provider/key availability - and that a fully-populated
synthetic packet reaches decision gates 3-9 instead of stopping at gate 2
(MISSING_LIVE_DATA), all while decision rules v1 (`config/decision_rules.yaml`,
`nfl_predict.decision.engine`) remain completely untouched. Tests #1, #3, #5, #9, #10,
#13, #14 from the correction brief."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nfl_predict.backtesting.walk_forward import PredictionRecord
from nfl_predict.live import run as run_module
from nfl_predict.live.injury_automation import InjuryProviderStatus
from nfl_predict.live.odds_automation import OddsProviderStatus
from nfl_predict.live.research_automation import ResearchProviderStatus
from nfl_predict.market.odds_provider import OddsEvent, OddsMarketSnapshot, OddsProvider
from nfl_predict.research.llm_provider import FixtureLLMProvider, LLMCallResult, LLMResearchProvider

REAL_GAME_ID = "2026_01_DEN_KC"
LLM_FIXTURE = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "llm_fixture.json"


class _FakeOddsProvider(OddsProvider):
    def __init__(self, events, markets_by_event):
        self._events = events
        self._markets = markets_by_event

    def get_events(self):
        return self._events

    def get_markets_for_sport(self):
        return [s for snaps in self._markets.values() for s in snaps]


class _Result:
    def __init__(self, status, provider):
        self.status = status
        self.provider = provider


def _now_fresh() -> str:
    return datetime.now(timezone.utc).isoformat()


def _real_odds_event_and_snapshot(home_spread=-3.0, home_ml=-140, away_ml=120, fetched_at=None):
    fetched_at = fetched_at or _now_fresh()
    event = OddsEvent(provider_event_id="evt_denkc", game_id=None, home_team="Kansas City Chiefs", away_team="Denver Broncos", commence_time="2026-09-14T20:15:00Z")
    snapshot = OddsMarketSnapshot(
        provider_event_id="evt_denkc", fetched_at=fetched_at, bookmaker="book_a",
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread,
        home_spread_price=-110, away_spread_price=-110, home_moneyline=home_ml, away_moneyline=away_ml,
        total_line=44.5, over_price=-110, under_price=-110,
    )
    return event, snapshot


def _patch_common_storage(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_odds_provider_available_with_no_matching_event_leaves_market_unavailable(tmp_path, monkeypatch):
    """Test #1: odds_status AVAILABLE alone (the provider/key can be constructed) must not
    make MarketPoint.available True for a game the provider returned nothing usable for."""
    _patch_common_storage(monkeypatch, tmp_path)
    empty_provider = _FakeOddsProvider(events=[], markets_by_event={})
    monkeypatch.setattr(run_module, "get_production_odds_provider", lambda: _Result(OddsProviderStatus.AVAILABLE, empty_provider))

    result = run_module.run_slate(game_id=REAL_GAME_ID, dry_run=True, show_decisions=True)

    assert result["odds_status"] == "AVAILABLE"
    assert result["odds_games_matched"] == 0
    for row in result["decisions"]:
        assert row["market_home_spread"] is None  # never fabricated just because the provider is up


def test_research_provider_available_but_not_selected_leaves_researchpoint_unavailable(tmp_path, monkeypatch):
    """Test #5: research_status AVAILABLE alone must not populate ResearchPoint unless a
    real research run was actually selected and invoked - forced here via max_research_calls=0."""
    _patch_common_storage(monkeypatch, tmp_path)
    provider = FixtureLLMProvider(LLM_FIXTURE)
    monkeypatch.setattr(run_module, "get_production_research_provider", lambda: _Result(ResearchProviderStatus.AVAILABLE, provider))

    result = run_module.run_slate(game_id=REAL_GAME_ID, dry_run=True, max_research_calls=0, show_decisions=True)

    assert result["research_status"] == "AVAILABLE"
    assert result["research_calls_made"] == 0
    for row in result["decisions"]:
        assert row["research_classification"] is None


def test_injury_provider_unavailable_never_causes_missing_live_data_on_its_own(tmp_path, monkeypatch):
    """Test #10: the structured injury boundary is not a decision-rule input at all under
    v1 - its unavailability must never be the reason a decision fails to reach the
    market/research gate."""
    result = run_module.run_slate(game_id=REAL_GAME_ID, dry_run=True)
    assert result["injury_status"] == "INJURY_PROVIDER_UNAVAILABLE"  # no key configured in this test environment
    # Every decision's reasons in this default (no odds/research configured) run are exactly
    # MISSING_LIVE_DATA because of the ACTUAL missing market/research - never because of injury,
    # which is never referenced by decide() or config/decision_rules.yaml at all (see the
    # correction brief's item 5 / docs/PHASE8A_LIVE_PIPELINE_REPORT.md).
    for v in result["decision_counts"]:
        pass  # decision_counts has no per-reason breakdown; reason_code_counts does:
    assert set(result["reason_code_counts"].keys()) <= {"MISSING_LIVE_DATA"}


def test_full_pipeline_with_synthetic_valid_odds_and_research_reaches_gates_3_to_9(tmp_path, monkeypatch):
    """Test #14: with real market data AND a real (fixture-backed) research invocation, and
    strong enough model-market disagreement, at least one decision's reasons must be more
    than just MISSING_LIVE_DATA - proving gate 2 is passable and later gates are reached.
    Decision rules v1 (config/decision_rules.yaml, engine.py) are never touched to make this
    happen - only the packet's inputs are populated with real-shaped data."""
    _patch_common_storage(monkeypatch, tmp_path)

    event, snapshot = _real_odds_event_and_snapshot(home_spread=-3.0, home_ml=-140, away_ml=120)
    odds_provider = _FakeOddsProvider(events=[event], markets_by_event={"evt_denkc": [snapshot]})
    monkeypatch.setattr(run_module, "get_production_odds_provider", lambda: _Result(OddsProviderStatus.AVAILABLE, odds_provider))

    research_provider = FixtureLLMProvider(LLM_FIXTURE)  # NO_MATERIAL_NEW_INFORMATION - never caps below QUALIFIED_BET
    monkeypatch.setattr(run_module, "get_production_research_provider", lambda: _Result(ResearchProviderStatus.AVAILABLE, research_provider))

    # Strong, agreeing, low-dispersion model signal - same shape as
    # tests/decision/test_engine.py's own QUALIFIED_BET fixture, just produced via the live
    # model functions' return shape instead of decision-layer ModelPoints directly.
    monkeypatch.setattr(run_module, "predict_live_elo", lambda game_ids, seasons: [
        PredictionRecord(game_id=REAL_GAME_ID, season=2026, week=1, season_type="REG", kickoff_timestamp="2026-09-14T20:15:00", model_id="elo_v2", model_version="v1", feature_version=None, target="home_margin", predicted_value=10.0, training_cutoff_season=2026, training_cutoff_week=0, training_row_count=100, artifact_hash=None, feature_names_hash=None),
        PredictionRecord(game_id=REAL_GAME_ID, season=2026, week=1, season_type="REG", kickoff_timestamp="2026-09-14T20:15:00", model_id="elo_v2", model_version="v1", feature_version=None, target="home_win", predicted_value=0.7, training_cutoff_season=2026, training_cutoff_week=0, training_row_count=100, artifact_hash=None, feature_names_hash=None),
    ])

    def _fake_ridge_lgbm(game_ids, seasons):
        def rec(model_id, target, value):
            return PredictionRecord(game_id=REAL_GAME_ID, season=2026, week=1, season_type="REG", kickoff_timestamp="2026-09-14T20:15:00", model_id=model_id, model_version="v1", feature_version=None, target=target, predicted_value=value, training_cutoff_season=2026, training_cutoff_week=0, training_row_count=100, artifact_hash=None, feature_names_hash=None)
        return {
            "ridge_margin_E_v1": [rec("ridge_margin_E_v1", "home_margin", 8.0)],
            "lightgbm_F_v1": [rec("lightgbm_F_v1", "home_margin", 9.0), rec("lightgbm_F_v1", "home_win", 0.68)],
            "logistic_win_E_v1": [rec("logistic_win_E_v1", "home_win", 0.66)],
        }
    monkeypatch.setattr(run_module, "predict_live_ridge_lightgbm_logistic", _fake_ridge_lgbm)

    result = run_module.run_slate(game_id=REAL_GAME_ID, dry_run=True, research_all=True, show_decisions=True)

    assert result["odds_games_matched"] == 1
    assert result["research_calls_made"] == 1
    assert result["any_decision_reached_gates_3_to_9"] is True

    spread_row = next(r for r in result["decisions"] if r["market_type"] == "spread")
    assert spread_row["reasons"] != ["MISSING_LIVE_DATA"]
    assert spread_row["decision"] == "QUALIFIED_BET"
    assert spread_row["market_home_spread"] == -3.0
    assert spread_row["research_classification"] == "NO_MATERIAL_NEW_INFORMATION"

    moneyline_row = next(r for r in result["decisions"] if r["market_type"] == "moneyline")
    assert moneyline_row["reasons"] != ["MISSING_LIVE_DATA"]
    assert moneyline_row["decision"] == "QUALIFIED_BET"  # test #15 (moneyline): a synthetic packet meeting every frozen v1 gate reaches QUALIFIED_BET here too
    assert moneyline_row["market_no_vig_home_prob"] is not None


def test_a_persisted_decision_record_references_the_real_market_and_research_artifacts(tmp_path, monkeypatch):
    """Phase 8A provenance correction: a real (non-dry-run) decision record must carry a
    REAL content hash (never the old hard-coded "n/a") and must be able to identify exactly
    which persisted market snapshot and research run it was computed from."""
    _patch_common_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    event, snapshot = _real_odds_event_and_snapshot(home_spread=-3.0, home_ml=-140, away_ml=120)
    odds_provider = _FakeOddsProvider(events=[event], markets_by_event={"evt_denkc": [snapshot]})
    monkeypatch.setattr(run_module, "get_production_odds_provider", lambda: _Result(OddsProviderStatus.AVAILABLE, odds_provider))

    research_provider = FixtureLLMProvider(LLM_FIXTURE)
    monkeypatch.setattr(run_module, "get_production_research_provider", lambda: _Result(ResearchProviderStatus.AVAILABLE, research_provider))

    monkeypatch.setattr(run_module, "predict_live_elo", lambda game_ids, seasons: [
        PredictionRecord(game_id=REAL_GAME_ID, season=2026, week=1, season_type="REG", kickoff_timestamp="2026-09-14T20:15:00", model_id="elo_v2", model_version="v1", feature_version=None, target="home_margin", predicted_value=10.0, training_cutoff_season=2026, training_cutoff_week=0, training_row_count=100, artifact_hash=None, feature_names_hash=None),
        PredictionRecord(game_id=REAL_GAME_ID, season=2026, week=1, season_type="REG", kickoff_timestamp="2026-09-14T20:15:00", model_id="elo_v2", model_version="v1", feature_version=None, target="home_win", predicted_value=0.7, training_cutoff_season=2026, training_cutoff_week=0, training_row_count=100, artifact_hash=None, feature_names_hash=None),
    ])

    def _fake_ridge_lgbm(game_ids, seasons):
        def rec(model_id, target, value):
            return PredictionRecord(game_id=REAL_GAME_ID, season=2026, week=1, season_type="REG", kickoff_timestamp="2026-09-14T20:15:00", model_id=model_id, model_version="v1", feature_version=None, target=target, predicted_value=value, training_cutoff_season=2026, training_cutoff_week=0, training_row_count=100, artifact_hash=None, feature_names_hash=None)
        return {
            "ridge_margin_E_v1": [rec("ridge_margin_E_v1", "home_margin", 8.0)],
            "lightgbm_F_v1": [rec("lightgbm_F_v1", "home_margin", 9.0), rec("lightgbm_F_v1", "home_win", 0.68)],
            "logistic_win_E_v1": [rec("logistic_win_E_v1", "home_win", 0.66)],
        }
    monkeypatch.setattr(run_module, "predict_live_ridge_lightgbm_logistic", _fake_ridge_lgbm)

    run_module.run_slate(game_id=REAL_GAME_ID, dry_run=False, research_all=True)

    from nfl_predict.decision.decision_log import read_decision_records

    records = read_decision_records(2026, 1)
    spread = next(r for r in records if r["game_id"] == REAL_GAME_ID and r["market_type"] == "spread")

    assert spread["input_packet_hash"] != "n/a"
    assert len(spread["input_packet_hash"]) == 64  # a real sha256 hex digest
    assert spread["market_provider_event_id"] == "evt_denkc"
    assert spread["market_snapshot_reference"] is not None
    assert spread["market_snapshot_timestamp"] is not None
    assert spread["research_id"] is not None

    from nfl_predict.market.event_game_mapping import find_provider_event_ids_for_game

    assert spread["market_provider_event_id"] in find_provider_event_ids_for_game(2026, 1, REAL_GAME_ID)


class _RealCallMalformedOutputProvider(LLMResearchProvider):
    """Orchestrator-level regression for the real DEN@KC incident: a real, billed call (real
    usage numbers) whose output has the malformed nested shape (a bare URL string instead of
    a source object) that used to crash uncaught past run_component_safely() and get
    reported as research_games_selected=0/research_calls_made=0 despite money being spent."""

    def run_research(self, prompt, prompt_version):
        import json as _json
        payload = {
            "material_facts": [{
                "text": "QB questionable", "category": "REPORTED_NOT_CONFIRMED", "materiality_level": 2,
                "confidence_in_fact": 0.6, "reason": "single report",
                "sources": ["https://example.com/beat-reporter-tweet"],
            }],
            "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
            "qb_status": "Uncertain.", "ol_status": "No change.", "skill_position_status": "No change.",
            "defensive_personnel_status": "No change.", "weather_status": "Not checked.", "coaching_status": "No change.",
            "missing_information": [], "research_classification": "HIGH_UNCERTAINTY",
        }
        return LLMCallResult(
            provider_name="anthropic", model_name="claude-sonnet-5", raw_output_text=_json.dumps(payload),
            input_tokens=69942, output_tokens=7586, estimated_cost_usd=0.245744,
            cache_creation_input_tokens=1200, cache_read_input_tokens=0, web_search_requests=3,
        )

    def run_evaluation(self, prompt, prompt_version):
        raise NotImplementedError


def test_a_real_call_with_malformed_downstream_output_is_reported_as_selected_attempted_and_failed_not_zero(tmp_path, monkeypatch):
    """Orchestrator-level version of the DEN@KC incident: proves run_slate()'s summary
    reports the real selection/attempt/cost, never silently collapsing to zero just because
    the downstream parse failed."""
    _patch_common_storage(monkeypatch, tmp_path)

    event, snapshot = _real_odds_event_and_snapshot(home_spread=-3.0, home_ml=-140, away_ml=120)
    odds_provider = _FakeOddsProvider(events=[event], markets_by_event={"evt_denkc": [snapshot]})
    monkeypatch.setattr(run_module, "get_production_odds_provider", lambda: _Result(OddsProviderStatus.AVAILABLE, odds_provider))

    research_provider = _RealCallMalformedOutputProvider()
    monkeypatch.setattr(run_module, "get_production_research_provider", lambda: _Result(ResearchProviderStatus.AVAILABLE, research_provider))

    result = run_module.run_slate(game_id=REAL_GAME_ID, dry_run=True, research_all=True)

    assert result["research_games_selected"] == 1
    assert result["research_calls_made"] == 1
    assert result["research_calls_ok"] == 0
    assert result["research_calls_failed"] == 1
    assert result["research_llm_calls_attempted"] == 1  # the real call reached Anthropic and returned - never silently 0
    assert result["research_llm_calls_succeeded"] == 0
    assert result["research_llm_calls_failed"] == 1
    assert result["research_artifacts_persisted"] == 1  # the FailedResearchRun was written, not lost
    assert result["research"]["web_searches"] == 3
    assert result["research"]["input_tokens"] == 69942
    assert result["research"]["estimated_cost_usd"] == pytest.approx(0.245744)
    for row in result.get("decisions", []) or []:
        assert row.get("research_classification") is None  # a failed research run never populates ResearchPoint


def test_staleness_uses_the_real_snapshot_timestamp_not_a_manufactured_now(tmp_path, monkeypatch):
    """Test #13: an odds snapshot fetched more than 24h ago must produce WATCH/STALE_MARKET,
    even though the run's own `now` is fresh - proving age is computed from the snapshot's
    real `fetched_at`, never re-stamped to "now" to dodge the staleness gate."""
    _patch_common_storage(monkeypatch, tmp_path)
    stale_timestamp = "2020-01-01T00:00:00+00:00"  # far more than 24h before "now"
    event, snapshot = _real_odds_event_and_snapshot(fetched_at=stale_timestamp)
    odds_provider = _FakeOddsProvider(events=[event], markets_by_event={"evt_denkc": [snapshot]})
    monkeypatch.setattr(run_module, "get_production_odds_provider", lambda: _Result(OddsProviderStatus.AVAILABLE, odds_provider))

    research_provider = FixtureLLMProvider(LLM_FIXTURE)
    monkeypatch.setattr(run_module, "get_production_research_provider", lambda: _Result(ResearchProviderStatus.AVAILABLE, research_provider))

    result = run_module.run_slate(game_id=REAL_GAME_ID, dry_run=True, research_all=True, show_decisions=True)

    assert result["odds_games_matched"] == 1  # the market snapshot itself IS real and matched
    for row in result["decisions"]:
        assert "STALE_MARKET" in row["reasons"]
        assert row["decision"] == "WATCH"
