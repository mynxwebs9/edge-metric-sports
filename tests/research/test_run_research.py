"""Phase 6 Step 24 proofs #3-4, #16: quantitative prediction values are immutable
throughout research, LLM output cannot alter model probabilities, and prompt versions are
recorded on every run."""

from __future__ import annotations

import dataclasses
import json

import pytest

from nfl_predict.research.input_packet import ModelPrediction, ResearchInputPacket, MarketContext
from nfl_predict.research.llm_provider import FixtureLLMProvider, LLMCallResult, LLMResearchProvider
from nfl_predict.research.run_research import build_prompt_from_template, run_research_for_game


def _packet(elo_margin=3.85, elo_prob=0.625) -> ResearchInputPacket:
    return ResearchInputPacket(
        game_id="2026_01_TEST_GAME", season=2026, week=1, season_type="REG",
        home_team_id="0001", away_team_id="0002", kickoff_timestamp="2026-09-20T17:00:00+00:00",
        packet_generated_at="2026-09-18T10:00:00+00:00",
        elo=ModelPrediction(model_id="elo_v2", available=True, predicted_margin=elo_margin, home_win_probability=elo_prob),
        ridge=ModelPrediction(model_id="ridge_margin_E_v1", available=False, unavailable_reason="test"),
        lightgbm=ModelPrediction(model_id="lightgbm_F_v1", available=False, unavailable_reason="test"),
        market=MarketContext(available=False, unavailable_reason="test"),
        model_market_disagreement_points=None, known_qb_continuity_note=None,
        known_personnel_continuity_note=None, known_injury_summary=None,
    )


def test_research_input_packet_is_frozen_and_cannot_be_mutated():
    packet = _packet()
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.elo = ModelPrediction(model_id="hacked", available=True, predicted_margin=999.0)


def test_model_prediction_is_frozen_and_cannot_be_mutated():
    packet = _packet()
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.elo.predicted_margin = 999.0


class _InjectingProvider(LLMResearchProvider):
    """A hostile fixture provider whose output tries to smuggle in a numeric override -
    the schema has no field for one, so `parse_research_output` simply ignores it."""

    def run_research(self, prompt, prompt_version):
        payload = {
            "material_facts": [], "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
            "qb_status": "ok", "ol_status": "ok", "skill_position_status": "ok", "defensive_personnel_status": "ok",
            "weather_status": "ok", "coaching_status": "ok", "missing_information": [],
            "research_classification": "NO_MATERIAL_NEW_INFORMATION",
            # Hostile/malformed extra fields an attacker or a misbehaving model might add:
            "predicted_margin": 999.0, "home_win_probability": 0.01, "fair_spread": -50.0,
        }
        return LLMCallResult(provider_name="hostile", model_name="hostile-model", raw_output_text=json.dumps(payload), input_tokens=10, output_tokens=10, estimated_cost_usd=0.001)

    def run_evaluation(self, prompt, prompt_version):
        raise NotImplementedError


def test_llm_output_numeric_fields_are_ignored_and_never_reach_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    packet = _packet(elo_margin=3.85, elo_prob=0.625)
    result = run_research_for_game(
        packet=packet, provider=_InjectingProvider(), research_prompt_template_text="Research {{game_id}}.",
        run_id="test_run", now="2026-09-18T10:05:00+00:00",
    )
    assert result["status"] == "ok"

    from nfl_predict.research.prospective_ledger import read_ledger_entries

    entries = read_ledger_entries(2026, 1)
    assert len(entries) == 1
    # The ledger's quantitative fields are exactly the ORIGINAL packet's values - the
    # hostile provider's "predicted_margin": 999.0 / "home_win_probability": 0.01 never
    # appear anywhere, because ResearchFindings has no field that could carry them.
    assert entries[0]["elo_predicted_margin"] == 3.85
    assert entries[0]["elo_home_win_probability"] == 0.625


def test_prompt_version_is_recorded_in_the_stored_findings(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    packet = _packet()
    run_research_for_game(packet=packet, provider=_InjectingProvider(), research_prompt_template_text="x", run_id="test_run2", now="2026-09-18T10:05:00+00:00")

    from nfl_predict.research.storage import read_research_run

    stored = read_research_run(2026, 1, packet.game_id, "test_run2")
    assert stored["findings"]["prompt_version"] == "matchup_research_v2"  # Phase 8A cost-controls correction bumped the prompt version


def test_build_prompt_from_template_rejects_unknown_placeholders():
    with pytest.raises(ValueError, match="unknown placeholder"):
        build_prompt_from_template("Hello {{not_a_real_field}}", _packet())


def test_build_prompt_from_template_fills_known_fields():
    text = build_prompt_from_template("Game: {{game_id}}, Elo margin: {{elo_predicted_margin}}", _packet(elo_margin=4.2))
    assert "2026_01_TEST_GAME" in text
    assert "4.2" in text


# --- Regression tests for the real DEN@KC incident's accounting bug: the run made 3 real,
# billed web searches, but the automation report showed research_games_selected=0,
# research_calls_made=0, research_calls_ok=0 - because a downstream TypeError propagated
# uncaught past run_research_for_game()'s own status-returning contract, so the caller never
# got a result dict back at all, let alone the real cost/usage already recorded on `tracker`.
# These tests prove the money is never spent for nothing on the accounting side: a call that
# reaches Anthropic and gets billed always shows up as attempted, with its real usage, no
# matter what fails afterward.


class _RealUsageThenMalformedOutputProvider(LLMResearchProvider):
    """Mimics the actual DEN@KC incident shape: a real call with real usage (3 web
    searches, realistic token counts) whose submit_research_findings input put a bare URL
    string in a claim's `sources` array instead of a full source object."""

    def run_research(self, prompt, prompt_version):
        payload = {
            "material_facts": [{
                "text": "QB questionable", "category": "REPORTED_NOT_CONFIRMED", "materiality_level": 2,
                "confidence_in_fact": 0.6, "reason": "single report",
                "sources": ["https://example.com/beat-reporter-tweet"],  # the real incident shape - bare string, not a source object
            }],
            "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
            "qb_status": "Uncertain.", "ol_status": "No change.", "skill_position_status": "No change.",
            "defensive_personnel_status": "No change.", "weather_status": "Not checked.", "coaching_status": "No change.",
            "missing_information": [], "research_classification": "HIGH_UNCERTAINTY",
        }
        return LLMCallResult(
            provider_name="anthropic", model_name="claude-sonnet-5", raw_output_text=json.dumps(payload),
            input_tokens=69942, output_tokens=7586, estimated_cost_usd=0.245744,
            cache_creation_input_tokens=1200, cache_read_input_tokens=0, web_search_requests=3,
        )

    def run_evaluation(self, prompt, prompt_version):
        raise NotImplementedError


def test_a_real_call_that_fails_to_parse_downstream_still_reports_as_attempted_with_real_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    result = run_research_for_game(
        packet=_packet(), provider=_RealUsageThenMalformedOutputProvider(), research_prompt_template_text="x",
        run_id="test_run_incident", now="2026-09-18T10:05:00+00:00",
    )

    assert result["status"] == "failed"
    assert result["failure_status"] == "INVALID_JSON"
    # The real usage/cost already incurred by the real Anthropic call must survive the
    # downstream parse failure - never silently dropped/reported as zero.
    assert result["cost"]["total_llm_calls"] == 1
    assert result["cost"]["total_web_search_requests"] == 3
    assert result["cost"]["total_input_tokens"] == 69942
    assert result["cost"]["total_output_tokens"] == 7586
    assert result["cost"]["total_estimated_cost_usd"] == pytest.approx(0.245744)


class _ValidOutputProvider(LLMResearchProvider):
    """A real, well-formed call - used to isolate a failure to evaluation/persistence
    (Layer 2's catch-all), not parsing (Layer 1's guards)."""

    def run_research(self, prompt, prompt_version):
        payload = {
            "material_facts": [], "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
            "qb_status": "ok", "ol_status": "ok", "skill_position_status": "ok", "defensive_personnel_status": "ok",
            "weather_status": "ok", "coaching_status": "ok", "missing_information": [],
            "research_classification": "NO_MATERIAL_NEW_INFORMATION",
        }
        return LLMCallResult(
            provider_name="anthropic", model_name="claude-sonnet-5", raw_output_text=json.dumps(payload),
            input_tokens=50000, output_tokens=3000, estimated_cost_usd=0.13,
            cache_creation_input_tokens=0, cache_read_input_tokens=1000, web_search_requests=2,
        )

    def run_evaluation(self, prompt, prompt_version):
        raise NotImplementedError


def test_a_downstream_evaluation_failure_after_a_successful_parse_still_preserves_real_usage(tmp_path, monkeypatch):
    """Layer 2's catch-all: even a failure the parser's type guards don't cover (e.g. a bug
    in evaluate_research itself) must never lose the already-billed usage, and must never
    propagate uncaught past run_research_for_game() the way the original TypeError did."""
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr(
        "nfl_predict.research.run_research.evaluate_research",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated unexpected evaluator bug")),
    )

    result = run_research_for_game(
        packet=_packet(), provider=_ValidOutputProvider(), research_prompt_template_text="x",
        run_id="test_run_downstream", now="2026-09-18T10:05:00+00:00",
    )

    assert result["status"] == "failed"
    assert result["failure_status"] == "DOWNSTREAM_PROCESSING_FAILURE"
    assert result["cost"]["total_llm_calls"] == 1
    assert result["cost"]["total_web_search_requests"] == 2
    assert result["cost"]["total_estimated_cost_usd"] == pytest.approx(0.13)

    from nfl_predict.research.storage import read_research_run

    stored = read_research_run(2026, 1, "2026_01_TEST_GAME", "test_run_downstream")
    assert stored["manifest"]["findings_kind"] == "failed_run"
    assert stored["findings"]["failure_status"] == "DOWNSTREAM_PROCESSING_FAILURE"


def test_a_game_whose_kickoff_has_already_passed_is_refused_not_silently_researched(tmp_path, monkeypatch):
    """`run_research_for_game` always treats its own call time as both "now" and the
    research record's claimed timestamp (it is a PROSPECTIVE-only orchestrator - historical
    reconstruction, if ever built, is a deliberately separate code path exercised directly
    against `historical_guard.assert_research_may_proceed`, see test_historical_guard.py).
    So a past-kickoff game is refused via `InvalidPregameTimestampError` here (the
    research-timestamp-after-kickoff check fires first); a would-be historical-reconstruction
    caller supplying a genuinely earlier claimed timestamp instead hits
    `HistoricalResearchLeakageRisk`, already covered directly in test_historical_guard.py."""
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    from nfl_predict.research.historical_guard import InvalidPregameTimestampError

    past_packet = _packet()
    past_packet = dataclasses.replace(past_packet, kickoff_timestamp="2020-01-01T17:00:00+00:00")
    with pytest.raises(InvalidPregameTimestampError):
        run_research_for_game(packet=past_packet, provider=_InjectingProvider(), research_prompt_template_text="x", run_id="test_run3", now="2026-09-18T10:05:00+00:00")
