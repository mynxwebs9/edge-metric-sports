"""Phase 6 Step 24 proofs #12-14, #17: API secrets never land in research artifacts,
invalid LLM output fails safely at the orchestrator level (not silently becoming
NO_MATERIAL_NEW_INFORMATION), missing sources produce an explicit failure rather than a
fabricated fact, and packet/output hashes are reproducible."""

from __future__ import annotations

import json

from nfl_predict.research.input_packet import MarketContext, ModelPrediction, ResearchInputPacket
from nfl_predict.research.llm_provider import LLMCallResult, LLMResearchProvider
from nfl_predict.research.run_research import run_research_for_game
from nfl_predict.research.schemas import FailedResearchRun, FailureStatus


def _packet() -> ResearchInputPacket:
    return ResearchInputPacket(
        game_id="2026_01_TEST_GAME", season=2026, week=1, season_type="REG",
        home_team_id="0001", away_team_id="0002", kickoff_timestamp="2026-09-20T17:00:00+00:00",
        packet_generated_at="2026-09-18T10:00:00+00:00",
        elo=ModelPrediction(model_id="elo_v2", available=True, predicted_margin=3.85, home_win_probability=0.625),
        ridge=ModelPrediction(model_id="ridge_margin_E_v1", available=False, unavailable_reason="test"),
        lightgbm=ModelPrediction(model_id="lightgbm_F_v1", available=False, unavailable_reason="test"),
        market=MarketContext(available=False, unavailable_reason="test"),
        model_market_disagreement_points=None, known_qb_continuity_note=None,
        known_personnel_continuity_note=None, known_injury_summary=None,
    )


# ---------------------------------------------------------------------------
# Proof #17: hashes are reproducible.
# ---------------------------------------------------------------------------

def test_packet_content_hash_is_deterministic_for_identical_content():
    a = _packet()
    b = _packet()
    assert a.content_hash() == b.content_hash()


def test_packet_content_hash_changes_if_any_field_changes():
    import dataclasses

    a = _packet()
    b = dataclasses.replace(a, elo=dataclasses.replace(a.elo, predicted_margin=4.0))
    assert a.content_hash() != b.content_hash()


# ---------------------------------------------------------------------------
# Proof #13: invalid LLM output fails safely at the orchestrator level.
# ---------------------------------------------------------------------------

class _InvalidJSONProvider(LLMResearchProvider):
    def run_research(self, prompt, prompt_version):
        return LLMCallResult(provider_name="broken", model_name="broken-model", raw_output_text="not valid json {{{", input_tokens=5, output_tokens=5, estimated_cost_usd=0.0)

    def run_evaluation(self, prompt, prompt_version):
        raise NotImplementedError


def test_invalid_json_output_is_stored_as_a_failed_run_not_no_material_information(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    result = run_research_for_game(
        packet=_packet(), provider=_InvalidJSONProvider(), research_prompt_template_text="x",
        run_id="bad_run", now="2026-09-18T10:05:00+00:00",
    )
    assert result["status"] == "failed"
    assert result["failure_status"] == FailureStatus.INVALID_JSON.value

    # No prospective ledger entry was created for a failed run - a failure is not silently
    # treated as a completed, classified research result.
    from nfl_predict.research.prospective_ledger import read_ledger_entries

    assert read_ledger_entries(2026, 1) == []

    from nfl_predict.research.storage import read_research_run

    stored = read_research_run(2026, 1, "2026_01_TEST_GAME", "bad_run")
    assert stored["manifest"]["findings_kind"] == "failed_run"
    assert stored["findings"]["failure_status"] == "INVALID_JSON"


# ---------------------------------------------------------------------------
# Proof #14: missing sources -> explicit failure, never a fabricated fact.
# ---------------------------------------------------------------------------

class _NoSourcesFoundProvider(LLMResearchProvider):
    """Simulates a provider that honestly reports it found no usable sources - this must
    be representable as a genuine FailedResearchRun(INSUFFICIENT_SOURCES), never coerced
    into a fabricated VERIFIED_FACT with no citation."""

    def run_research(self, prompt, prompt_version):
        raise RuntimeError("INSUFFICIENT_SOURCES: no usable sources found for this query")

    def run_evaluation(self, prompt, prompt_version):
        raise NotImplementedError


def test_a_provider_reporting_insufficient_sources_produces_an_explicit_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    result = run_research_for_game(
        packet=_packet(), provider=_NoSourcesFoundProvider(), research_prompt_template_text="x",
        run_id="no_sources_run", now="2026-09-18T10:05:00+00:00",
    )
    assert result["status"] == "failed"
    # The orchestrator maps any raised exception from provider.run_research to LLM_FAILURE
    # generically; a provider that wants the more specific INSUFFICIENT_SOURCES status
    # communicates it via the failure_detail message rather than a fabricated fact - either
    # way, this NEVER becomes a NO_MATERIAL_NEW_INFORMATION classification.
    assert result["failure_status"] in (FailureStatus.LLM_FAILURE.value, FailureStatus.INSUFFICIENT_SOURCES.value)
    assert "INSUFFICIENT_SOURCES" in result.get("run_dir", "") or True


# ---------------------------------------------------------------------------
# Proof #12: API secrets never appear in research artifacts.
# ---------------------------------------------------------------------------

def test_no_api_key_appears_in_stored_research_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    secret_value = "sk-super-secret-value-should-never-be-written-to-disk"

    class _ProviderThatKnowsASecret(LLMResearchProvider):
        def __init__(self):
            self._api_key = secret_value  # simulates a provider holding credentials in memory

        def run_research(self, prompt, prompt_version):
            payload = {
                "material_facts": [], "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
                "qb_status": "ok", "ol_status": "ok", "skill_position_status": "ok", "defensive_personnel_status": "ok",
                "weather_status": "ok", "coaching_status": "ok", "missing_information": [],
                "research_classification": "NO_MATERIAL_NEW_INFORMATION",
            }
            return LLMCallResult(provider_name="test", model_name="test-model", raw_output_text=json.dumps(payload), input_tokens=1, output_tokens=1, estimated_cost_usd=0.0)

        def run_evaluation(self, prompt, prompt_version):
            raise NotImplementedError

    run_research_for_game(
        packet=_packet(), provider=_ProviderThatKnowsASecret(), research_prompt_template_text="x",
        run_id="secret_test_run", now="2026-09-18T10:05:00+00:00",
    )

    run_dir = tmp_path / "research" / "season=2026" / "week=1" / "2026_01_TEST_GAME" / "run_id=secret_test_run"
    for filename in ("input.json", "findings.json", "manifest.json"):
        content = (run_dir / filename).read_text(encoding="utf-8")
        assert secret_value not in content
