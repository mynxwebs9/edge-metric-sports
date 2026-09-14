"""Phase 8B follow-up: the game-preview content step. No live network call anywhere - the
Anthropic provider's own pre-flight cost guard and a couple of forced-failure providers
exercise every failure path without ever reaching `_post`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predict.content.prediction_writer import (
    PREDICTION_WRITER_PROMPT_VERSION,
    AnthropicContentWriterProvider,
    ContentWriteResult,
    ContentWriterNotConfiguredError,
    ContentWriterProvider,
    ContentWriterResponseError,
    FixtureContentWriterProvider,
    build_prompt_from_template,
    gather_game_content_context,
    write_prediction_preview,
)
from nfl_predict.content.storage import read_preview_run
from nfl_predict.decision.decision_log import append_decision_record
from nfl_predict.decision.schemas import Decision, DecisionRecord, ReasonCode
from nfl_predict.research.cost_tracking import ResearchCostBudgetExceededError

GAME_ID = "2026_01_DEN_KC"
SEASON, WEEK = 2026, 1
HOME_ID, AWAY_ID = "2310", "1400"
KICKOFF = "2026-09-14T20:15:00"
CONTENT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "content_fixture.json"
PROMPT_TEMPLATE = (Path(__file__).resolve().parents[2] / "prompts" / "prediction_writer_v1.md").read_text(encoding="utf-8")


def test_gather_game_content_context_with_no_data_yet_is_all_none(content_data_dir):
    context = gather_game_content_context(SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF)
    assert context["home_team_abbr"] == "KC"
    assert context["away_team_abbr"] == "DEN"
    assert context["model"] is None
    assert context["market"] is None
    assert context["research"] is None
    assert context["decisions"] == {"spread": None, "moneyline": None}


def test_build_prompt_from_template_fills_real_fields_and_rejects_unknown_placeholders():
    context = {
        "game_id": GAME_ID, "season": SEASON, "week": WEEK, "home_team_id": HOME_ID, "away_team_id": AWAY_ID,
        "home_team_name": "Kansas City Chiefs", "away_team_name": "Denver Broncos",
        "home_team_abbr": "KC", "away_team_abbr": "DEN", "kickoff_timestamp": KICKOFF,
        "model": {"elo_predicted_margin": -3.2, "elo_home_win_probability": 0.39, "ridge_predicted_margin": -3.4, "lightgbm_predicted_margin": -0.9},
        "market": {"available": True, "home_spread_traditional": -2.5, "no_vig_home_win_probability": 0.56, "consensus_book_keys": ["draftkings", "fanduel"]},
        "decisions": {"spread": None, "moneyline": None},
        "research": None,
    }
    prompt = build_prompt_from_template(PROMPT_TEMPLATE, context)
    assert "Denver Broncos" in prompt
    assert "Kansas City Chiefs" in prompt
    assert "DEN by 3.2" in prompt  # elo_margin_text
    assert "KC -2.5" in prompt  # market_summary_text

    with pytest.raises(ValueError, match="unknown placeholder"):
        build_prompt_from_template("Hello {{not_a_real_field}}", context)


def test_write_prediction_preview_with_the_fixture_provider_persists_a_real_article(content_data_dir):
    result = write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=FixtureContentWriterProvider(CONTENT_FIXTURE), prompt_template_text=PROMPT_TEMPLATE,
        run_id="test_run_1", now="2026-09-12T12:00:00+00:00",
    )
    assert result["status"] == "ok"
    assert "Elo projects DEN by 3.2" in result["text"]
    assert result["cost"]["estimated_cost_usd"] == pytest.approx(0.0026)

    stored = read_preview_run(SEASON, WEEK, GAME_ID, "test_run_1")
    assert stored["manifest"]["kind"] == "preview"
    assert stored["preview"]["prompt_version"] == PREDICTION_WRITER_PROMPT_VERSION
    assert stored["preview"]["model_provider"] == "fixture"
    assert stored["preview"]["context_hash"]


def test_write_prediction_preview_reflects_real_persisted_decision_and_research_data(content_data_dir):
    """The article's underlying context must reflect real, already-persisted state - not
    just whatever the fixture provider was told to say."""
    record = DecisionRecord(
        decision_id="d1", game_id=GAME_ID, decision_timestamp="2026-09-12T09:00:00+00:00", kickoff_timestamp=KICKOFF,
        decision=Decision.QUALIFIED_BET, reason_codes=(ReasonCode.MODEL_MARKET_DISAGREEMENT,), decision_rule_version="v1",
        market_type="spread", input_packet_hash="h", validation_status="PROSPECTIVE",
    )
    append_decision_record(record, SEASON, WEEK)

    write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=FixtureContentWriterProvider(CONTENT_FIXTURE), prompt_template_text=PROMPT_TEMPLATE,
        run_id="test_run_2", now="2026-09-12T12:00:00+00:00",
    )

    stored = read_preview_run(SEASON, WEEK, GAME_ID, "test_run_2")
    assert stored["context"]["decisions"]["spread"]["decision"] == "QUALIFIED_BET"


def test_write_prediction_preview_when_provider_not_configured_persists_an_explicit_failure(content_data_dir):
    class _AlwaysUnconfigured(ContentWriterProvider):
        def write(self, prompt):
            raise ContentWriterNotConfiguredError("no key configured")

    result = write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=_AlwaysUnconfigured(), prompt_template_text=PROMPT_TEMPLATE,
        run_id="test_run_3", now="2026-09-12T12:00:00+00:00",
    )
    assert result["status"] == "failed"
    assert result["failure_status"] == "PROVIDER_UNAVAILABLE"
    stored = read_preview_run(SEASON, WEEK, GAME_ID, "test_run_3")
    assert stored["manifest"]["kind"] == "failed_run"


def test_write_prediction_preview_on_cost_budget_exceeded_persists_an_explicit_failure(content_data_dir):
    class _AlwaysOverBudget(ContentWriterProvider):
        def write(self, prompt):
            raise ResearchCostBudgetExceededError("refusing: over budget")

    result = write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=_AlwaysOverBudget(), prompt_template_text=PROMPT_TEMPLATE,
        run_id="test_run_4", now="2026-09-12T12:00:00+00:00",
    )
    assert result["status"] == "failed"
    assert result["failure_status"] == "COST_BUDGET_EXCEEDED"


def test_write_prediction_preview_on_empty_response_persists_an_explicit_failure(content_data_dir):
    class _AlwaysEmpty(ContentWriterProvider):
        def write(self, prompt):
            raise ContentWriterResponseError("No text content in response")

    result = write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=_AlwaysEmpty(), prompt_template_text=PROMPT_TEMPLATE,
        run_id="test_run_5", now="2026-09-12T12:00:00+00:00",
    )
    assert result["status"] == "failed"
    assert result["failure_status"] == "EMPTY_RESPONSE"


def test_write_prediction_preview_on_unexpected_error_still_returns_not_raises(content_data_dir):
    class _AlwaysExplodes(ContentWriterProvider):
        def write(self, prompt):
            raise RuntimeError("something genuinely unexpected")

    result = write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=_AlwaysExplodes(), prompt_template_text=PROMPT_TEMPLATE,
        run_id="test_run_6", now="2026-09-12T12:00:00+00:00",
    )
    assert result["status"] == "failed"
    assert result["failure_status"] == "LLM_FAILURE"


def test_anthropic_provider_refuses_before_any_http_call_when_over_budget():
    """A real pre-flight guard, exercised with a fake key (no network reached, since the
    guard raises before `_post` is ever called)."""
    provider = AnthropicContentWriterProvider(api_key="fake-key-never-used", max_tokens=700, max_estimated_cost_usd=0.0000001)
    with pytest.raises(ResearchCostBudgetExceededError):
        provider.write("irrelevant prompt")


def test_anthropic_provider_requires_a_configured_key():
    with pytest.raises(ContentWriterNotConfiguredError):
        AnthropicContentWriterProvider(api_key=None)


def test_preview_runs_are_immutable_a_second_write_to_the_same_run_id_is_refused(content_data_dir):
    from nfl_predict.content.storage import PreviewRunAlreadyExistsError

    write_prediction_preview(
        SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
        provider=FixtureContentWriterProvider(CONTENT_FIXTURE), prompt_template_text=PROMPT_TEMPLATE,
        run_id="dup_run", now="2026-09-12T12:00:00+00:00",
    )
    with pytest.raises(PreviewRunAlreadyExistsError):
        write_prediction_preview(
            SEASON, WEEK, GAME_ID, HOME_ID, AWAY_ID, KICKOFF,
            provider=FixtureContentWriterProvider(CONTENT_FIXTURE), prompt_template_text=PROMPT_TEMPLATE,
            run_id="dup_run", now="2026-09-12T13:00:00+00:00",
        )
