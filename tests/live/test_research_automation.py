"""Phase 8A Step 28 proofs #12-13: missing LLM key produces an explicit unavailable
status, and production mode cannot use the fixture/manual LLM provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predict.live.research_automation import (
    ProductionFixtureProviderError,
    ResearchProviderStatus,
    assert_provider_allowed_in_production,
    get_production_research_provider,
)
from nfl_predict.research.llm_provider import FixtureLLMProvider
from nfl_predict.research.manual_provider import ManualResearchProvider


def test_missing_llm_key_produces_explicit_unavailable_status(monkeypatch):
    monkeypatch.delenv("NFL_RESEARCH_LLM_API_KEY", raising=False)
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    result = get_production_research_provider()
    assert result.status == ResearchProviderStatus.RESEARCH_PROVIDER_UNAVAILABLE
    assert result.provider is None
    get_settings.cache_clear()


def test_configured_key_produces_available_status(monkeypatch):
    monkeypatch.setenv("NFL_RESEARCH_LLM_API_KEY", "test-key-not-a-real-secret")
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    result = get_production_research_provider()
    assert result.status == ResearchProviderStatus.AVAILABLE
    assert result.provider is not None
    get_settings.cache_clear()


def test_fixture_provider_is_rejected_in_production():
    fixture_path = Path(__file__).resolve().parents[1] / "research" / "fixtures" / "llm_fixture.json"
    provider = FixtureLLMProvider(fixture_path)
    with pytest.raises(ProductionFixtureProviderError):
        assert_provider_allowed_in_production(provider)


def test_manual_provider_is_rejected_in_production():
    provider = ManualResearchProvider(research_output_text="{}")
    with pytest.raises(ProductionFixtureProviderError):
        assert_provider_allowed_in_production(provider)


def test_a_live_anthropic_provider_is_allowed_in_production():
    from nfl_predict.research.llm_provider import AnthropicMessagesProvider

    provider = AnthropicMessagesProvider(api_key="test-key")
    assert_provider_allowed_in_production(provider)  # does not raise
