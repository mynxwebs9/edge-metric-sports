"""Phase 6 Step 20/24: current-data provider interfaces are modular and credential-gated,
never fabricate live data."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predict.research.current_data_providers import (
    FixtureInjuryProvider,
    InjuryProvider,
    LiveNewsResearchProvider,
    LiveWeatherProvider,
    ProviderNotConfiguredError,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "injury_fixture.json"


def test_fixture_injury_provider_implements_the_interface():
    provider = FixtureInjuryProvider(FIXTURE_PATH)
    assert isinstance(provider, InjuryProvider)
    entries = provider.get_injury_report("2026_01_DEN_KC")
    assert len(entries) == 2
    assert entries[0].status == "OUT"


def test_fixture_injury_provider_returns_empty_list_for_unknown_game():
    provider = FixtureInjuryProvider(FIXTURE_PATH)
    assert provider.get_injury_report("nonexistent_game") == []


def test_live_news_provider_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("NFL_NEWS_API_KEY", raising=False)
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ProviderNotConfiguredError):
        LiveNewsResearchProvider()
    get_settings.cache_clear()


def test_live_weather_provider_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("NFL_WEATHER_API_KEY", raising=False)
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(ProviderNotConfiguredError):
        LiveWeatherProvider()
    get_settings.cache_clear()


def test_live_news_provider_never_fabricates_results():
    provider = LiveNewsResearchProvider(api_key="test-key")
    with pytest.raises(NotImplementedError):
        provider.search("test query")
