"""Phase 6 Step 20: provider-neutral interfaces for current structured context.

`InjuryProvider`, `WeatherProvider`, and `NewsResearchProvider` are each a small ABC so no
specific vendor is hard-coded into the research pipeline. Each has a `Fixture*Provider` for
tests/offline use; a live adapter is credential-gated exactly like Phase 5's
`TheOddsAPIProvider` and Phase 6's `AnthropicMessagesProvider` - raises `*NotConfiguredError`
rather than fabricating data when its key is unset. No live adapter's HTTP calls are wired
up yet (no specific provider has been chosen - see docs/DATA_SOURCES.md); this module
exists so a provider can be dropped in later without changing any caller.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from nfl_predict.config import get_settings


class ProviderNotConfiguredError(Exception):
    """Raised when a live current-data provider is used without its required credentials."""


@dataclass(frozen=True)
class InjuryReportEntry:
    team_id: str
    player_name: str
    position: str
    status: str  # e.g. "OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "FULL_PARTICIPANT"
    injury_description: str
    report_date: str  # UTC ISO 8601


@dataclass(frozen=True)
class WeatherForecast:
    venue: str
    forecast_timestamp: str
    kickoff_timestamp: str
    temperature_f: float | None
    wind_mph: float | None
    wind_gust_mph: float | None  # Phase 8A Step 12 - added alongside the live Open-Meteo provider
    precipitation_probability: float | None
    roof_status: str | None  # "outdoors" / "dome" / "retractable_open" / "retractable_closed" / None if unknown


@dataclass(frozen=True)
class NewsItem:
    headline: str
    url: str
    publisher: str
    published_at: str | None
    retrieved_at: str


class InjuryProvider(ABC):
    @abstractmethod
    def get_injury_report(self, game_id: str) -> list[InjuryReportEntry]: ...


class WeatherProvider(ABC):
    @abstractmethod
    def get_forecast(self, game_id: str) -> WeatherForecast | None: ...


class NewsResearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[NewsItem]: ...


class FixtureInjuryProvider(InjuryProvider):
    def __init__(self, fixture_path: str | Path):
        self._fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))

    def get_injury_report(self, game_id: str) -> list[InjuryReportEntry]:
        return [InjuryReportEntry(**e) for e in self._fixture.get(game_id, [])]


class FixtureWeatherProvider(WeatherProvider):
    def __init__(self, fixture_path: str | Path):
        self._fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))

    def get_forecast(self, game_id: str) -> WeatherForecast | None:
        entry = self._fixture.get(game_id)
        return WeatherForecast(**entry) if entry else None


class FixtureNewsResearchProvider(NewsResearchProvider):
    def __init__(self, fixture_path: str | Path):
        self._fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))

    def search(self, query: str, max_results: int = 10) -> list[NewsItem]:
        raw = self._fixture.get(query, [])
        return [NewsItem(**e) for e in raw[:max_results]]


class LiveNewsResearchProvider(NewsResearchProvider):
    """Gated exactly like the other live providers - no vendor chosen/wired yet."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key if api_key is not None else get_settings().news_api_key
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "No news-research API key is configured - set NFL_NEWS_API_KEY in .env. "
                "Live calls are refused, not fabricated."
            )

    def search(self, query: str, max_results: int = 10) -> list[NewsItem]:
        raise NotImplementedError("No news-research provider has been chosen/wired up yet - see docs/DATA_SOURCES.md.")


class LiveInjuryProvider(InjuryProvider):
    """Phase 8A Step 11: a genuinely STRUCTURED injury-report feed (e.g. official
    team/league injury designations), gated exactly like the other live providers - never
    wired to a chosen vendor yet, see docs/DATA_SOURCES.md. This is a different boundary
    than the LLM research agent's `qb_status`/`skill_position_status`/etc. fields in
    `research.schemas.ResearchFindings`: those are LLM/WEB RESEARCH (unstructured claims,
    tiered by `SourceTier`, subject to `ClaimCategory` fact/opinion separation), while this
    provider's output is STRUCTURED INJURY DATA (a direct feed, not an LLM's summary of one).
    The two are never merged into a single field - callers that need both must read each
    separately and keep the distinction visible downstream."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key if api_key is not None else get_settings().injury_api_key
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "No structured injury-data API key is configured - set NFL_INJURY_API_KEY "
                "in .env. Live calls are refused, not fabricated."
            )

    def get_injury_report(self, game_id: str) -> list[InjuryReportEntry]:
        raise NotImplementedError("No structured injury-data provider has been chosen/wired up yet - see docs/DATA_SOURCES.md.")


class LiveWeatherProvider(WeatherProvider):
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key if api_key is not None else get_settings().weather_api_key
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "No weather API key is configured - set NFL_WEATHER_API_KEY in .env. "
                "Live calls are refused, not fabricated."
            )

    def get_forecast(self, game_id: str) -> WeatherForecast | None:
        raise NotImplementedError("No weather provider has been chosen/wired up yet - see docs/DATA_SOURCES.md.")
