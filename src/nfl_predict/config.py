"""Single entry point for configuration.

Two distinct kinds of configuration, never mixed (see docs/ARCHITECTURE.md#configuration):

- Versioned, non-secret config lives in ``config/*.yaml`` and is read here via
  :func:`load_yaml_config` / the ``get_*_config`` helpers.
- Secrets and environment-specific values live in environment variables (loaded from a
  local ``.env`` file in development) and are read here via :func:`get_settings`.

No other module should call ``os.environ`` directly or open a YAML config file directly —
route everything through this module so there is exactly one place that knows how
configuration is sourced.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_VALID_STORAGE_BACKENDS = {"sqlite", "postgres"}

# Loaded once on import. Safe to call even if no .env file exists (dev machines without
# one simply fall back to whatever is already in the environment).
load_dotenv(PROJECT_ROOT / ".env")


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load and parse ``config/<name>.yaml``.

    Raises FileNotFoundError with an actionable message if the file is missing, rather than
    returning an empty/default config that would mask a misconfigured deployment.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected config file at {path}, but it does not exist. "
            "Config files are checked into the repository; if you're seeing this, "
            "something removed or renamed it."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_features_config() -> dict[str, Any]:
    """Return the parsed contents of config/features.yaml (see docs/FEATURE_DICTIONARY.md)."""
    return load_yaml_config("features")


def get_sources_config() -> dict[str, Any]:
    """Return the parsed contents of config/sources.yaml (see docs/DATA_SOURCES.md)."""
    return load_yaml_config("sources")


def get_decision_thresholds_config() -> dict[str, Any]:
    """Return the parsed contents of config/decision_thresholds.yaml (see docs/DECISION_ENGINE.md).
    Superseded by `get_decision_rules_config` (Phase 7) for anything the decision engine
    actually reads - kept only because it still validly documents "not configured" (all
    null) and nothing has migrated its schema forward."""
    return load_yaml_config("decision_thresholds")


def get_decision_rules_config() -> dict[str, Any]:
    """Return the parsed contents of config/decision_rules.yaml (Phase 7 - see
    docs/DECISION_ENGINE.md and docs/PHASE7_DECISION_ENGINE_REPORT.md). This is what
    `nfl_predict.decision.rules_config` actually loads."""
    return load_yaml_config("decision_rules")


def get_feature_engine_config() -> dict[str, Any]:
    """Return the parsed contents of config/feature_engine.yaml (see docs/PHASE2_FEATURE_REPORT.md)."""
    return load_yaml_config("feature_engine")


def get_venues_config() -> dict[str, Any]:
    """Return the parsed contents of config/venues.yaml (Phase 8A - stadium coordinates
    and roof type, see src/nfl_predict/live/weather_provider.py)."""
    return load_yaml_config("venues")


def get_llm_pricing_config() -> dict[str, Any]:
    """Return the parsed contents of config/llm_pricing.yaml (Phase 8A correction - the
    ONLY source of per-model dollar pricing; nfl_predict.research.cost_tracking computes
    cost from this, never a hard-coded number). See
    docs/PHASE8A_LIVE_PIPELINE_REPORT.md for the real fetched-pricing provenance."""
    return load_yaml_config("llm_pricing")


@dataclass(frozen=True)
class Settings:
    """Environment-derived settings. Never holds anything read from config/*.yaml."""

    storage_backend: str
    data_dir: Path
    database_url: str | None
    log_level: str
    odds_api_key: str | None  # Phase 5 live odds provider (The Odds API) - see docs/DATA_SOURCES.md
    research_llm_api_key: str | None  # Phase 6 LLM research provider - see docs/PHASE6_RESEARCH_AGENT_REPORT.md
    weather_api_key: str | None  # Phase 6 WeatherProvider - see docs/DATA_SOURCES.md
    news_api_key: str | None  # Phase 6 NewsResearchProvider - see docs/DATA_SOURCES.md
    injury_api_key: str | None  # Phase 8A structured InjuryProvider - see docs/DATA_SOURCES.md
    odds_max_credits_per_run: int  # Phase 8A correction: hard per-run cap on The Odds API credits (a 500-credit exhaustion incident is documented in docs/PHASE8A_LIVE_PIPELINE_REPORT.md)
    research_max_web_searches_per_game: int  # Phase 8A cost-controls correction: hard per-matchup cap on Anthropic web-search tool calls
    research_max_estimated_cost_per_game_usd: float  # Phase 8A cost-controls correction: refuse a research call whose known-floor cost (output tokens + searches) would already exceed this
    research_max_games_per_run: int  # Phase 8A cost-controls correction: hard cap on how many games get a real research call in one run, absent --research-all
    content_max_estimated_cost_per_game_usd: float | None  # Phase 8B follow-up (prediction-writer content step): pre-flight cost-floor cap per game, same guard mechanism as research's - None means no cap configured


def _require_valid_storage_backend(value: str) -> str:
    if value not in _VALID_STORAGE_BACKENDS:
        raise ValueError(
            f"NFL_STORAGE_BACKEND={value!r} is not valid; expected one of "
            f"{sorted(_VALID_STORAGE_BACKENDS)}."
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings sourced from environment variables.

    Cached because environment variables are not expected to change mid-process; tests that
    need different settings should call ``get_settings.cache_clear()`` after patching
    ``os.environ``.
    """
    storage_backend = _require_valid_storage_backend(
        os.environ.get("NFL_STORAGE_BACKEND", "sqlite")
    )
    data_dir = Path(os.environ.get("NFL_DATA_DIR", str(PROJECT_ROOT / "data")))
    database_url = os.environ.get("NFL_DATABASE_URL") or None
    log_level = os.environ.get("NFL_LOG_LEVEL", "INFO").upper()
    odds_api_key = os.environ.get("NFL_ODDS_API_KEY") or None
    research_llm_api_key = os.environ.get("NFL_RESEARCH_LLM_API_KEY") or None
    weather_api_key = os.environ.get("NFL_WEATHER_API_KEY") or None
    news_api_key = os.environ.get("NFL_NEWS_API_KEY") or None
    injury_api_key = os.environ.get("NFL_INJURY_API_KEY") or None
    odds_max_credits_per_run = int(os.environ.get("NFL_ODDS_MAX_CREDITS_PER_RUN", "5"))
    research_max_web_searches_per_game = int(os.environ.get("NFL_RESEARCH_MAX_WEB_SEARCHES_PER_GAME", "3"))
    research_max_estimated_cost_per_game_usd = float(os.environ.get("NFL_RESEARCH_MAX_ESTIMATED_COST_PER_GAME", "1.00"))
    research_max_games_per_run = int(os.environ.get("NFL_RESEARCH_MAX_GAMES_PER_RUN", "3"))
    _content_cost_raw = os.environ.get("NFL_CONTENT_MAX_COST_PER_GAME", "0.05")
    content_max_estimated_cost_per_game_usd = float(_content_cost_raw) if _content_cost_raw else None
    return Settings(
        storage_backend=storage_backend,
        data_dir=data_dir,
        database_url=database_url,
        log_level=log_level,
        odds_api_key=odds_api_key,
        research_llm_api_key=research_llm_api_key,
        weather_api_key=weather_api_key,
        news_api_key=news_api_key,
        injury_api_key=injury_api_key,
        odds_max_credits_per_run=odds_max_credits_per_run,
        research_max_web_searches_per_game=research_max_web_searches_per_game,
        research_max_estimated_cost_per_game_usd=research_max_estimated_cost_per_game_usd,
        research_max_games_per_run=research_max_games_per_run,
        content_max_estimated_cost_per_game_usd=content_max_estimated_cost_per_game_usd,
    )
