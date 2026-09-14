"""Phase 8A Step 13: production research automation.

Phase 6 used `ManualResearchProvider` because no LLM API was configured (a human/agent
session stood in for the automated research step). This module is the production entry
point: it returns a real, live `AnthropicMessagesProvider` when `NFL_RESEARCH_LLM_API_KEY`
is set, and an explicit `RESEARCH_PROVIDER_UNAVAILABLE` status - never a silent fallback to
`FixtureLLMProvider` or `ManualResearchProvider` - when it is not. Production mode is a hard
gate: `get_production_research_provider` refuses to return a fixture/manual provider even
if one is explicitly requested, because those exist for tests and human-in-the-loop pilots,
not for unattended production runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nfl_predict.config import get_settings
from nfl_predict.research.llm_provider import AnthropicMessagesProvider, LLMProviderNotConfiguredError, LLMResearchProvider


class ResearchProviderStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    RESEARCH_PROVIDER_UNAVAILABLE = "RESEARCH_PROVIDER_UNAVAILABLE"


class ProductionFixtureProviderError(Exception):
    """Raised if production mode is asked to use a fixture/manual provider - those are for
    tests and human-in-the-loop pilots only, never unattended production runs."""


@dataclass(frozen=True)
class ResearchProviderResult:
    status: ResearchProviderStatus
    provider: LLMResearchProvider | None
    detail: str


_DISALLOWED_IN_PRODUCTION = ("FixtureLLMProvider", "ManualResearchProvider")


def get_production_research_provider() -> ResearchProviderResult:
    """The ONLY function production code should call to get a research provider. Never
    returns a fixture/manual provider - if no live key is configured, returns
    `RESEARCH_PROVIDER_UNAVAILABLE` with `provider=None` rather than pretending research
    can proceed."""
    try:
        provider = AnthropicMessagesProvider()
    except LLMProviderNotConfiguredError as e:
        return ResearchProviderResult(status=ResearchProviderStatus.RESEARCH_PROVIDER_UNAVAILABLE, provider=None, detail=str(e))
    return ResearchProviderResult(status=ResearchProviderStatus.AVAILABLE, provider=provider, detail="AnthropicMessagesProvider configured via NFL_RESEARCH_LLM_API_KEY")


def assert_provider_allowed_in_production(provider: LLMResearchProvider) -> None:
    """A defense-in-depth check: raises if `provider` is one of the disallowed
    test/pilot-only provider classes, even if a caller tried to hand one to a production
    code path directly."""
    class_name = type(provider).__name__
    if class_name in _DISALLOWED_IN_PRODUCTION:
        raise ProductionFixtureProviderError(
            f"{class_name} may not be used as the research provider in production - it is "
            "for tests/human-in-the-loop pilots only. Configure NFL_RESEARCH_LLM_API_KEY "
            "and use get_production_research_provider() instead."
        )
