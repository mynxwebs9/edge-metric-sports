"""Phase 8A Step 11: production structured-injury-provider gating, mirroring
`nfl_predict.live.odds_automation` / `nfl_predict.live.research_automation`'s pattern - an
explicit `INJURY_PROVIDER_UNAVAILABLE` status when `NFL_INJURY_API_KEY` is unset, never a
silent fallback to `FixtureInjuryProvider`.

This is a distinct boundary from `nfl_predict.live.research_automation`'s LLM research
provider: `get_production_injury_provider()` gates access to STRUCTURED INJURY DATA (a
direct feed of official injury designations), while the research provider's `qb_status`
and similar fields are LLM/WEB RESEARCH (an LLM's summary of claims found via search, tiered
and fact/opinion-separated per `nfl_predict.research.schemas`). A caller that needs injury
context must consult both separately - this module never merges the two into one status or
one field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nfl_predict.research.current_data_providers import InjuryProvider, LiveInjuryProvider, ProviderNotConfiguredError


class InjuryProviderStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    INJURY_PROVIDER_UNAVAILABLE = "INJURY_PROVIDER_UNAVAILABLE"


@dataclass(frozen=True)
class InjuryProviderResult:
    status: InjuryProviderStatus
    provider: InjuryProvider | None
    detail: str


def get_production_injury_provider() -> InjuryProviderResult:
    """The ONLY function production code should call to get a structured injury-data
    provider. Never returns a fixture provider - if no live key is configured, returns
    `INJURY_PROVIDER_UNAVAILABLE` with `provider=None` rather than pretending structured
    injury data is available."""
    try:
        provider = LiveInjuryProvider()
    except ProviderNotConfiguredError as e:
        return InjuryProviderResult(status=InjuryProviderStatus.INJURY_PROVIDER_UNAVAILABLE, provider=None, detail=str(e))
    return InjuryProviderResult(status=InjuryProviderStatus.AVAILABLE, provider=provider, detail="LiveInjuryProvider configured via NFL_INJURY_API_KEY")
