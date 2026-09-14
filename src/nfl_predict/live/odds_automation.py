"""Phase 8A Step 8/11: production odds-provider gating, mirroring
`nfl_predict.live.research_automation`'s pattern - an explicit `ODDS_PROVIDER_UNAVAILABLE`
status when `NFL_ODDS_API_KEY` is unset, never a silent fallback to `FixtureOddsProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nfl_predict.market.odds_provider import OddsProvider, OddsProviderNotConfiguredError, TheOddsAPIProvider


class OddsProviderStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    ODDS_PROVIDER_UNAVAILABLE = "ODDS_PROVIDER_UNAVAILABLE"


@dataclass(frozen=True)
class OddsProviderResult:
    status: OddsProviderStatus
    provider: OddsProvider | None
    detail: str


def get_production_odds_provider() -> OddsProviderResult:
    try:
        provider = TheOddsAPIProvider()
    except OddsProviderNotConfiguredError as e:
        return OddsProviderResult(status=OddsProviderStatus.ODDS_PROVIDER_UNAVAILABLE, provider=None, detail=str(e))
    return OddsProviderResult(status=OddsProviderStatus.AVAILABLE, provider=provider, detail="TheOddsAPIProvider configured via NFL_ODDS_API_KEY")
