"""Phase 7: the decision engine's structured schemas.

Nothing here calls an LLM, touches the network, or reads/writes files. `Decision` is a
closed 5-value enum - `docs/DECISION_ENGINE.md`'s update to this phase's actual categories,
not the earlier BET/LEAN/NO_BET/VETO placeholder. `ReasonCode` is a closed enum too (Step
20: "Every decision should be explainable from reason codes... do not rely only on LLM
prose") - a decision's `reason_codes` tuple is the audit trail, and the deterministic engine
(`engine.py`) never invokes an LLM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum


class Decision(str, Enum):
    NO_BET = "NO_BET"
    WATCH = "WATCH"
    LEAN = "LEAN"
    QUALIFIED_BET = "QUALIFIED_BET"
    VETO = "VETO"


DEFAULT_DECISION = Decision.NO_BET


class ReasonCode(str, Enum):
    MODEL_MARKET_DISAGREEMENT = "MODEL_MARKET_DISAGREEMENT"
    MODEL_MARKET_AGREEMENT = "MODEL_MARKET_AGREEMENT"
    MODEL_AGREEMENT_HIGH = "MODEL_AGREEMENT_HIGH"
    MODEL_DISAGREEMENT_HIGH = "MODEL_DISAGREEMENT_HIGH"
    RESEARCH_SUPPORTS_MODEL = "RESEARCH_SUPPORTS_MODEL"
    RESEARCH_SUPPORTS_MARKET = "RESEARCH_SUPPORTS_MARKET"
    RESEARCH_MIXED = "RESEARCH_MIXED"
    RESEARCH_NO_MATERIAL_INFORMATION = "RESEARCH_NO_MATERIAL_INFORMATION"
    MAJOR_INJURY_RISK = "MAJOR_INJURY_RISK"
    QB_UNCERTAINTY = "QB_UNCERTAINTY"
    STALE_MARKET = "STALE_MARKET"
    STALE_RESEARCH = "STALE_RESEARCH"
    MISSING_LIVE_DATA = "MISSING_LIVE_DATA"
    HIGH_MODEL_UNCERTAINTY = "HIGH_MODEL_UNCERTAINTY"
    NO_DEMONSTRATED_EDGE = "NO_DEMONSTRATED_EDGE"
    VETO_CONSIDERATION_UNRESOLVED = "VETO_CONSIDERATION_UNRESOLVED"
    DISAGREEMENT_BELOW_MINIMUM = "DISAGREEMENT_BELOW_MINIMUM"
    INSUFFICIENT_MODEL_COVERAGE = "INSUFFICIENT_MODEL_COVERAGE"
    DEFAULT_NO_EVIDENCE = "DEFAULT_NO_EVIDENCE"


@dataclass(frozen=True)
class ModelPoint:
    model_id: str
    available: bool
    predicted_margin: float | None = None
    home_win_probability: float | None = None
    uncertainty_note: str | None = None


@dataclass(frozen=True)
class ModelAgreementDescriptor:
    """Step 8: a qualitative description of how the AVAILABLE models relate to each other -
    never a claim about profitability. `models_considered` lists exactly which models
    contributed (missing models are never silently treated as agreeing - Step 21)."""

    models_considered: tuple[str, ...]
    all_agree_on_direction: bool | None  # None if fewer than 2 models are available to compare
    margin_dispersion: float | None  # max(margin) - min(margin) across available models, None if <2 available
    n_models_available: int


@dataclass(frozen=True)
class MarketPoint:
    """Phase 8A provenance correction: `provider_event_id`/`market_snapshot_reference`/
    `underlying_snapshot_keys` let a MarketPoint point back at the EXACT persisted odds rows
    (in `market/live_snapshots/event=<provider_event_id>/snapshots.parquet`) it was computed
    from - never just a final averaged number with no way to reconstruct it. All default to
    None/() so a `MarketPoint(available=False)` (no real market data at all) never has to
    fabricate provenance for data that doesn't exist."""

    available: bool
    source: str | None = None
    home_spread_traditional: float | None = None
    home_spread_price: int | None = None
    away_spread_price: int | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    no_vig_home_win_probability: float | None = None
    snapshot_timestamp: str | None = None
    provider_event_id: str | None = None
    consensus_algorithm_version: str | None = None
    consensus_book_keys: tuple[str, ...] = ()
    consensus_excluded_book_keys: tuple[str, ...] = ()
    consensus_exclusion_reason: str | None = None
    underlying_snapshot_keys: tuple[str, ...] = ()
    market_snapshot_reference: str | None = None


@dataclass(frozen=True)
class ResearchPoint:
    available: bool
    research_id: str | None = None
    classification: str | None = None  # ResearchClassification.value, or None if unavailable
    materiality_level: int | None = None
    source_quality_note: str | None = None
    unresolved_risks: tuple[str, ...] = ()
    research_timestamp: str | None = None


@dataclass(frozen=True)
class SystemHealth:
    missing_required_data: tuple[str, ...]
    stale_flags: tuple[str, ...]
    market_age_seconds: float | None
    research_age_seconds: float | None


@dataclass(frozen=True)
class DecisionInputPacket:
    """Immutable. Every decision is computed from exactly this - no field here is ever
    mutated after construction, and `engine.decide()` never reaches outside it (Step 23:
    determinism)."""

    game_id: str
    decision_timestamp: str
    kickoff_timestamp: str | None

    elo: ModelPoint
    ridge: ModelPoint
    lightgbm: ModelPoint
    model_agreement: ModelAgreementDescriptor

    market: MarketPoint
    research: ResearchPoint
    system_health: SystemHealth

    def content_hash(self) -> str:
        """Mirrors `research.input_packet.ResearchInputPacket.content_hash()` - a real hash
        of every input this decision was actually computed from, never the `"n/a"` placeholder
        `run.py` used to write for every real live decision (Phase 8A provenance
        correction)."""
        canonical = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    game_id: str
    decision_timestamp: str
    kickoff_timestamp: str | None
    decision: Decision
    reason_codes: tuple[ReasonCode, ...]
    decision_rule_version: str
    market_type: str  # "spread" or "moneyline" - Step 16, totals not yet supported
    input_packet_hash: str
    validation_status: str  # "PROSPECTIVE" for all Phase 7 decisions (Step 18)
    # Phase 8A provenance correction: lets a stored DecisionRecord identify the exact
    # persisted market/research artifacts it was computed from, without needing the full
    # DecisionInputPacket - None/empty when that input was unavailable for this decision.
    market_provider_event_id: str | None = None
    market_snapshot_reference: str | None = None
    market_snapshot_timestamp: str | None = None
    research_id: str | None = None
