"""Phase 6: the structured schemas every research artifact is built from.

Nothing in this module calls an LLM, touches the network, or reads/writes files - it is
pure data shape, imported by every other `nfl_predict.research` module and by
`nfl_predict.decision` (Phase 7+) so the boundary between "what the research layer produces"
and "what anything downstream may read" is a typed contract, not a free-form dict.

Per `docs/RESEARCH_AGENT.md` and this phase's brief: nothing here can express a numeric
prediction, a modified spread, or a betting recommendation - there is no field for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceTier(str, Enum):
    """Phase 6 Step 5's documented source hierarchy, most to least trusted."""

    TIER_1_OFFICIAL = "TIER_1_OFFICIAL"  # official team/league reports, injury reports, press conferences
    TIER_2_REPUTABLE_REPORTER = "TIER_2_REPUTABLE_REPORTER"  # beat reporters, national reporters, credible news orgs
    TIER_3_ANALYTICS_PUBLICATION = "TIER_3_ANALYTICS_PUBLICATION"  # established analytics/betting publications
    TIER_4_COMMUNITY = "TIER_4_COMMUNITY"  # Reddit and similar - never treated as verified fact without corroboration


class ClaimCategory(str, Enum):
    """Step 7's fact/opinion separation - a claim belongs to exactly one of these, never
    blended. "Never convert analyst consensus into a factual football condition" is enforced
    by construction: only VERIFIED_FACT-category claims may populate the *_status summary
    fields on `ResearchFindings` (see `evaluator.py`)."""

    VERIFIED_FACT = "VERIFIED_FACT"
    REPORTED_NOT_CONFIRMED = "REPORTED_NOT_CONFIRMED"
    ANALYST_OPINION = "ANALYST_OPINION"
    COMMUNITY_SENTIMENT = "COMMUNITY_SENTIMENT"
    MODEL_ANALYTICS_OPINION = "MODEL_ANALYTICS_OPINION"


class MaterialityLevel(int, Enum):
    """Step 9's materiality rubric. An int Enum so ordering/comparison work naturally
    (`finding.materiality_level >= MaterialityLevel.MODERATE`), while `.name` still gives
    the human label."""

    NOISE = 0
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    CRITICAL = 4


MATERIALITY_RUBRIC: dict[MaterialityLevel, str] = {
    MaterialityLevel.NOISE: "Irrelevant/noise - no plausible effect on expected team strength.",
    MaterialityLevel.MINOR: "Minor - unlikely to materially affect expected team strength.",
    MaterialityLevel.MODERATE: "Moderate - could plausibly affect matchup expectations.",
    MaterialityLevel.MAJOR: "Major - likely affects team strength or market interpretation.",
    MaterialityLevel.CRITICAL: "Critical - e.g. starting QB status genuinely uncertain, or similarly extreme information.",
}


class ResearchClassification(str, Enum):
    """Step 8's allowed classification enum - closed set, never a free-form string. These
    summarize the research's relationship to the EXISTING quantitative/market information;
    they never create or cancel a bet (Phase 7's job, later, using deterministic code)."""

    SUPPORTS_MODEL = "SUPPORTS_MODEL"
    SUPPORTS_MARKET = "SUPPORTS_MARKET"
    MIXED = "MIXED"
    NO_MATERIAL_NEW_INFORMATION = "NO_MATERIAL_NEW_INFORMATION"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    VETO_CONSIDERATION = "VETO_CONSIDERATION"


class FailureStatus(str, Enum):
    """Step 23: a failed research run must never silently collapse into
    NO_MATERIAL_NEW_INFORMATION - it gets one of these instead, and the run is stored with
    this status rather than a `ResearchClassification`."""

    NO_WEB_ACCESS = "NO_WEB_ACCESS"
    SOURCE_TIMEOUT = "SOURCE_TIMEOUT"
    LLM_FAILURE = "LLM_FAILURE"
    INVALID_JSON = "INVALID_JSON"
    INSUFFICIENT_SOURCES = "INSUFFICIENT_SOURCES"
    CONFLICTING_REPORTS = "CONFLICTING_REPORTS"
    COST_BUDGET_EXCEEDED = "COST_BUDGET_EXCEEDED"  # Phase 8A cost-controls correction: a real, computed pre-flight cost cap refused this call before any HTTP request
    DOWNSTREAM_PROCESSING_FAILURE = "DOWNSTREAM_PROCESSING_FAILURE"  # a real LLM call succeeded and was parsed, but evaluation/persistence/ledger-write failed - never conflated with LLM_FAILURE or INVALID_JSON, both of which happen before this point; usage/cost already incurred is preserved via the caller's returned "cost" field regardless


@dataclass(frozen=True)
class SourceRecord:
    """Step 6: every meaningful claim's provenance. `publication_timestamp` is None (never
    fabricated) when it cannot be established - see Step 6's explicit "record null rather
    than inventing one.\""""

    source_url: str
    source_title: str
    publisher: str
    source_tier: SourceTier
    retrieval_timestamp: str  # UTC ISO 8601 - when THIS research run observed the source
    publication_timestamp: str | None  # UTC ISO 8601 or None if not establishable
    corroborated_by: tuple[str, ...] = field(default_factory=tuple)  # other source_urls independently confirming the same claim


@dataclass(frozen=True)
class Claim:
    """One factual/opinion statement plus everything needed to judge it - never bare text.
    `category` fixes which of the five Step-7 buckets this belongs in; a `Claim` never
    silently changes category after being written (immutable dataclass)."""

    text: str
    category: ClaimCategory
    sources: tuple[SourceRecord, ...]
    materiality_level: MaterialityLevel
    confidence_in_fact: float  # 0.0-1.0, confidence the underlying claim is accurate/current - NOT confidence in its effect on the game
    reason: str  # why this materiality/confidence was assigned

    def __post_init__(self):
        if not (0.0 <= self.confidence_in_fact <= 1.0):
            raise ValueError(f"confidence_in_fact must be in [0.0, 1.0], got {self.confidence_in_fact}")
        if self.category == ClaimCategory.VERIFIED_FACT and not self.sources:
            raise ValueError("A VERIFIED_FACT claim must carry at least one source - unsupported factual assertions are not allowed (Step 6).")


@dataclass(frozen=True)
class ExternalPrediction:
    """Step 12: another person's/model's prediction, collected as its own category -
    deduplicated upstream (see `evaluator.py`) so syndicated copies of the same analysis
    are never counted as independent corroboration."""

    source: str
    prediction_text: str
    market: str | None  # e.g. "spread", "moneyline", "total" - None if not market-specific
    line_at_publication: str | None
    publication_timestamp: str | None
    stated_confidence: str | None


@dataclass(frozen=True)
class ResearchFindings:
    """Step 8's structured output schema. Every `*_status` field is a short, human-readable
    summary string built ONLY from VERIFIED_FACT-category claims (enforced in
    `evaluator.py`, never in the LLM's raw output) - never from opinions or unconfirmed
    reports. `missing_information` names what could not be established, so a reader can see
    what was NOT checked, not just what was found."""

    research_id: str
    game_id: str
    research_timestamp: str
    prompt_version: str
    model_provider: str
    model_name: str
    input_packet_hash: str

    material_facts: tuple[Claim, ...]
    uncertain_reports: tuple[Claim, ...]
    external_model_opinions: tuple[ExternalPrediction, ...]
    analyst_opinions: tuple[Claim, ...]

    qb_status: str
    ol_status: str
    skill_position_status: str
    defensive_personnel_status: str
    weather_status: str
    coaching_status: str
    missing_information: tuple[str, ...]

    research_classification: ResearchClassification


@dataclass(frozen=True)
class FailedResearchRun:
    """What gets stored instead of `ResearchFindings` when a run fails - explicit, never
    coerced into a fake "no material information" result (Step 23)."""

    research_id: str
    game_id: str
    research_timestamp: str
    prompt_version: str
    model_provider: str
    model_name: str
    input_packet_hash: str
    failure_status: FailureStatus
    failure_detail: str


@dataclass(frozen=True)
class EvaluationResult:
    """Step 15's evaluator output - re-derives/confirms the final classification and
    materiality after checking internal consistency, downgrading weak sources, and
    deduplicating evidence. Never generates a betting pick."""

    evaluation_id: str
    research_id: str
    game_id: str
    evaluation_timestamp: str
    prompt_version: str
    model_provider: str
    model_name: str

    final_classification: ResearchClassification
    overall_materiality: MaterialityLevel
    downgraded_claim_texts: tuple[str, ...]  # claims whose category/confidence the evaluator lowered, and why (see notes)
    duplicate_claim_groups: tuple[tuple[str, ...], ...]  # groups of claim texts judged to be the same underlying evidence
    unsupported_claim_texts: tuple[str, ...]  # claims the evaluator could not verify against any source record
    notes: str
