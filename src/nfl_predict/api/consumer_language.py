"""Phase 8B: translates internal enum values into consumer-facing text. Pure data, no
network/storage access, so it is trivially unit-testable and reusable by every route.

Never invents a translation for a code that doesn't exist - `translate_reason_code` and the
other lookups here raise `KeyError` on an unknown code rather than falling back to the raw
enum text, so a newly added reason code/classification is a loud, caught-by-tests failure to
translate, not a silent leak of ugly internal enum text into the UI.
"""

from __future__ import annotations

DECISION_LABELS: dict[str, str] = {
    "NO_BET": "No Bet",
    "WATCH": "Watch",
    "LEAN": "Lean",
    # Deliberately NOT "Best Bet" - QUALIFIED_BET means this market_type cleared every
    # automated decision-engine gate, which is real, but distinct from "this is one of our
    # official, published Best Bets" (a separate, deliberate publish step - see
    # DecisionBlock.is_published_best_bet for that fact instead). Reusing "Best Bet" here
    # made both a game's spread AND moneyline decision cards say "BEST BET" even when only
    # one was ever actually published - a real, confirmed reader-facing confusion.
    "QUALIFIED_BET": "Qualified Bet",
    "VETO": "Veto",
}

REASON_CODE_LABELS: dict[str, str] = {
    "MODEL_MARKET_DISAGREEMENT": "Model and market disagree meaningfully",
    "MODEL_MARKET_AGREEMENT": "Model and market are aligned",
    "MODEL_AGREEMENT_HIGH": "Our models strongly agree with each other",
    "MODEL_DISAGREEMENT_HIGH": "Our models disagree too much with each other",
    "RESEARCH_SUPPORTS_MODEL": "Current information supports the model's view",
    "RESEARCH_SUPPORTS_MARKET": "Current information supports the market price",
    "RESEARCH_MIXED": "Current information points in mixed directions",
    "RESEARCH_NO_MATERIAL_INFORMATION": "No material new information found",
    "MAJOR_INJURY_RISK": "A major injury situation adds real risk",
    "QB_UNCERTAINTY": "Starting quarterback status is uncertain",
    "STALE_MARKET": "Market data is too old to trust right now",
    "STALE_RESEARCH": "Research data is too old to trust right now",
    "MISSING_LIVE_DATA": "Waiting for complete market/research data",
    "HIGH_MODEL_UNCERTAINTY": "Models are too uncertain to act on",
    "NO_DEMONSTRATED_EDGE": "No demonstrated edge over the market",
    "VETO_CONSIDERATION_UNRESOLVED": "Significant unresolved pregame risk",
    "DISAGREEMENT_BELOW_MINIMUM": "Model/market gap is too small to act on",
    "INSUFFICIENT_MODEL_COVERAGE": "Not enough model coverage for this game",
    "DEFAULT_NO_EVIDENCE": "Not enough evidence to make a call",
}

RESEARCH_CLASSIFICATION_LABELS: dict[str, str] = {
    "SUPPORTS_MODEL": "Supports the model",
    "SUPPORTS_MARKET": "Supports the market",
    "MIXED": "Mixed signals",
    "NO_MATERIAL_NEW_INFORMATION": "No material new information",
    "HIGH_UNCERTAINTY": "High uncertainty",
    "VETO_CONSIDERATION": "Significant risk flagged",
}

RESEARCH_FAILURE_STATUS_LABELS: dict[str, str] = {
    "NO_WEB_ACCESS": "Research unavailable this cycle",
    "SOURCE_TIMEOUT": "Research unavailable this cycle",
    "LLM_FAILURE": "Research unavailable this cycle",
    "INVALID_JSON": "Research unavailable this cycle",
    "INSUFFICIENT_SOURCES": "Not enough sources to draw a conclusion",
    "CONFLICTING_REPORTS": "Conflicting reports found",
    "COST_BUDGET_EXCEEDED": "Research skipped this cycle",
    "DOWNSTREAM_PROCESSING_FAILURE": "Research unavailable this cycle",
}


def translate_decision(decision: str) -> str:
    return DECISION_LABELS[decision]


def translate_reason_code(code: str) -> str:
    return REASON_CODE_LABELS[code]


def translate_research_classification(classification: str) -> str:
    return RESEARCH_CLASSIFICATION_LABELS[classification]


def translate_research_failure_status(status: str) -> str:
    return RESEARCH_FAILURE_STATUS_LABELS[status]
