"""Phase 8B follow-up: structured schemas for the generated game-preview article step.

Nothing here calls an LLM or touches storage - pure data shape, mirroring the
`research.schemas` pattern (a real content record vs. an explicit `FailedPreviewRun`, never
a silent empty-string standing in for a failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PreviewFailureStatus(str, Enum):
    """A failed preview generation never silently becomes an empty/missing article - it gets
    one of these instead, recorded alongside the run."""

    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    COST_BUDGET_EXCEEDED = "COST_BUDGET_EXCEEDED"
    LLM_FAILURE = "LLM_FAILURE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"


@dataclass(frozen=True)
class GamePreview:
    preview_id: str
    game_id: str
    generated_at: str
    prompt_version: str
    model_provider: str
    model_name: str
    context_hash: str  # ties this article to the exact gathered context it was written from
    text: str


@dataclass(frozen=True)
class FailedPreviewRun:
    preview_id: str
    game_id: str
    generated_at: str
    prompt_version: str
    context_hash: str
    failure_status: PreviewFailureStatus
    failure_detail: str
