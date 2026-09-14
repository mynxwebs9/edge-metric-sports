"""Phase 6 Step 2: a configurable research-trigger mechanism.

Scores each game's RESEARCH PRIORITY from whichever trigger inputs happen to be available -
model-market disagreement, unusual line movement, QB/injury/OL/coaching uncertainty flags,
model uncertainty, cross-model (Elo vs. Ridge vs. LightGBM) disagreement, and
information-quality concerns. A missing input contributes 0 and is listed in
`inputs_unavailable` - it is never penalized or fabricated.

Per the brief: "Because Phase 5 did NOT prove that large model-market disagreement predicts
value, do not assume a disagreement automatically creates a bet. It merely creates a
RESEARCH PRIORITY." **This module's output is a ranking signal only - it never gates
research and never implies a bet.** `ALWAYS_RESEARCHABLE = True` documents that any game may
be researched regardless of its computed score, per Step 2's explicit instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALWAYS_RESEARCHABLE = True  # any game may be researched on demand, regardless of priority score


@dataclass(frozen=True)
class TriggerWeights:
    """Every weight is overridable per call - nothing here is hardcoded logic baked into
    the scoring function itself."""

    model_market_disagreement: float = 1.0
    line_movement: float = 1.0
    qb_uncertainty: float = 2.0
    injury_uncertainty: float = 1.5
    offensive_line_issue: float = 1.0
    coaching_roster_change: float = 0.5
    model_uncertainty: float = 1.0
    cross_model_disagreement: float = 1.0
    information_quality_concern: float = 0.5


DEFAULT_WEIGHTS = TriggerWeights()


@dataclass(frozen=True)
class TriggerInputs:
    model_market_disagreement_points: float | None = None
    line_movement_points: float | None = None
    qb_uncertainty_flag: bool = False
    injury_uncertainty_flag: bool = False
    offensive_line_issue_flag: bool = False
    coaching_roster_change_flag: bool = False
    model_uncertainty_score: float | None = None  # 0.0 (very confident) to 1.0 (maximally uncertain, e.g. win_prob near 0.5)
    cross_model_disagreement_points: float | None = None  # e.g. max(candidate margins) - min(candidate margins)
    information_quality_concern_flag: bool = False


@dataclass(frozen=True)
class TriggerResult:
    priority_score: float
    reasons: tuple[str, ...]
    inputs_used: tuple[str, ...]
    inputs_unavailable: tuple[str, ...]


def compute_research_priority(inputs: TriggerInputs, weights: TriggerWeights = DEFAULT_WEIGHTS) -> TriggerResult:
    score = 0.0
    reasons: list[str] = []
    used: list[str] = []
    unavailable: list[str] = []

    if inputs.model_market_disagreement_points is not None:
        score += abs(inputs.model_market_disagreement_points) * weights.model_market_disagreement
        used.append("model_market_disagreement")
        if abs(inputs.model_market_disagreement_points) >= 1.0:
            reasons.append(f"Model-market disagreement of {inputs.model_market_disagreement_points:+.2f} points")
    else:
        unavailable.append("model_market_disagreement")

    if inputs.line_movement_points is not None:
        score += abs(inputs.line_movement_points) * weights.line_movement
        used.append("line_movement")
        if abs(inputs.line_movement_points) >= 1.0:
            reasons.append(f"Line movement of {inputs.line_movement_points:+.2f} points")
    else:
        unavailable.append("line_movement")

    flag_fields = [
        ("qb_uncertainty_flag", weights.qb_uncertainty, "QB uncertainty flagged"),
        ("injury_uncertainty_flag", weights.injury_uncertainty, "Injury uncertainty flagged"),
        ("offensive_line_issue_flag", weights.offensive_line_issue, "Offensive-line issue flagged"),
        ("coaching_roster_change_flag", weights.coaching_roster_change, "Coaching/roster change flagged"),
        ("information_quality_concern_flag", weights.information_quality_concern, "Information-quality concern flagged"),
    ]
    for field_name, weight, reason in flag_fields:
        flag_value = getattr(inputs, field_name)
        used.append(field_name)
        if flag_value:
            score += weight
            reasons.append(reason)

    if inputs.model_uncertainty_score is not None:
        score += inputs.model_uncertainty_score * weights.model_uncertainty
        used.append("model_uncertainty")
        if inputs.model_uncertainty_score >= 0.7:
            reasons.append(f"High model uncertainty (score={inputs.model_uncertainty_score:.2f})")
    else:
        unavailable.append("model_uncertainty")

    if inputs.cross_model_disagreement_points is not None:
        score += abs(inputs.cross_model_disagreement_points) * weights.cross_model_disagreement
        used.append("cross_model_disagreement")
        if abs(inputs.cross_model_disagreement_points) >= 1.0:
            reasons.append(f"Elo/Ridge/LightGBM disagreement of {inputs.cross_model_disagreement_points:.2f} points")
    else:
        unavailable.append("cross_model_disagreement")

    return TriggerResult(
        priority_score=score, reasons=tuple(reasons), inputs_used=tuple(used), inputs_unavailable=tuple(unavailable)
    )
