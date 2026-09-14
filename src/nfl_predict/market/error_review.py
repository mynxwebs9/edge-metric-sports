"""Phase 5 Step 19: representative case review - football-level analysis of cases where
the model and market strongly disagreed (and who was right), and cases where they agreed
and both were wrong. Qualitative only: it never retroactively alters the frozen Phase 4
model or any stored prediction. Findings here are meant to motivate the future LLM research
layer (Phase 6), not to be acted on now.
"""

from __future__ import annotations

from dataclasses import dataclass

from nfl_predict.market.ats import ATSBetRecord


@dataclass(frozen=True)
class ReviewCase:
    category: str  # "A_model_disagreed_and_was_right" / "B_model_disagreed_and_market_was_right" / "C_agreed_and_both_wrong"
    game_id: str
    season: int
    week: int
    edge_points: float
    side: str
    grade: str
    actual_margin: float
    home_spread_traditional: float
    pbp_context: dict


def select_review_cases(bets: list[ATSBetRecord], n_per_category: int = 5, agreement_threshold: float = 0.5, big_miss_threshold: float = 14.0) -> dict[str, list[ATSBetRecord]]:
    """Pure selection logic (no I/O) - `pbp_context` is attached separately by the caller
    (see `attach_pbp_context`) so this function stays testable with plain synthetic bets."""
    disagreements = [b for b in bets if abs(b.edge_points) > agreement_threshold]
    agreements = [b for b in bets if abs(b.edge_points) <= agreement_threshold]

    def _model_won(b: ATSBetRecord) -> bool:
        return (b.side == "home" and b.grade == "home_covers") or (b.side == "away" and b.grade == "away_covers")

    case_a = sorted((b for b in disagreements if b.grade != "push" and _model_won(b)), key=lambda b: -abs(b.edge_points))[:n_per_category]
    case_b = sorted((b for b in disagreements if b.grade != "push" and not _model_won(b)), key=lambda b: -abs(b.edge_points))[:n_per_category]

    def _both_wrong(b: ATSBetRecord) -> bool:
        # "Both wrong": the game's actual margin missed the shared (agreed-upon) line by a
        # lot in EITHER direction - a big surprise nobody saw coming, not graded ATS at all
        # since there was no real disagreement to bet on.
        return abs(b.actual_margin - (-b.home_spread_traditional)) >= big_miss_threshold

    case_c = sorted((b for b in agreements if _both_wrong(b)), key=lambda b: -abs(b.actual_margin - (-b.home_spread_traditional)))[:n_per_category]

    return {
        "A_model_disagreed_and_was_right": case_a,
        "B_model_disagreed_and_market_was_right": case_b,
        "C_agreed_and_both_wrong": case_c,
    }


def build_review_cases(bets: list[ATSBetRecord], pbp_context_by_game_id: dict[str, dict], **kwargs) -> dict[str, list[ReviewCase]]:
    selected = select_review_cases(bets, **kwargs)
    out: dict[str, list[ReviewCase]] = {}
    for category, category_bets in selected.items():
        out[category] = [
            ReviewCase(
                category=category, game_id=b.game_id, season=b.season, week=b.week,
                edge_points=b.edge_points, side=b.side, grade=b.grade, actual_margin=b.actual_margin,
                home_spread_traditional=b.home_spread_traditional,
                pbp_context=pbp_context_by_game_id.get(b.game_id, {}),
            )
            for b in category_bets
        ]
    return out
