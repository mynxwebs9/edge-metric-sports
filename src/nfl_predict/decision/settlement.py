"""Phase 7 Step 16: deterministic settlement for spread and moneyline picks. Totals are
NOT supported (Phase 4/5 showed insufficient independent totals signal - see
docs/PHASE4_BACKTEST_REPORT.md/docs/PHASE5_MARKET_REPORT.md's totals conclusions).
Settlement always uses the immutable published line/price - see `pick_ledger.py`.
"""

from __future__ import annotations

from nfl_predict.decision.pick_ledger import SettlementResult
from nfl_predict.market.odds_math import grade_ats

SUPPORTED_MARKET_TYPES = ("spread", "moneyline")


class UnsupportedMarketTypeError(Exception):
    pass


def settle_spread_pick(selection: str, line: float, actual_home_margin: float) -> SettlementResult:
    """`selection`: "home" or "away". `line`: the HOME team's traditional-sign spread as
    published. Reuses `nfl_predict.market.odds_math.grade_ats` for the actual grading logic
    - the same sign convention and push handling Phase 5 already validated."""
    grade = grade_ats(actual_home_margin, line)
    if grade == "push":
        return SettlementResult.PUSH
    won = (selection == "home" and grade == "home_covers") or (selection == "away" and grade == "away_covers")
    return SettlementResult.WIN if won else SettlementResult.LOSS


def settle_moneyline_pick(selection: str, actual_home_win: bool | None) -> SettlementResult:
    """`actual_home_win`: True/False, or None for a tied game - graded PUSH, the standard
    real-world sportsbook settlement rule for an NFL tie (same convention
    `nfl_predict.market.moneyline_edge` already documents and uses)."""
    if actual_home_win is None:
        return SettlementResult.PUSH
    won = (selection == "home" and actual_home_win) or (selection == "away" and not actual_home_win)
    return SettlementResult.WIN if won else SettlementResult.LOSS


def settle_pick(
    market_type: str, selection: str, line: float | None = None,
    actual_home_margin: float | None = None, actual_home_win: bool | None = None,
) -> SettlementResult:
    if market_type == "spread":
        if line is None or actual_home_margin is None:
            raise ValueError("Spread settlement requires both 'line' and 'actual_home_margin'")
        return settle_spread_pick(selection, line, actual_home_margin)
    if market_type == "moneyline":
        return settle_moneyline_pick(selection, actual_home_win)
    raise UnsupportedMarketTypeError(f"market_type={market_type!r} not supported - only {SUPPORTED_MARKET_TYPES} (no totals yet)")
