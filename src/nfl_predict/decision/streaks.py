"""Phase 7 Step 13/14: truthful, predefined-window streak/marketing summaries, computed
algorithmically from the chronological official ledger. Only a fixed, predeclared set of
windows exists - `last_5`/`last_10`/`last_20`/`last_30`/`season_to_date`/`current_streak` -
there is no code path that accepts an arbitrary start date, so a headline like "17-3 since
August 29" cannot be produced by this module at all.

Pushes never break a win/loss streak (Step 24 proof #29) - they are excluded from the
decided sequence a streak is computed over, but still counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_predict.market.odds_math import moneyline_profit_units

PREDEFINED_WINDOWS = ("last_5", "last_10", "last_20", "last_30", "season_to_date", "current_streak")
MIN_SETTLED_FOR_HEADLINE = 5
HEADLINE_CATEGORY = "BEST_BETS"  # Step 14: a "hot streak" headline only ever describes this category


@dataclass(frozen=True)
class StreakSummary:
    window: str
    category: str
    n: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    total_units: float | None
    headline_eligible: bool
    headline: str | None


class UnknownWindowError(Exception):
    """Raised for any window name outside `PREDEFINED_WINDOWS` - there is no way to ask
    this module for an arbitrary/cherry-picked date range."""


def _settled_chronological(picks: list[dict], category: str) -> list[dict]:
    settled = [p for p in picks if p["category"] == category and p["status"] == "SETTLED"]
    return sorted(settled, key=lambda p: p["settled_at"])


def _summarize(subset: list[dict], window: str, category: str) -> StreakSummary:
    wins = sum(1 for p in subset if p["settlement"] == "WIN")
    losses = sum(1 for p in subset if p["settlement"] == "LOSS")
    pushes = sum(1 for p in subset if p["settlement"] == "PUSH")
    n_decided = wins + losses
    win_rate = wins / n_decided if n_decided else None
    total_units = float(sum(moneyline_profit_units(p["price"], p["settlement"] == "WIN") if p["settlement"] != "PUSH" else 0.0 for p in subset)) if subset else None

    eligible = category == HEADLINE_CATEGORY and len(subset) >= MIN_SETTLED_FOR_HEADLINE
    headline = None
    if eligible:
        window_label = {
            "last_5": "Last 5", "last_10": "Last 10", "last_20": "Last 20", "last_30": "Last 30",
            "season_to_date": "Season", "current_streak": "Current streak",
        }[window]
        if window == "current_streak":
            headline = f"{'Won' if subset[-1]['settlement'] == 'WIN' else 'Lost'} {len(subset)} straight {HEADLINE_CATEGORY.replace('_', ' ').title()}"
        else:
            headline = f"{wins}-{losses}" + (f"-{pushes}" if pushes else "") + f" {window_label} {HEADLINE_CATEGORY.replace('_', ' ').title()}"
            if total_units is not None:
                headline += f" ({total_units:+.1f} units)"

    return StreakSummary(window=window, category=category, n=len(subset), wins=wins, losses=losses, pushes=pushes, win_rate=win_rate, total_units=total_units, headline_eligible=eligible, headline=headline)


def compute_window(picks: list[dict], category: str, window: str, season: int | None = None) -> StreakSummary:
    if window not in PREDEFINED_WINDOWS:
        raise UnknownWindowError(f"window={window!r} is not one of the predefined windows: {PREDEFINED_WINDOWS}")

    chronological = _settled_chronological(picks, category)

    if window == "current_streak":
        decided = [p for p in chronological if p["settlement"] != "PUSH"]
        if not decided:
            return _summarize([], window, category)
        last_result = decided[-1]["settlement"]
        streak: list[dict] = []
        for p in reversed(decided):
            if p["settlement"] == last_result:
                streak.append(p)
            else:
                break
        return _summarize(list(reversed(streak)), window, category)

    if window == "season_to_date":
        if season is None:
            raise ValueError("season_to_date requires a 'season' argument")
        subset = [p for p in chronological if str(season) in str(p.get("kickoff_at", ""))]
        return _summarize(subset, window, category)

    n_window = int(window.split("_")[1])
    subset = chronological[-n_window:]
    return _summarize(subset, window, category)


def compute_all_predefined_windows(picks: list[dict], category: str, season: int | None = None) -> dict[str, StreakSummary]:
    result = {}
    for window in PREDEFINED_WINDOWS:
        if window == "season_to_date" and season is None:
            continue
        result[window] = compute_window(picks, category, window, season=season)
    return result
