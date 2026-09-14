"""Phase 7 Step 12/15: price-aware record calculations, kept strictly separate per
category (`ALL_MODEL_PREDICTIONS` / `LEANS` / `BEST_BETS`) - never combined after the fact,
never used to retroactively promote a winning LEAN or demote a losing BEST_BET (those are
identity fields on the published pick itself, fixed at publish time - see `pick_ledger.py`).

Official units/ROI use ONLY the actual recorded price on each published pick
(`PublishedPick.price` is a required field - a pick is never published without one, so the
official record never needs to silently assume a price). A separate, explicitly-labeled
"hypothetical assumed-price" calculation exists for reference and must never be presented as
the official record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_predict.market.odds_math import moneyline_profit_units

ASSUMED_HYPOTHETICAL_PRICE = -110


@dataclass(frozen=True)
class CategoryRecord:
    category: str
    n_settled: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    total_units: float | None
    roi_per_bet: float | None
    average_price: float | None
    is_hypothetical: bool = False


def _settled_picks(picks: list[dict], category: str) -> list[dict]:
    return [p for p in picks if p["category"] == category and p["status"] == "SETTLED"]


def _profit_for_pick(pick: dict, price: int) -> float:
    if pick["settlement"] == "PUSH":
        return 0.0
    return moneyline_profit_units(price, pick["settlement"] == "WIN")


def compute_category_record(picks: list[dict], category: str) -> CategoryRecord:
    """The OFFICIAL, price-aware record - uses each pick's own real recorded price."""
    category_picks = _settled_picks(picks, category)
    wins = sum(1 for p in category_picks if p["settlement"] == "WIN")
    losses = sum(1 for p in category_picks if p["settlement"] == "LOSS")
    pushes = sum(1 for p in category_picks if p["settlement"] == "PUSH")
    n_decided = wins + losses
    win_rate = wins / n_decided if n_decided else None

    if not category_picks:
        return CategoryRecord(category=category, n_settled=0, wins=0, losses=0, pushes=0, win_rate=None, total_units=None, roi_per_bet=None, average_price=None)

    profits = [_profit_for_pick(p, p["price"]) for p in category_picks]
    return CategoryRecord(
        category=category, n_settled=len(category_picks), wins=wins, losses=losses, pushes=pushes,
        win_rate=win_rate, total_units=float(sum(profits)), roi_per_bet=float(np.mean(profits)),
        average_price=float(np.mean([p["price"] for p in category_picks])),
    )


def compute_hypothetical_assumed_price_record(picks: list[dict], category: str, assumed_price: int = ASSUMED_HYPOTHETICAL_PRICE) -> CategoryRecord:
    """A SEPARATE, explicitly-labeled hypothetical record using `assumed_price` for every
    pick instead of its real recorded price - never to be presented as, or blended with,
    `compute_category_record`'s official output (Step 15)."""
    category_picks = _settled_picks(picks, category)
    wins = sum(1 for p in category_picks if p["settlement"] == "WIN")
    losses = sum(1 for p in category_picks if p["settlement"] == "LOSS")
    pushes = sum(1 for p in category_picks if p["settlement"] == "PUSH")
    n_decided = wins + losses
    win_rate = wins / n_decided if n_decided else None

    if not category_picks:
        return CategoryRecord(category=category, n_settled=0, wins=0, losses=0, pushes=0, win_rate=None, total_units=None, roi_per_bet=None, average_price=None, is_hypothetical=True)

    profits = [_profit_for_pick(p, assumed_price) for p in category_picks]
    return CategoryRecord(
        category=category, n_settled=len(category_picks), wins=wins, losses=losses, pushes=pushes,
        win_rate=win_rate, total_units=float(sum(profits)), roi_per_bet=float(np.mean(profits)),
        average_price=float(assumed_price), is_hypothetical=True,
    )
