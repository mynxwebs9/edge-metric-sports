"""Phase 5 Step 11: closing-line value (CLV).

CLV requires at least TWO timestamped market snapshots per game (an entry point and a
closing point). The historical `market_snapshot` layer built in Steps 1-2 has exactly ONE
(`snapshot_type=UNKNOWN`) row per game - CLV literally cannot be computed from it, and this
module refuses to fabricate a second snapshot to make the math "work" (per the brief's
explicit instruction). It is implemented and unit-tested here against synthetic
multi-snapshot data so it is ready the moment Steps 21-22's live odds storage starts
accumulating real multiple snapshots per game.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.market.odds_math import moneyline_clv_probability, no_vig_two_way, spread_clv_points


class InsufficientSnapshotsError(Exception):
    """Raised when CLV is requested for a game with fewer than 2 market snapshots."""


def clv_available_for_historical_data() -> bool:
    """Always False for the Step 1-2 nflverse-sourced historical layer - exactly one
    UNKNOWN-timing snapshot per game, so there is no documented "closing" snapshot to
    compare against. Becomes meaningful only once live multi-snapshot storage (Step 22) has
    accumulated real data for a game."""
    return False


def compute_spread_clv(snapshots_for_game: pl.DataFrame, bet_time_row_index: int, side: str) -> float:
    """`snapshots_for_game`: every known snapshot for ONE game, each row having at least
    `snapshot_timestamp` and `home_spread_traditional`. The LAST snapshot after sorting by
    timestamp is treated as closing. Raises `InsufficientSnapshotsError` on fewer than 2
    rows - never silently returns a fabricated 0.0."""
    if snapshots_for_game.height < 2:
        raise InsufficientSnapshotsError(f"Need >= 2 market snapshots to compute CLV, got {snapshots_for_game.height}")
    ordered = snapshots_for_game.sort("snapshot_timestamp")
    bet_row = ordered.row(bet_time_row_index, named=True)
    closing_row = ordered.row(-1, named=True)
    return spread_clv_points(bet_row["home_spread_traditional"], closing_row["home_spread_traditional"], side)


def compute_moneyline_clv(snapshots_for_game: pl.DataFrame, bet_time_row_index: int, side: str) -> float:
    if snapshots_for_game.height < 2:
        raise InsufficientSnapshotsError(f"Need >= 2 market snapshots to compute CLV, got {snapshots_for_game.height}")
    ordered = snapshots_for_game.sort("snapshot_timestamp")
    bet_row = ordered.row(bet_time_row_index, named=True)
    closing_row = ordered.row(-1, named=True)

    bet_nv = no_vig_two_way(bet_row["home_moneyline"], bet_row["away_moneyline"])
    close_nv = no_vig_two_way(closing_row["home_moneyline"], closing_row["away_moneyline"])
    bet_prob = bet_nv.no_vig_prob_a if side == "home" else bet_nv.no_vig_prob_b
    close_prob = close_nv.no_vig_prob_a if side == "home" else close_nv.no_vig_prob_b
    return moneyline_clv_probability(bet_prob, close_prob)


def proportion_beating_closing_price(clv_values: list[float]) -> float | None:
    """Fraction of positions with favorable (>0) CLV - Step 11's "proportion of positions
    beating closing price" metric. None (not 0.0) on an empty input - an empty sample is
    "no data," not "0% beat the close.\""""
    if not clv_values:
        return None
    return sum(1 for v in clv_values if v > 0) / len(clv_values)
