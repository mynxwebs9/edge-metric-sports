"""Leakage-safe rolling windows over per-team-game atomic stats.

The core leakage guard lives here: for a team's game at position i in its own chronological
sequence, every rolling/season-to-date value uses ONLY games at positions < i. Implemented
as an explicit per-team sequential pass (not a vectorized polars window expression) because
correctness here is the single most important property of this entire phase - an off-by-one
in a vectorized rolling expression is exactly the kind of bug that silently leaks one game of
future information into every row, and a straightforward Python loop over each team's own
chronological list of games is easy to read, easy to test, and easy to prove correct by
inspection. Performance is not a concern at this data scale (tens of thousands of team-game
rows total).

Rates are computed from rolled SUMS and COUNTS, not from averaging per-game rates - see
pbp_aggregate.py's module docstring for why that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class MetricSpec:
    """Defines one rate metric to roll: `output_name` = sum(`sum_col`) / sum(`count_col`)
    over a window of prior games."""

    output_name: str
    sum_col: str
    count_col: str


def compute_rolling_and_season_to_date(
    team_game: pl.DataFrame,
    metrics: list[MetricSpec],
    windows: list[int],
    min_observations: int = 1,
) -> pl.DataFrame:
    """Adds, for each metric and each window in `windows`, a `{output_name}_{window}g`
    column (trailing N-game rate) plus `{output_name}_season` (season-to-date rate, reset
    per season). Also adds shared `n_games_trailing_{w}` and `games_played_current_season`
    sample-size columns (shared across all metrics computed over the same window, since
    they all draw from the identical set of prior games for a given team-game).

    `team_game` must already be sorted by (team_id, sort_ts) and contain every atomic
    sum/count column referenced by `metrics`, plus `team_id`, `season`, `sort_ts`.
    """
    all_sum_count_cols = sorted({c for m in metrics for c in (m.sum_col, m.count_col)})

    out_rows: list[dict] = []
    for _, group in team_game.sort(["team_id", "sort_ts"]).group_by("team_id", maintain_order=True):
        rows = group.to_dicts()
        # Prior-games history for THIS team only, in chronological order. history[j] is the
        # row that occurred strictly before rows[j+1] onward - never includes the current row.
        history: list[dict] = []
        current_season: int | None = None
        season_start_idx = 0  # index into `history` where the current season's games begin

        for row in rows:
            if row["season"] != current_season:
                current_season = row["season"]
                season_start_idx = len(history)

            games_played_current_season = len(history) - season_start_idx
            row["games_played_current_season"] = games_played_current_season

            for w in windows:
                window_games = history[-w:] if w > 0 else []
                row[f"n_games_trailing_{w}"] = len(window_games)
                for m in metrics:
                    row[f"{m.output_name}_{w}g"] = _rate(window_games, m, min_observations)

            season_games = history[season_start_idx:]
            for m in metrics:
                row[f"{m.output_name}_season"] = _rate(season_games, m, min_observations)

            out_rows.append(row)
            # Only the columns needed for future rolling are kept in history, to bound memory.
            history.append({c: row.get(c) for c in all_sum_count_cols})

    return pl.DataFrame(out_rows) if out_rows else team_game


def _rate(games: list[dict], metric: MetricSpec, min_observations: int) -> float | None:
    if len(games) < min_observations:
        return None
    total_count = sum((g.get(metric.count_col) or 0) for g in games)
    if total_count <= 0:
        return None
    total_sum = sum((g.get(metric.sum_col) or 0) for g in games)
    return total_sum / total_count
