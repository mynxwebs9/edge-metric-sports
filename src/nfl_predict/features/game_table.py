"""Joins home + away team-game feature rows into one row per game.

Per the Phase 2 brief: preserve both home and away values (never discard the underlying
team-level numbers), and additionally compute `diff_*` columns for numeric features where a
symmetric difference is meaningful (home minus away) - purely a convenience transform of
two already-leakage-safe values, so it introduces no new leakage risk itself.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.features.registry import assert_no_denylisted_columns

# Columns that describe identity/context rather than a comparable numeric measurement -
# never diffed, and not duplicated with a home_/away_ prefix beyond what's naturally
# useful (e.g. the game only needs one `season`, one `week`, not home_season/away_season).
_SHARED_GAME_COLUMNS = {
    "game_id", "season", "season_type", "week", "game_date", "kickoff_time_naive", "as_of_timestamp", "venue",
}
_NON_DIFFABLE_TEAM_COLUMNS = {
    "team_id", "opponent_id", "is_home", "is_neutral_site", "is_divisional_game", "qb_primary_id",
}


def build_game_features(team_game: pl.DataFrame) -> pl.DataFrame:
    home = team_game.filter(pl.col("is_home")).rename(
        {c: f"home_{c}" for c in team_game.columns if c not in _SHARED_GAME_COLUMNS}
    )
    away = team_game.filter(~pl.col("is_home")).rename(
        {c: f"away_{c}" for c in team_game.columns if c not in _SHARED_GAME_COLUMNS}
    )

    game = home.join(away, on=list(_SHARED_GAME_COLUMNS), how="inner")

    numeric_team_cols = [
        c for c in team_game.columns
        if c not in _SHARED_GAME_COLUMNS and c not in _NON_DIFFABLE_TEAM_COLUMNS
        and team_game.schema[c] in (pl.Float64, pl.Int64, pl.Int32, pl.Boolean)
    ]
    diff_exprs = [
        (pl.col(f"home_{c}").cast(pl.Float64) - pl.col(f"away_{c}").cast(pl.Float64)).alias(f"diff_{c}")
        for c in numeric_team_cols
    ]
    result = game.with_columns(*diff_exprs) if diff_exprs else game
    assert_no_denylisted_columns(result.columns)
    return result
