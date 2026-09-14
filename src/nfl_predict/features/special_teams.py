"""Restrained special-teams features (season-to-date only) - field goals and punting.

Deliberately small per the Phase 2 brief ("do not overbuild this area in V1"). Punt yardage
is GROSS distance (`kick_distance`), not net of return yardage - computing true net yards
would need to reconcile touchbacks/fair-catches/out-of-bounds/return-yards cases, which is
real added complexity for a metric this project isn't leaning on yet; documented here and
in docs/FEATURE_DICTIONARY.md as a known limitation rather than silently approximated.
Return performance is deferred entirely for the same reason.
"""

from __future__ import annotations

import polars as pl


def aggregate_special_teams(pbp: pl.DataFrame) -> pl.DataFrame:
    fg = (
        pbp.filter(pl.col("field_goal_attempt") == 1)
        .group_by(["game_id", "posteam_id"])
        .agg(
            pl.len().alias("fg_attempt_n"),
            (pl.col("field_goal_result") == "made").sum().alias("fg_made_n"),
        )
        .rename({"posteam_id": "team_id"})
    )
    punt = (
        pbp.filter(pl.col("play_type") == "punt")
        .group_by(["game_id", "posteam_id"])
        .agg(
            pl.col("kick_distance").filter(pl.col("kick_distance").is_not_null()).sum().alias("punt_yards_sum"),
            pl.col("kick_distance").is_not_null().sum().alias("punt_n"),
        )
        .rename({"posteam_id": "team_id"})
    )
    return fg.join(punt, on=["game_id", "team_id"], how="full", coalesce=True)
