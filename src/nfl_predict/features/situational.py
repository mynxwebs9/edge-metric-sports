"""Pregame situational context: rest, home field, neutral site, short week, bye, division.

Every value here is either static game metadata (known the moment the schedule is
published, long before kickoff) or derived from it - none of it depends on any other team's
performance or any play data, so there is no rolling-window leakage concern for this
module. The one thing to get right is not accidentally reading anything about the game's
OUTCOME (this module never touches scores).
"""

from __future__ import annotations

import sqlite3

import polars as pl

from nfl_predict.config import get_feature_engine_config
from nfl_predict.data.raw_store import list_manifests, read_raw_snapshot


def _load_neutral_site_map(seasons: list[int]) -> dict[str, bool]:
    """`location` ('Home' | 'Neutral') isn't part of the Phase 1 normalized `games` schema
    (deliberately kept minimal - see docs/PHASE1_DATA_REPORT.md) but it's untouched-preserved
    in the immutable raw `schedules` snapshots, which is exactly what raw snapshots are for.
    Reads it from there rather than expanding Phase 1's already-completed normalized table."""
    result: dict[str, bool] = {}
    manifests = [m for m in list_manifests("nflverse", "schedules") if m.duplicate_of_retrieval_id is None]
    for m in manifests:
        if len(m.requested_seasons) != 1 or m.requested_seasons[0] not in seasons:
            continue
        raw = read_raw_snapshot(m)
        if "location" not in raw.columns:
            continue
        for row in raw.select(["game_id", "location"]).iter_rows(named=True):
            result[row["game_id"]] = (row["location"] == "Neutral")
    return result


def add_situational_features(team_game: pl.DataFrame, conn: sqlite3.Connection, seasons: list[int]) -> pl.DataFrame:
    cfg = get_feature_engine_config()
    short_week_max = cfg["short_week_max_rest_days"]
    post_bye_min = cfg["post_bye_min_rest_days"]

    division_rows = conn.execute("SELECT team_id, division FROM teams").fetchall()
    team_division = {r["team_id"]: r["division"] for r in division_rows}
    neutral_site = _load_neutral_site_map(seasons)

    out = team_game.with_columns(
        pl.col("team_id").replace_strict(team_division, default=None).alias("_team_division"),
        pl.col("opponent_id").replace_strict(team_division, default=None).alias("_opp_division"),
        pl.col("game_id").replace_strict(neutral_site, default=False).alias("is_neutral_site"),
    )

    out = out.with_columns(
        (
            (pl.col("_team_division") == pl.col("_opp_division"))
            & pl.col("_team_division").is_not_null()
        ).alias("is_divisional_game"),
        pl.when(pl.col("days_rest").is_not_null() & (pl.col("days_rest") <= short_week_max))
        .then(True).otherwise(False).alias("short_week"),
        pl.when(pl.col("days_rest").is_not_null() & (pl.col("days_rest") >= post_bye_min))
        .then(True).otherwise(False).alias("post_bye"),
    ).drop(["_team_division", "_opp_division"])

    return out


def add_rest_difference(team_game: pl.DataFrame) -> pl.DataFrame:
    """`opponent_days_rest`/`rest_diff` require looking up the OPPONENT's own `days_rest`
    for the same game_id - a same-game lookup (both teams' rest is public schedule
    information, known long before kickoff), not a cross-game leak."""
    rest_by_game_team = team_game.select(["game_id", "team_id", "days_rest"]).rename({"team_id": "opponent_id", "days_rest": "opponent_days_rest"})
    out = team_game.join(rest_by_game_team, left_on=["game_id", "opponent_id"], right_on=["game_id", "opponent_id"], how="left")
    return out.with_columns(
        (pl.col("days_rest") - pl.col("opponent_days_rest")).alias("rest_diff")
    )
