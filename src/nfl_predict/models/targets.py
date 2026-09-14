"""Versioned modeling-target dataset.

Targets are derived from the normalized `games` table (Phase 1) and stored SEPARATELY from
the Phase 2 feature Parquet files - never written back into
`data/features/{team_game,game}/`. This is what keeps "features" and "results" physically
distinct, not just conceptually distinct (CLAUDE.md principle: targets/results must remain
separate from feature storage).

Storage: data/targets/target_version=<v>/season=<year>/data.parquet
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings

TARGET_VERSION = "v1"
TARGETS_DIRNAME = "targets"

TARGET_COLUMNS = [
    "game_id", "season", "week", "season_type", "kickoff_time_naive", "game_date",
    "home_team_id", "away_team_id",
    "home_score", "away_score", "home_margin", "total_points", "is_tie", "home_win",
]


def build_targets(conn: sqlite3.Connection, seasons: list[int]) -> pl.DataFrame:
    """One row per completed game. `home_win` is null for tied games (a tie is not a win or
    a loss for either side) - see `is_tie` to distinguish "tied" from "not yet computed".
    Ties are rare (13 in the 2010-2025 window, all regular season) but must be handled
    explicitly rather than silently coerced into a win or a loss for either team.
    """
    placeholders = ",".join("?" for _ in seasons)
    rows = conn.execute(
        f"""SELECT game_id, season, week, season_type, kickoff_time_naive, game_date,
                   home_team_id, away_team_id, home_score, away_score
            FROM games WHERE season IN ({placeholders}) AND game_status = 'final'
            ORDER BY season, week, game_id""",
        seasons,
    ).fetchall()

    records = []
    for r in rows:
        home_score, away_score = r["home_score"], r["away_score"]
        is_tie = home_score == away_score
        records.append({
            "game_id": r["game_id"], "season": r["season"], "week": r["week"],
            "season_type": r["season_type"], "kickoff_time_naive": r["kickoff_time_naive"],
            "game_date": r["game_date"],
            "home_team_id": r["home_team_id"], "away_team_id": r["away_team_id"],
            "home_score": home_score, "away_score": away_score,
            "home_margin": home_score - away_score,
            "total_points": home_score + away_score,
            "is_tie": is_tie,
            "home_win": None if is_tie else int(home_score > away_score),
        })

    if not records:
        return pl.DataFrame(schema={c: pl.Utf8 for c in TARGET_COLUMNS})
    return pl.DataFrame(records).select(TARGET_COLUMNS)


def targets_partition_path(season: int, target_version: str = TARGET_VERSION) -> Path:
    return get_settings().data_dir / TARGETS_DIRNAME / f"target_version={target_version}" / f"season={season}" / "data.parquet"


def write_targets(conn: sqlite3.Connection, seasons: list[int], target_version: str = TARGET_VERSION) -> None:
    targets = build_targets(conn, seasons)
    for season in seasons:
        season_targets = targets.filter(pl.col("season") == season)
        if season_targets.height == 0:
            continue
        path = targets_partition_path(season, target_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        season_targets.write_parquet(path)


def read_targets(seasons: list[int], target_version: str = TARGET_VERSION) -> pl.DataFrame:
    frames = []
    for season in seasons:
        path = targets_partition_path(season, target_version)
        if path.is_file():
            frames.append(pl.read_parquet(path, memory_map=False))
    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in TARGET_COLUMNS})
    return pl.concat(frames, how="diagonal_relaxed")
