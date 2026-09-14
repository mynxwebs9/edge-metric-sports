"""Normalized play-by-play storage.

Per the "do not write millions of play rows into SQLite" instruction and
docs/ARCHITECTURE.md#storage, play-by-play stays in Parquet, partitioned by season, under
`data/normalized/play_by_play/season=<year>/data.parquet`. Normalization here is limited to
standardizing team identifiers (adding resolved team_id columns alongside the existing
abbreviation columns) - no feature engineering, no derived stats. That starts in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings

NORMALIZED_DIRNAME = "normalized"
PBP_SUBDIR = "play_by_play"


def pbp_partition_path(season: int) -> Path:
    return get_settings().data_dir / NORMALIZED_DIRNAME / PBP_SUBDIR / f"season={season}" / "data.parquet"


def normalize_and_write_pbp(raw_pbp: pl.DataFrame, season: int, abbr_to_team_id: dict[str, str]) -> Path:
    team_cols = [c for c in ("posteam", "defteam", "home_team", "away_team") if c in raw_pbp.columns]
    df = raw_pbp
    for col in team_cols:
        df = df.with_columns(
            pl.col(col).replace_strict(abbr_to_team_id, default=None).alias(f"{col}_id")
        )

    path = pbp_partition_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_pbp_season(season: int) -> pl.DataFrame:
    # memory_map=False: polars memory-maps Parquet reads by default, which holds an open
    # mapped section on the file. On Windows this blocks a subsequent write to that same
    # path (e.g. a test overwriting it to prove a leakage guard) with
    # "OSError: ... a file with a user-mapped section open" - reading fully into memory
    # instead avoids that, at the cost of a full copy (acceptable; a season's pbp Parquet
    # is a few tens of MB).
    return pl.read_parquet(pbp_partition_path(season), memory_map=False)
