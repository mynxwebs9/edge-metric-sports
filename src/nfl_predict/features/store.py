"""Parquet storage for generated feature tables, partitioned by feature_version/season.

Layout:
    data/features/team_game/feature_version=<v>/season=<year>/data.parquet
    data/features/game/feature_version=<v>/season=<year>/data.parquet
    data/features/team_game/feature_version=<v>/season=<year>/manifest.json
    data/features/game/feature_version=<v>/season=<year>/manifest.json

Large analytical data stays in Parquet, never SQLite, per docs/ARCHITECTURE.md#storage - a
feature version's full historical table can be tens of thousands of rows x 100+ columns,
squarely in "large analytical dataset" territory.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings
from nfl_predict.features.provenance import FeatureBuildManifest

FEATURES_DIRNAME = "features"


def partition_dir(table: str, feature_version: str, season: int) -> Path:
    return get_settings().data_dir / FEATURES_DIRNAME / table / f"feature_version={feature_version}" / f"season={season}"


def write_feature_partition(df: pl.DataFrame, table: str, feature_version: str, season: int, manifest: FeatureBuildManifest) -> Path:
    directory = partition_dir(table, feature_version, season)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "data.parquet"
    df.write_parquet(data_path)
    (directory / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return data_path


def read_feature_partition(table: str, feature_version: str, season: int) -> pl.DataFrame:
    # memory_map=False: see pbp_store.read_pbp_season for why - this is the most heavily
    # reused read path in the test suite (every Phase 3 model test loads game-level
    # features through read_feature_table, below), so leaving it memory-mapped is exactly
    # the kind of thing that would eventually collide with a rewrite of the same partition.
    return pl.read_parquet(partition_dir(table, feature_version, season) / "data.parquet", memory_map=False)


def read_feature_table(table: str, feature_version: str, seasons: list[int]) -> pl.DataFrame:
    frames = []
    for season in seasons:
        path = partition_dir(table, feature_version, season) / "data.parquet"
        if path.is_file():
            frames.append(pl.read_parquet(path, memory_map=False))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")
