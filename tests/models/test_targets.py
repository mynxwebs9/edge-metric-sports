"""Target dataset: tie handling, and physical separation from feature Parquet storage."""

from __future__ import annotations

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.models.targets import TARGET_COLUMNS, build_targets
from nfl_predict.models.split import DEVELOPMENT_SEASONS


def test_tied_games_are_explicit_not_coerced():
    conn = get_connection()
    init_schema(conn)
    targets = build_targets(conn, [2016])  # 2016 contains a real tie (SEA-ARI, week 7)
    conn.close()

    tied = targets.filter(pl.col("is_tie"))
    assert tied.height >= 1
    for row in tied.iter_rows(named=True):
        assert row["home_margin"] == 0
        assert row["home_win"] is None  # neither a win nor a loss - never coerced to 0 or 1


def test_non_tied_games_have_a_definite_home_win_value():
    conn = get_connection()
    init_schema(conn)
    targets = build_targets(conn, [2022])
    conn.close()

    non_tied = targets.filter(~pl.col("is_tie"))
    assert non_tied.filter(pl.col("home_win").is_null()).height == 0
    for row in non_tied.iter_rows(named=True):
        expected = int(row["home_score"] > row["away_score"])
        assert row["home_win"] == expected


def test_margin_and_total_are_computed_correctly():
    conn = get_connection()
    init_schema(conn)
    targets = build_targets(conn, [2022])
    conn.close()

    for row in targets.iter_rows(named=True):
        assert row["home_margin"] == row["home_score"] - row["away_score"]
        assert row["total_points"] == row["home_score"] + row["away_score"]


def test_target_columns_are_never_present_in_the_feature_parquet():
    """CLAUDE.md / Phase 3 brief: outcome fields must never be written back into the
    Phase 2 feature Parquet files. Checked directly against the actual stored file."""
    from nfl_predict.features.store import read_feature_table

    game_features = read_feature_table("game", "v1", [2022])
    outcome_fields = {"home_score", "away_score", "home_margin", "total_points", "home_win", "is_tie"}
    assert not (outcome_fields & set(game_features.columns))


def test_targets_are_metadata_complete():
    conn = get_connection()
    init_schema(conn)
    targets = build_targets(conn, [2022])
    conn.close()
    assert set(targets.columns) == set(TARGET_COLUMNS)
    assert targets.filter(pl.col("game_id").is_null()).height == 0
    assert targets.filter(pl.col("season_type").is_null()).height == 0


def test_development_targets_exist_on_disk():
    from nfl_predict.models.targets import read_targets

    targets = read_targets(DEVELOPMENT_SEASONS)
    assert targets.height > 0
    assert set(targets["season"].unique().to_list()) == set(DEVELOPMENT_SEASONS)
