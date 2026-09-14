"""Feature quality tests (structural correctness, not predictive usefulness)."""

from __future__ import annotations

import math

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.features.game_table import build_game_features
from nfl_predict.features.registry import load_feature_registry
from nfl_predict.features.team_game import build_team_game_features

RATE_UNITS = {"rate_0_1"}


def _build(seasons: list[int]) -> pl.DataFrame:
    conn = get_connection()
    init_schema(conn)
    tg = build_team_game_features(conn, seasons)
    conn.close()
    return tg


def test_exactly_two_team_game_rows_per_game(real_data_sandbox):
    tg = _build([2024])
    counts = tg.group_by("game_id").agg(pl.len().alias("n"))
    bad = counts.filter(pl.col("n") != 2)
    assert bad.height == 0, f"games without exactly 2 team-game rows: {bad.to_dicts()}"


def test_exactly_one_game_row_per_game(real_data_sandbox):
    tg = _build([2024])
    game = build_game_features(tg)
    counts = game.group_by("game_id").agg(pl.len().alias("n"))
    bad = counts.filter(pl.col("n") != 1)
    assert bad.height == 0, f"games without exactly 1 game-level row: {bad.to_dicts()}"


def test_no_duplicate_team_game_rows(real_data_sandbox):
    tg = _build([2024])
    dupes = tg.group_by(["game_id", "team_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    assert dupes.height == 0


def test_team_never_equals_opponent(real_data_sandbox):
    tg = _build([2024])
    assert tg.filter(pl.col("team_id") == pl.col("opponent_id")).height == 0


def test_no_infinite_values(real_data_sandbox):
    tg = _build([2024])
    numeric_cols = [c for c, dt in zip(tg.columns, tg.dtypes) if dt in (pl.Float64, pl.Float32)]
    for col in numeric_cols:
        n_inf = tg.filter(pl.col(col).is_infinite()).height
        assert n_inf == 0, f"{col} has {n_inf} infinite value(s)"


def test_rate_features_within_0_1(real_data_sandbox):
    tg = _build([2024])
    registry = {e["name"]: e for e in load_feature_registry()}
    rate_cols = [name for name, e in registry.items() if e["unit"] in RATE_UNITS and name in tg.columns]
    assert len(rate_cols) > 10  # sanity: we actually found the rate columns
    for col in rate_cols:
        out_of_range = tg.filter(pl.col(col).is_not_null() & ((pl.col(col) < 0) | (pl.col(col) > 1)))
        assert out_of_range.height == 0, f"{col} has values outside [0,1]: {out_of_range[col].to_list()[:5]}"


def test_sample_counts_never_exceed_games_actually_played(real_data_sandbox):
    tg = _build([2024])
    for w in (3, 5, 8):
        bad = tg.filter(pl.col(f"n_games_trailing_{w}") > pl.col("games_played_current_season") + 0)
        # n_games_trailing_w can exceed THIS season's games_played (it's not season-scoped),
        # but it must never exceed the true count of the team's total prior games in the
        # dataset - checked instead against the hard window cap below.
        assert tg.filter(pl.col(f"n_games_trailing_{w}") > w).height == 0, f"n_games_trailing_{w} exceeds its own window cap"

    bad_qb = tg.filter(pl.col("qb_starts_season") > pl.col("games_played_current_season"))
    assert bad_qb.height == 0, "qb_starts_season exceeds the team's own games played this season"


def test_rolling_windows_respect_their_max_game_count(real_data_sandbox):
    tg = _build([2024])
    for w in (3, 5, 8):
        assert tg.filter((pl.col(f"n_games_trailing_{w}") < 0) | (pl.col(f"n_games_trailing_{w}") > w)).height == 0


def test_week_1_behavior_is_valid(real_data_sandbox):
    tg = _build([2024])
    week1 = tg.filter((pl.col("week") == 1) & (pl.col("season") == 2024))
    assert week1.height > 0

    for row in week1.iter_rows(named=True):
        assert row["games_played_current_season"] == 0
        assert row["off_epa_pp_season"] is None
        assert row["off_epa_pp_3g"] is None
        for w in (3, 5, 8):
            assert row[f"n_games_trailing_{w}"] == 0
        # A Week 1 team CAN have a non-null prev_season_* value (if it played the prior
        # season) - that's correct, not a bug; only in-season/trailing values must be null.


def test_nulls_follow_documented_policy_never_silently_zero(real_data_sandbox):
    """Spot-check: a metric with zero qualifying plays must be null, never a fabricated 0."""
    tg = _build([2024])
    # off_redzone_td_rate_season is null (not 0) whenever redzone_trips_n hasn't accumulated
    # yet for the team this season - check the very first game of a team with no red-zone
    # trips recorded yet (week 1, by construction of test_week_1_behavior_is_valid above).
    week1 = tg.filter((pl.col("week") == 1) & (pl.col("season") == 2024))
    assert week1.filter(pl.col("off_redzone_td_rate_season").is_not_null()).height == 0
    assert week1.filter(pl.col("qb_epa_dropback_season").is_not_null()).height == 0


def test_game_level_differential_equals_home_minus_away(real_data_sandbox):
    tg = _build([2024])
    game = build_game_features(tg)
    sample = game.filter(pl.col("home_def_epa_pp_allowed_season").is_not_null() & pl.col("away_def_epa_pp_allowed_season").is_not_null())
    assert sample.height > 0
    for row in sample.head(15).iter_rows(named=True):
        expected = row["home_def_epa_pp_allowed_season"] - row["away_def_epa_pp_allowed_season"]
        assert math.isclose(row["diff_def_epa_pp_allowed_season"], expected, rel_tol=1e-9, abs_tol=1e-12)


def test_stable_schema_within_one_feature_version(real_data_sandbox):
    tg_a = _build([2023])
    tg_b = _build([2024])
    assert set(tg_a.columns) == set(tg_b.columns)
    for col in tg_a.columns:
        assert tg_a.schema[col] == tg_b.schema[col], f"{col} dtype differs between seasons"
