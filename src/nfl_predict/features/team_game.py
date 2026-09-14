"""Assembles the team-game pregame feature table.

This is the orchestrator: game backbone -> atomic per-game stats (from pbp) -> rolling
windows -> QB context -> personnel continuity -> situational context -> opponent quality
-> previous-season summary. Order matters for a few steps (opponent quality needs the
team-level season-to-date columns to already exist; previous-season needs the full table
first) but nothing here computes anything about a game using that same game's own outcome
or any other team's future games - see docs/PHASE2_FEATURE_REPORT.md for the full leakage
argument, and tests/features/test_leakage.py for the automated proof.
"""

from __future__ import annotations

import sqlite3

import polars as pl

from nfl_predict.config import get_feature_engine_config
from nfl_predict.data.nflverse_loader import SOURCE_NAME
from nfl_predict.data.pbp_store import read_pbp_season
from nfl_predict.data.repositories import TeamsRepository
from nfl_predict.features.game_index import add_rest_days, build_team_game_index
from nfl_predict.features.opponent_context import add_opponent_quality_faced
from nfl_predict.features.pbp_aggregate import aggregate_team_game_pbp
from nfl_predict.features.personnel import add_snap_continuity, load_snap_counts_for_seasons
from nfl_predict.features.prev_season import add_previous_season_summary
from nfl_predict.features.qb import build_qb_atomic, build_qb_features, identify_primary_qb
from nfl_predict.features.registry import assert_no_denylisted_columns
from nfl_predict.features.rolling import MetricSpec, compute_rolling_and_season_to_date
from nfl_predict.features.situational import add_rest_difference, add_situational_features
from nfl_predict.features.special_teams import aggregate_special_teams

# 14 metrics rolled over trailing 3/5/8 games AND season-to-date (56 columns total, plus
# shared sample-size counters). See docs/PHASE2_FEATURE_REPORT.md for the full formula
# table - this list is the authoritative source, mirrored (not re-derived) in
# config/features.yaml.
WINDOWED_METRICS = [
    MetricSpec("off_epa_pp", "off_epa_sum", "off_plays_n"),
    MetricSpec("off_success_rate", "off_success_sum", "off_plays_n"),
    MetricSpec("off_pass_epa_dropback", "off_pass_epa_sum", "off_dropback_n"),
    MetricSpec("off_pass_success_rate", "off_pass_success_sum", "off_dropback_n"),
    MetricSpec("off_rush_epa", "off_rush_epa_sum", "off_rush_n"),
    MetricSpec("off_rush_success_rate", "off_rush_success_sum", "off_rush_n"),
    MetricSpec("def_epa_pp_allowed", "def_epa_sum", "def_plays_n"),
    MetricSpec("def_success_rate_allowed", "def_success_sum", "def_plays_n"),
    MetricSpec("def_pass_epa_allowed", "def_pass_epa_sum", "def_dropback_n"),
    MetricSpec("def_pass_success_rate_allowed", "def_pass_success_sum", "def_dropback_n"),
    MetricSpec("def_rush_epa_allowed", "def_rush_epa_sum", "def_rush_n"),
    MetricSpec("def_rush_success_rate_allowed", "def_rush_success_sum", "def_rush_n"),
    MetricSpec("off_epa_pp_comp", "off_epa_sum_comp", "off_plays_n_comp"),
    MetricSpec("off_success_rate_comp", "off_success_sum_comp", "off_plays_n_comp"),
]

# 23 metrics, season-to-date ONLY (no 3g/5g/8g variant) - situational/specialized stats
# where a single trailing window adds less value relative to the column-count cost. See
# docs/PHASE2_FEATURE_REPORT.md's "Feature windows" section for the rationale.
SEASON_ONLY_METRICS = [
    MetricSpec("off_explosive_pass_rate", "off_explosive_pass_n", "off_pass_attempt_n"),
    MetricSpec("off_explosive_rush_rate", "off_explosive_rush_n", "off_rush_n"),
    MetricSpec("off_sack_rate_allowed", "off_sack_n", "off_pass_attempt_n"),
    MetricSpec("off_qb_hit_rate_allowed", "off_qb_hit_n", "off_pass_attempt_n"),
    MetricSpec("off_cpoe", "off_cpoe_sum", "off_cpoe_n"),
    MetricSpec("off_early_down_epa", "off_early_down_epa_sum", "off_early_down_n"),
    MetricSpec("off_early_down_success_rate", "off_early_down_success_sum", "off_early_down_n"),
    MetricSpec("off_third_down_conv_rate", "off_third_down_conv_n", "off_third_down_att_n"),
    MetricSpec("off_redzone_td_rate", "redzone_td_n", "redzone_trips_n"),
    MetricSpec("off_yards_per_dropback", "off_dropback_yards_sum", "off_dropback_n"),
    MetricSpec("off_yards_per_rush", "off_rush_yards_sum", "off_rush_n"),
    MetricSpec("off_int_rate", "off_int_n", "off_pass_attempt_n"),
    MetricSpec("off_fumble_lost_rate", "off_fumble_lost_n", "off_plays_n"),
    MetricSpec("def_explosive_pass_rate_allowed", "def_explosive_pass_n", "def_pass_attempt_n"),
    MetricSpec("def_explosive_rush_rate_allowed", "def_explosive_rush_n", "def_rush_n"),
    MetricSpec("def_sack_rate_generated", "def_sack_n", "def_pass_attempt_n"),
    MetricSpec("def_qb_hit_rate_generated", "def_qb_hit_n", "def_pass_attempt_n"),
    MetricSpec("def_early_down_epa_allowed", "def_early_down_epa_sum", "def_early_down_n"),
    MetricSpec("def_early_down_success_rate_allowed", "def_early_down_success_sum", "def_early_down_n"),
    MetricSpec("def_third_down_allowed", "def_third_down_conv_n", "def_third_down_att_n"),
    MetricSpec("def_redzone_td_rate_allowed", "redzone_td_n_allowed", "redzone_trips_n_allowed"),
    MetricSpec("def_int_rate_generated", "def_int_n", "def_pass_attempt_n"),
    MetricSpec("def_fumble_forced_rate", "def_fumble_lost_n", "def_plays_n"),
]

SPECIAL_TEAMS_METRICS = [
    MetricSpec("fg_pct", "fg_made_n", "fg_attempt_n"),
    MetricSpec("punt_yards_avg", "punt_yards_sum", "punt_n"),
]

# Identity/context columns carried into the public feature table verbatim (never a result).
IDENTITY_COLUMNS = [
    "game_id", "season", "season_type", "week", "game_date", "kickoff_time_naive",
    "as_of_timestamp", "team_id", "opponent_id", "is_home", "venue",
]

# Sample-size / missingness-indicator columns exposed alongside the rate features so a
# downstream model (or a human) can tell how much data backs a value - see
# docs/PHASE2_FEATURE_REPORT.md's "Missing data and sample size" section for why these are
# SHARED counters (one per window) rather than one per metric: every metric computed over
# the same window for the same team-game draws on the identical set of prior games, so a
# per-metric copy of the same number would be pure duplication.
SAMPLE_SIZE_COLUMNS = [
    "games_played_current_season", "n_games_trailing_3", "n_games_trailing_5", "n_games_trailing_8",
    "qb_starts_season", "redzone_trips_n", "redzone_trips_n_allowed", "fg_attempt_n", "punt_n",
    "has_prev_season_data",
]

# Every column ending in one of these is a raw intermediate ingredient (a sum or a play
# count feeding a rate computed elsewhere) - never part of the public feature table. This
# is enforced by `select_public_columns`, not just by convention.
_INTERNAL_SUFFIXES = ("_sum", "_sum_comp")
_INTERNAL_EXACT_DROP = {
    "sort_ts", "home_score", "away_score", "game_status",  # results/bookkeeping - never features
    "off_plays_n", "off_plays_n_comp", "off_dropback_n", "off_rush_n", "off_pass_attempt_n",
    "off_complete_n", "off_sack_n", "off_int_n", "off_fumble_lost_n", "off_qb_hit_n",
    "off_explosive_pass_n", "off_explosive_rush_n", "off_early_down_n", "off_third_down_att_n",
    "off_third_down_conv_n", "off_cpoe_n", "off_scramble_n", "redzone_td_n",
    "def_plays_n", "def_plays_n_comp", "def_dropback_n", "def_rush_n", "def_pass_attempt_n",
    "def_complete_n", "def_sack_n", "def_int_n", "def_fumble_lost_n", "def_qb_hit_n",
    "def_explosive_pass_n", "def_explosive_rush_n", "def_early_down_n", "def_third_down_att_n",
    "def_third_down_conv_n", "def_cpoe_n", "def_scramble_n", "redzone_td_n_allowed",
    "fg_made_n",
}


def select_public_columns(team_game: pl.DataFrame) -> pl.DataFrame:
    """Drops every internal/atomic/result column, keeping identity + sample-size context +
    every computed rate/feature column. This is the boundary between "how the engine works"
    and "what the feature table actually contains" - see leakage test #7 (target columns
    cannot enter the registry), which asserts this function's output never contains a
    result column."""
    drop_cols = {
        c for c in team_game.columns
        if c in _INTERNAL_EXACT_DROP or any(c.endswith(suf) for suf in _INTERNAL_SUFFIXES)
    }
    keep_cols = [c for c in team_game.columns if c not in drop_cols]
    assert_no_denylisted_columns(keep_cols)
    return team_game.select(keep_cols)


def build_team_game_features(conn: sqlite3.Connection, seasons: list[int]) -> pl.DataFrame:
    cfg = get_feature_engine_config()
    windows = cfg["rolling_windows_games"]
    min_obs = cfg["min_observations_rolling"]

    index = build_team_game_index(conn, seasons)
    index = add_rest_days(index)

    pbp_frames = []
    for season in seasons:
        try:
            pbp_frames.append(read_pbp_season(season))
        except FileNotFoundError:
            continue  # season not ingested/promoted - handled as missing downstream, not fatal
    if not pbp_frames:
        raise ValueError(f"No play-by-play data available for any of {seasons}. Run ingestion first.")
    pbp = pl.concat(pbp_frames, how="diagonal_relaxed")

    atomic_pbp = aggregate_team_game_pbp(pbp)
    atomic_st = aggregate_special_teams(pbp)

    team_game = (
        index.join(atomic_pbp, on=["game_id", "team_id"], how="left")
        .join(atomic_st, on=["game_id", "team_id"], how="left")
    )

    team_game = compute_rolling_and_season_to_date(team_game, WINDOWED_METRICS, windows, min_obs)
    season_only_result = compute_rolling_and_season_to_date(team_game, SEASON_ONLY_METRICS + SPECIAL_TEAMS_METRICS, windows=[], min_observations=min_obs)
    new_cols = [c for c in season_only_result.columns if c not in team_game.columns]
    team_game = team_game.join(
        season_only_result.select(["game_id", "team_id", *new_cols]), on=["game_id", "team_id"], how="left"
    )

    qb_atomic = build_qb_atomic(pbp)
    primary_qb = identify_primary_qb(qb_atomic)
    team_game = build_qb_features(team_game, qb_atomic, primary_qb, min_obs)

    teams_repo = TeamsRepository(conn)
    abbr_to_team_id = teams_repo.get_abbr_to_team_id()
    snap_counts = load_snap_counts_for_seasons(seasons)
    team_game = add_snap_continuity(team_game, snap_counts, abbr_to_team_id, cfg["min_observations_snap_continuity"])

    team_game = add_situational_features(team_game, conn, seasons)
    team_game = add_rest_difference(team_game)
    team_game = add_opponent_quality_faced(team_game)
    team_game = add_previous_season_summary(team_game)

    team_game = team_game.with_columns(
        pl.when(pl.col("kickoff_time_naive").is_not_null())
        .then(pl.col("kickoff_time_naive").str.to_datetime() - pl.duration(seconds=1))
        .otherwise((pl.col("game_date") + pl.lit(" 00:00:00")).str.to_datetime() - pl.duration(seconds=1))
        .alias("as_of_timestamp")
    )

    return select_public_columns(team_game)
