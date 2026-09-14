"""Regression tests for the FATAL-blocks-promotion contract.

These exercise the real orchestration function (`ingest.process_fetch_outcome`), not just
`is_promotable()` in isolation - a synthetic broken DataFrame is written as a real raw
snapshot (via `write_raw_snapshot`, isolated to a tmp data dir) and handed through the exact
same code path `ingest.py`'s CLI uses, so a regression in the wiring (not just in the
validation functions themselves) would actually be caught. No live network access.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.ingest import process_fetch_outcome
from nfl_predict.data.ingestion import FetchOutcome
from nfl_predict.data.nflverse_loader import ALL_DATASET_SPECS
from nfl_predict.data.pbp_store import pbp_partition_path
from nfl_predict.data.raw_store import write_raw_snapshot
from nfl_predict.data.repositories import GamesRepository, ManifestsRepository, TeamsRepository, ValidationRepository


def _outcome_for(dataset_key: str, df: pl.DataFrame, season: int | None) -> FetchOutcome:
    manifest = write_raw_snapshot(
        df, source_name="nflverse", dataset_name=dataset_key,
        requested_seasons=[season] if season is not None else [],
        loader_function=f"load_{dataset_key}", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
        season_for_retrieval_id=season,
    )
    return FetchOutcome(dataset_key, season, "fetched", manifest, df.height, None, dataframe=df)


def _repos(conn):
    return ManifestsRepository(conn), ValidationRepository(conn)


def test_schedules_missing_critical_column_blocks_promotion(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    manifests_repo, validation_repo = _repos(conn)

    # A real schedules frame is missing "stadium" here - one of games.REQUIRED_RAW_COLUMNS,
    # which schedules.critical_columns now matches exactly (see nflverse_loader.py).
    broken = pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI"], "season": [2025], "game_type": ["REG"], "week": [1],
        "gameday": ["2025-09-04"], "gametime": ["20:20"], "home_team": ["PHI"], "away_team": ["DAL"],
        "home_score": [24], "away_score": [20],
        # "stadium" deliberately omitted
    })
    outcome = _outcome_for("schedules", broken, season=2025)

    issues = process_fetch_outcome("schedules", ALL_DATASET_SPECS["schedules"], outcome, conn, manifests_repo, validation_repo)

    assert any(i.level == "FATAL" and i.code == "missing_critical_columns" for i in issues)
    assert GamesRepository(conn).get_by_season(2025) == []
    conn.close()


def test_pbp_missing_critical_column_blocks_promotion_and_no_parquet_written(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    manifests_repo, validation_repo = _repos(conn)

    # Missing "play_id", one of pbp's critical_columns.
    broken = pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_DAL_PHI"],
        "season": [2025, 2025],
        "week": [1, 1],
    })
    outcome = _outcome_for("pbp", broken, season=2025)

    issues = process_fetch_outcome("pbp", ALL_DATASET_SPECS["pbp"], outcome, conn, manifests_repo, validation_repo)

    fatals = [i for i in issues if i.level == "FATAL"]
    assert [i.code for i in fatals] == ["missing_critical_columns"]
    # validate_pbp() (inside promote_pbp) checks the same game_id/play_id columns and would
    # produce its own "missing_critical_columns" FATAL - but promote_pbp is never called
    # here (the generic gate blocked it first), so there is exactly one FATAL recorded, not
    # two duplicate ones from two validation layers finding the same problem.
    assert not pbp_partition_path(2025).exists()
    conn.close()


def test_teams_missing_critical_column_blocks_promotion(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    manifests_repo, validation_repo = _repos(conn)

    # Missing "team_nick", one of teams.REQUIRED_RAW_COLUMNS (which teams.critical_columns
    # now matches exactly).
    broken = pl.DataFrame({
        "team_abbr": ["PHI"], "team_id": ["3700"], "team_name": ["Philadelphia Eagles"],
        "team_conf": ["NFC"], "team_division": ["NFC East"],
        # "team_nick" deliberately omitted
    })
    outcome = _outcome_for("teams", broken, season=None)

    issues = process_fetch_outcome("teams", ALL_DATASET_SPECS["teams"], outcome, conn, manifests_repo, validation_repo)

    fatals = [i for i in issues if i.level == "FATAL"]
    assert [i.code for i in fatals] == ["missing_critical_columns"]
    # validate_raw_teams() (inside promote_teams) would independently find the same missing
    # column - but promote_teams is never called here, so this isn't double-recorded.
    assert TeamsRepository(conn).get_all() == []
    conn.close()


def test_promotion_proceeds_normally_when_no_fatal_issue(isolated_data_dir):
    """Sanity check that the FATAL gate doesn't also block the happy path."""
    conn = get_connection()
    init_schema(conn)
    manifests_repo, validation_repo = _repos(conn)

    teams_df = pl.DataFrame({
        "team_abbr": ["PHI", "DAL"], "team_id": ["3700", "1200"],
        "team_name": ["Philadelphia Eagles", "Dallas Cowboys"],
        "team_nick": ["Eagles", "Cowboys"], "team_conf": ["NFC", "NFC"],
        "team_division": ["NFC East", "NFC East"],
    })
    teams_outcome = _outcome_for("teams", teams_df, season=None)
    process_fetch_outcome("teams", ALL_DATASET_SPECS["teams"], teams_outcome, conn, manifests_repo, validation_repo)
    assert len(TeamsRepository(conn).get_all()) == 2

    schedules_df = pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI"], "season": [2025], "game_type": ["REG"], "week": [1],
        "gameday": ["2025-09-04"], "gametime": ["20:20"], "home_team": ["PHI"], "away_team": ["DAL"],
        "home_score": [24], "away_score": [20], "stadium": ["Lincoln Financial Field"],
    })
    schedules_outcome = _outcome_for("schedules", schedules_df, season=2025)
    issues = process_fetch_outcome("schedules", ALL_DATASET_SPECS["schedules"], schedules_outcome, conn, manifests_repo, validation_repo)

    assert not any(i.level == "FATAL" for i in issues)
    assert len(GamesRepository(conn).get_by_season(2025)) == 1
    conn.close()
