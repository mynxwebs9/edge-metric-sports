"""Coverage report `fetched`/`usable` semantics - see coverage.py's module docstring and
docs/PHASE1_DATA_REPORT.md's "Coverage report terminology" section for the policy this
verifies. No live network access.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.data.coverage import generate_coverage_rows
from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.ingest import process_fetch_outcome
from nfl_predict.data.ingestion import FetchOutcome
from nfl_predict.data.nflverse_loader import ALL_DATASET_SPECS
from nfl_predict.data.raw_store import write_raw_snapshot
from nfl_predict.data.repositories import ManifestsRepository, ValidationRepository


def _ingest(conn, dataset_key: str, df: pl.DataFrame, season: int) -> None:
    manifest = write_raw_snapshot(
        df, source_name="nflverse", dataset_name=dataset_key, requested_seasons=[season],
        loader_function=f"load_{dataset_key}", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None, season_for_retrieval_id=season,
    )
    outcome = FetchOutcome(dataset_key, season, "fetched", manifest, df.height, None, dataframe=df)
    process_fetch_outcome(dataset_key, ALL_DATASET_SPECS[dataset_key], outcome, conn, ManifestsRepository(conn), ValidationRepository(conn))


def _row(rows: list[dict], dataset_name: str, season: int) -> dict:
    return next(r for r in rows if r["dataset_name"] == dataset_name and r["season"] == season)


def test_unfetched_season_is_fetched_false_usable_false(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    conn.close()

    rows = generate_coverage_rows([2025])
    row = _row(rows, "player_stats", 2025)

    assert row["fetched"] is False
    assert row["usable"] is False


def test_fetched_with_fatal_is_usable_false(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    broken_pbp = pl.DataFrame({"game_id": ["x"], "season": [2025], "week": [1]})  # missing play_id
    _ingest(conn, "pbp", broken_pbp, 2025)
    conn.close()

    rows = generate_coverage_rows([2025])
    row = _row(rows, "pbp", 2025)

    assert row["fetched"] is True
    assert row["validation_fatal"] >= 1
    assert row["usable"] is False


def test_fetched_zero_row_snapshot_is_usable_false(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    empty_snap_counts = pl.DataFrame({
        "game_id": [], "season": [], "week": [], "pfr_player_id": [], "team": [],
    }).cast({"season": pl.Int64, "week": pl.Int64})
    _ingest(conn, "snap_counts", empty_snap_counts, 2012)
    conn.close()

    rows = generate_coverage_rows([2012])
    row = _row(rows, "snap_counts", 2012)

    assert row["fetched"] is True
    assert row["row_count"] == 0
    assert row["validation_fatal"] == 0  # a real, valid, empty result - not a validation defect
    assert row["usable"] is False  # but still not usable - see coverage.py's docstring


def test_fetched_clean_nonempty_snapshot_is_usable_true(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    teams_df = pl.DataFrame({
        "team_abbr": ["PHI"], "team_id": ["3700"], "team_name": ["Philadelphia Eagles"],
        "team_nick": ["Eagles"], "team_conf": ["NFC"], "team_division": ["NFC East"],
    })
    manifest = write_raw_snapshot(
        teams_df, source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    outcome = FetchOutcome("teams", None, "fetched", manifest, teams_df.height, None, dataframe=teams_df)
    process_fetch_outcome("teams", ALL_DATASET_SPECS["teams"], outcome, conn, ManifestsRepository(conn), ValidationRepository(conn))
    conn.close()

    rows = generate_coverage_rows([])
    row = _row(rows, "teams", None)

    assert row["fetched"] is True
    assert row["usable"] is True
