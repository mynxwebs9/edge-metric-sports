"""Integration tests for raw snapshotting, normalization, and idempotency.

No live network access: all data here is small, synthetic, constructed in-code to mimic the
real nflreadpy schemas discovered in docs/PHASE1_DATA_REPORT.md - never a network fetch and
never a stored full dataset fixture, per the "CI must not depend on live internet access,
do not put full NFL datasets in git" instruction.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.promote import promote_games, promote_pbp, promote_teams
from nfl_predict.data.raw_store import list_manifests, read_raw_snapshot, write_raw_snapshot
from nfl_predict.data.repositories import GamesRepository, ManifestsRepository, TeamsRepository
from nfl_predict.data.validation import is_promotable


def _sample_teams() -> pl.DataFrame:
    return pl.DataFrame({
        "team_abbr": ["PHI", "DAL", "KC", "LAC"],
        "team_id": ["3700", "1200", "2310", "4400"],
        "team_name": ["Philadelphia Eagles", "Dallas Cowboys", "Kansas City Chiefs", "Los Angeles Chargers"],
        "team_nick": ["Eagles", "Cowboys", "Chiefs", "Chargers"],
        "team_conf": ["NFC", "NFC", "AFC", "AFC"],
        "team_division": ["NFC East", "NFC East", "AFC West", "AFC West"],
    })


def _sample_schedules() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_KC_LAC"],
        "season": [2025, 2025],
        "game_type": ["REG", "REG"],
        "week": [1, 1],
        "gameday": ["2025-09-04", "2025-09-05"],
        "gametime": ["20:20", "20:00"],
        "home_team": ["PHI", "LAC"],
        "away_team": ["DAL", "KC"],
        "home_score": [24, 27],
        "away_score": [20, 21],
        "stadium": ["Lincoln Financial Field", "SoFi Stadium"],
    })


def _sample_pbp() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["2025_01_DAL_PHI", "2025_01_DAL_PHI", "2025_01_KC_LAC"],
        "play_id": [1.0, 40.0, 1.0],
        "season": [2025, 2025, 2025],
        "week": [1, 1, 1],
        "posteam": ["DAL", "PHI", "KC"],
        "defteam": ["PHI", "DAL", "LAC"],
        "home_team": ["PHI", "PHI", "LAC"],
        "away_team": ["DAL", "DAL", "KC"],
    })


def test_raw_snapshot_persists_and_reads_back(isolated_data_dir):
    manifest = write_raw_snapshot(
        _sample_teams(), source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    assert manifest.row_count == 4
    assert manifest.column_count == 6
    assert manifest.duplicate_of_retrieval_id is None

    df_back = read_raw_snapshot(manifest)
    assert df_back.equals(_sample_teams())


def test_manifest_is_listed_after_write(isolated_data_dir):
    write_raw_snapshot(
        _sample_teams(), source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    manifests = list_manifests("nflverse", "teams")
    assert len(manifests) == 1
    assert manifests[0].dataset_name == "teams"


def test_repeated_identical_fetch_is_detected_as_duplicate(isolated_data_dir):
    m1 = write_raw_snapshot(
        _sample_teams(), source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    m2 = write_raw_snapshot(
        _sample_teams(), source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    assert m1.retrieval_id != m2.retrieval_id  # a fetch attempt was still recorded
    assert m2.duplicate_of_retrieval_id == m1.retrieval_id
    assert m2.raw_file_path == m1.raw_file_path  # no duplicate bytes on disk

    manifests = list_manifests("nflverse", "teams")
    assert len(manifests) == 2  # both attempts are in provenance


def test_changed_content_is_not_flagged_as_duplicate(isolated_data_dir):
    m1 = write_raw_snapshot(
        _sample_teams(), source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    changed = _sample_teams().with_columns(pl.lit("Changed Name").alias("team_name"))
    m2 = write_raw_snapshot(
        changed, source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    assert m2.duplicate_of_retrieval_id is None
    assert m2.content_sha256 != m1.content_sha256


def test_promote_teams_then_games_end_to_end(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)

    team_issues = promote_teams(_sample_teams(), retrieval_id="teams-1", conn=conn)
    assert is_promotable(team_issues)
    assert len(TeamsRepository(conn).get_all()) == 4

    game_issues = promote_games(_sample_schedules(), retrieval_id="sched-1", conn=conn)
    assert is_promotable(game_issues)
    games = GamesRepository(conn).get_by_season(2025)
    assert len(games) == 2
    assert {g["home_team_id"] for g in games} == {"3700", "4400"}

    conn.close()


def test_promote_pbp_links_cleanly_against_schedule(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    promote_teams(_sample_teams(), retrieval_id="teams-1", conn=conn)
    promote_games(_sample_schedules(), retrieval_id="sched-1", conn=conn)

    issues = promote_pbp(_sample_pbp(), season=2025, retrieval_id="pbp-1", conn=conn)
    assert is_promotable(issues)
    assert not any(i.code == "pbp_game_not_in_schedule" for i in issues)
    assert not any(i.code == "schedule_game_missing_pbp" for i in issues)

    conn.close()


def test_repeated_ingestion_does_not_duplicate_normalized_games(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    promote_teams(_sample_teams(), retrieval_id="teams-1", conn=conn)

    promote_games(_sample_schedules(), retrieval_id="sched-1", conn=conn)
    promote_games(_sample_schedules(), retrieval_id="sched-2", conn=conn)  # re-run

    games = GamesRepository(conn).get_by_season(2025)
    assert len(games) == 2  # not 4 - UPSERT keeps it idempotent

    conn.close()


def test_manifests_repository_records_and_retrieves(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    manifest = write_raw_snapshot(
        _sample_teams(), source_name="nflverse", dataset_name="teams", requested_seasons=[],
        loader_function="load_teams", loader_package_version="0.1.5",
        source_identifier="test", source_release_identifier=None,
    )
    repo = ManifestsRepository(conn)
    repo.record(manifest)

    rows = repo.by_dataset("nflverse", "teams")
    assert len(rows) == 1
    assert rows[0]["content_sha256"] == manifest.content_sha256
    assert rows[0]["row_count"] == 4

    conn.close()
