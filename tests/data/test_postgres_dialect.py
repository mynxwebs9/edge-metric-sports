"""Phase 10: `_translate_for_postgres` is pure string-transform logic (no real Postgres
connection needed to test it) - it must exactly translate every real SQL statement this
project's repository/schedule-provider code actually issues, and must fail loudly on an
unrecognized "INSERT OR REPLACE" rather than silently sending SQLite-only syntax to Postgres.
"""

from __future__ import annotations

import pytest

from nfl_predict.data.db import UnsupportedPostgresStatementError, _translate_for_postgres


def test_positional_placeholders_are_translated():
    assert _translate_for_postgres("SELECT * FROM games WHERE season = ? AND week = ?") == \
        "SELECT * FROM games WHERE season = %s AND week = %s"


def test_named_placeholders_are_translated():
    assert _translate_for_postgres("SELECT * FROM teams WHERE team_id = :team_id") == \
        "SELECT * FROM teams WHERE team_id = %(team_id)s"


def test_whitespace_only_differences_do_not_affect_translation():
    a = _translate_for_postgres("SELECT   *\nFROM games\n  WHERE season = ?")
    b = _translate_for_postgres("SELECT * FROM games WHERE season = ?")
    assert a == b


@pytest.mark.parametrize("table,pk", [("teams", "team_id"), ("team_abbr_aliases", "abbr"), ("games", "game_id"), ("ingestion_manifests", "retrieval_id")])
def test_every_real_insert_or_replace_statement_translates_to_on_conflict(table, pk):
    from nfl_predict.data.repositories import GamesRepository, ManifestsRepository, TeamsRepository  # noqa: F401 - source of truth for the real SQL, exercised indirectly below

    real_statements = {
        "teams": """INSERT OR REPLACE INTO teams
               (team_id, canonical_abbr, name, nickname, conference, division)
               VALUES (:team_id, :canonical_abbr, :name, :nickname, :conference, :division)""",
        "team_abbr_aliases": """INSERT OR REPLACE INTO team_abbr_aliases (abbr, team_id) VALUES (:abbr, :team_id)""",
        "games": """INSERT OR REPLACE INTO games
               (game_id, season, season_type, week, game_date, kickoff_time_naive,
                home_team_id, away_team_id, home_team_abbr, away_team_abbr,
                home_score, away_score, venue, game_status, source_retrieval_id, normalized_at)
               VALUES (:game_id, :season, :season_type, :week, :game_date, :kickoff_time_naive,
                :home_team_id, :away_team_id, :home_team_abbr, :away_team_abbr,
                :home_score, :away_score, :venue, :game_status, :source_retrieval_id, :normalized_at)""",
        "ingestion_manifests": """INSERT OR REPLACE INTO ingestion_manifests
               (retrieval_id, source_name, dataset_name, requested_seasons, loader_function,
                loader_package_version, retrieved_at, source_identifier, source_release_identifier,
                raw_file_path, content_sha256, row_count, column_count, column_names,
                schema_fingerprint, duplicate_of_retrieval_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    }
    translated = _translate_for_postgres(real_statements[table])
    assert translated.startswith(f"INSERT INTO {table} (")
    assert f"ON CONFLICT ({pk}) DO UPDATE SET" in translated
    assert "INSERT OR REPLACE" not in translated
    assert "?" not in translated  # every positional placeholder was translated too


def test_an_unrecognized_insert_or_replace_statement_fails_loudly_not_silently():
    with pytest.raises(UnsupportedPostgresStatementError):
        _translate_for_postgres("INSERT OR REPLACE INTO some_future_table (x) VALUES (?)")
