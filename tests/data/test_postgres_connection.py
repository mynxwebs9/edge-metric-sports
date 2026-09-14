"""Phase 10: the relational (teams/games) side of the Postgres backend, against a REAL
Postgres instance - see `tests/storage/test_postgres_blob_store.py`'s module docstring for
why this is skipped rather than mocked, and how to actually run it.

    NFL_TEST_POSTGRES_URL=postgresql://user:pass@host:5432/dbname pytest tests/data/test_postgres_connection.py -v
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NFL_TEST_POSTGRES_URL"),
    reason="Set NFL_TEST_POSTGRES_URL to a real, disposable Postgres database to run this - see module docstring.",
)


@pytest.fixture
def pg_conn(monkeypatch):
    from nfl_predict.data.db import _PostgresConnectionAdapter, init_schema

    monkeypatch.setattr(
        "nfl_predict.data.db.get_settings",
        lambda: type("S", (), {"storage_backend": "postgres", "database_url": os.environ["NFL_TEST_POSTGRES_URL"]})(),
    )
    conn = _PostgresConnectionAdapter(os.environ["NFL_TEST_POSTGRES_URL"])
    init_schema(conn)
    conn.execute("DELETE FROM team_abbr_aliases")
    conn.execute("DELETE FROM games")
    conn.execute("DELETE FROM teams")
    conn.commit()
    yield conn
    conn.close()


def test_teams_upsert_via_insert_or_replace_translation_round_trips(pg_conn):
    from nfl_predict.data.repositories import TeamsRepository

    repo = TeamsRepository(pg_conn)
    repo.upsert_teams([{"team_id": "2310", "canonical_abbr": "KC", "name": "Kansas City Chiefs", "nickname": "Chiefs", "conference": "AFC", "division": "West"}])
    rows = repo.get_all()
    assert len(rows) == 1
    assert rows[0]["name"] == "Kansas City Chiefs"

    # A second upsert with a changed field must REPLACE, not duplicate - proves the
    # ON CONFLICT DO UPDATE translation actually updates every column, not just insert-once.
    repo.upsert_teams([{"team_id": "2310", "canonical_abbr": "KC", "name": "Kansas City Chiefs (renamed)", "nickname": "Chiefs", "conference": "AFC", "division": "West"}])
    rows = repo.get_all()
    assert len(rows) == 1
    assert rows[0]["name"] == "Kansas City Chiefs (renamed)"


def test_games_repository_get_by_season_round_trips(pg_conn):
    from nfl_predict.data.repositories import GamesRepository

    repo = GamesRepository(pg_conn)
    repo.upsert_games(
        [{
            "game_id": "2026_01_DEN_KC", "season": 2026, "season_type": "REG", "week": 1,
            "game_date": "2026-09-14", "kickoff_time_naive": "2026-09-14T20:15:00",
            "home_team_id": "2310", "away_team_id": "1400", "home_team_abbr": "KC", "away_team_abbr": "DEN",
            "home_score": None, "away_score": None, "venue": "Arrowhead Stadium", "game_status": "scheduled",
        }],
        retrieval_id="r1",
    )
    rows = repo.get_by_season(2026)
    assert [r["game_id"] for r in rows] == ["2026_01_DEN_KC"]


def test_schedule_provider_reads_through_the_postgres_backend(pg_conn):
    from nfl_predict.data.repositories import GamesRepository, TeamsRepository
    from nfl_predict.live import schedule_provider

    TeamsRepository(pg_conn).upsert_teams([
        {"team_id": "2310", "canonical_abbr": "KC", "name": "Kansas City Chiefs", "nickname": None, "conference": None, "division": None},
        {"team_id": "1400", "canonical_abbr": "DEN", "name": "Denver Broncos", "nickname": None, "conference": None, "division": None},
    ])
    GamesRepository(pg_conn).upsert_games(
        [{
            "game_id": "2026_01_DEN_KC", "season": 2026, "season_type": "REG", "week": 1,
            "game_date": "2026-09-14", "kickoff_time_naive": "2026-09-14T20:15:00",
            "home_team_id": "2310", "away_team_id": "1400", "home_team_abbr": "KC", "away_team_abbr": "DEN",
            "home_score": None, "away_score": None, "venue": None, "game_status": "scheduled",
        }],
        retrieval_id="r1",
    )
    pg_conn.commit()

    snapshot = schedule_provider.get_schedule(2026, 1)
    assert [g.game_id for g in snapshot.games] == ["2026_01_DEN_KC"]
