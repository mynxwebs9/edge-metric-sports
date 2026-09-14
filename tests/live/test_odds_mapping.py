"""Phase 8A provenance correction: `match_odds_event_to_game` must never collapse an
`"unmatched"` (no team-name resolution, or zero candidate games) or an `"ambiguous"` (more
than one candidate game) outcome into a guessed `game_id` - both are explicit, distinct,
non-`"matched"` statuses. Uses an isolated in-memory sqlite connection (never the real
project database) so an artificially-constructed ambiguous scenario never touches real data.
"""

from __future__ import annotations

import sqlite3

import pytest

from nfl_predict.data.db import SQLITE_SCHEMA
from nfl_predict.live.odds_mapping import match_odds_event_to_game


@pytest.fixture
def seeded_conn(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SQLITE_SCHEMA)
    conn.execute("INSERT INTO teams (team_id, canonical_abbr, name) VALUES ('2310', 'KC', 'Kansas City Chiefs')")
    conn.execute("INSERT INTO teams (team_id, canonical_abbr, name) VALUES ('1400', 'DEN', 'Denver Broncos')")
    conn.execute("INSERT INTO teams (team_id, canonical_abbr, name) VALUES ('9999', 'ZZ', 'Zzyzx Zebras')")
    conn.commit()

    monkeypatch.setattr("nfl_predict.live.odds_mapping.get_connection", lambda: conn)
    monkeypatch.setattr("nfl_predict.live.odds_mapping.init_schema", lambda c: None)  # schema already applied above
    yield conn
    # match_odds_event_to_game() closes the connection it's handed in its `finally` block -
    # nothing further to tear down here.


def _insert_game(conn, game_id, season, week, home_id, away_id, kickoff="2026-09-14T20:15:00"):
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, game_status, normalized_at) VALUES (?, ?, 'REG', ?, ?, ?, ?, ?, 'SCHEDULED', ?)",
        (game_id, season, week, kickoff[:10], kickoff, home_id, away_id, kickoff),
    )
    conn.commit()


def test_a_real_matchup_resolves_to_exactly_one_matched_game(seeded_conn):
    _insert_game(seeded_conn, "2026_01_DEN_KC", 2026, 1, "2310", "1400")

    match = match_odds_event_to_game("Kansas City Chiefs", "Denver Broncos", 2026)

    assert match.status == "matched"
    assert match.game_id == "2026_01_DEN_KC"
    assert match.week == 1
    assert match.home_team_id == "2310"
    assert match.away_team_id == "1400"
    assert match.kickoff_timestamp == "2026-09-14T20:15:00"


def test_unresolvable_team_names_are_unmatched_not_guessed(seeded_conn):
    _insert_game(seeded_conn, "2026_01_DEN_KC", 2026, 1, "2310", "1400")

    match = match_odds_event_to_game("Not A Real Team", "Also Not Real", 2026)

    assert match.status == "unmatched"
    assert match.game_id is None


def test_resolvable_teams_with_no_scheduled_game_are_unmatched_not_guessed(seeded_conn):
    # Both team names resolve to real team_ids, but no game row exists for this combination.
    match = match_odds_event_to_game("Kansas City Chiefs", "Denver Broncos", 2026)

    assert match.status == "unmatched"
    assert match.game_id is None


def test_more_than_one_candidate_game_is_ambiguous_not_silently_the_first_row(seeded_conn):
    # A contrived data anomaly: two rows share the exact same (season, home, away) - this
    # must never resolve by picking whichever row sqlite happens to return first.
    _insert_game(seeded_conn, "2026_01_DEN_KC", 2026, 1, "2310", "1400", kickoff="2026-09-14T20:15:00")
    _insert_game(seeded_conn, "2026_99_DEN_KC_DUP", 2026, 1, "2310", "1400", kickoff="2026-09-14T20:15:00")

    match = match_odds_event_to_game("Kansas City Chiefs", "Denver Broncos", 2026)

    assert match.status == "ambiguous"
    assert match.game_id is None


def test_home_and_away_are_not_swapped(seeded_conn):
    """A reversed matchup (the same two teams, opposite home/away) is a DIFFERENT game -
    matching must respect the provider's own home/away assignment, never treat the pair as
    interchangeable."""
    _insert_game(seeded_conn, "2026_01_DEN_KC", 2026, 1, "2310", "1400")  # KC home, DEN away

    reversed_match = match_odds_event_to_game("Denver Broncos", "Kansas City Chiefs", 2026)  # DEN home, KC away - no such game

    assert reversed_match.status == "unmatched"
