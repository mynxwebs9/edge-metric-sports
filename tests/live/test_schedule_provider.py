"""Real production bug (caught by a user comparing a live game's actual kickoff time
against what the site displayed): `get_schedule()` is what the FastAPI read API - and
therefore the website - gets `kickoff_timestamp` from for every game card. It must hand
back a genuine, unambiguous UTC instant, not the schedule's raw ambiguous
`kickoff_time_naive` (Eastern Time, per nflverse - see nfl_predict.data.games), or the site
displays a kickoff time up to 5 hours wrong."""

from __future__ import annotations

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.live.schedule_provider import get_schedule


def _seed(conn):
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name) VALUES (?, ?, ?)",
        [("2310", "KC", "Kansas City Chiefs"), ("1400", "DEN", "Denver Broncos")],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
        "VALUES ('2026_01_DEN_KC', 2026, 'REG', 1, '2026-09-14', '2026-09-14T20:15:00', "
        "'2310', '1400', 'KC', 'DEN', 'scheduled', '2026-09-14T00:00:00')"
    )
    conn.commit()


def test_get_schedule_resolves_the_naive_eastern_kickoff_to_real_utc(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    _seed(conn)
    conn.close()

    snapshot = get_schedule(2026, 1)
    game = next(g for g in snapshot.games if g.game_id == "2026_01_DEN_KC")
    # "20:15:00" is the raw Eastern-time value nflverse publishes for this real game; the
    # real UTC kickoff is 00:15 the next day (5:15 PM Pacific) - not "8:15 PM UTC", which is
    # what the site wrongly showed before this fix.
    assert game.kickoff_timestamp == "2026-09-15T00:15:00+00:00"


def test_get_schedule_leaves_a_missing_kickoff_as_none_not_fabricated(isolated_data_dir):
    conn = get_connection()
    init_schema(conn)
    conn.executemany(
        "INSERT INTO teams (team_id, canonical_abbr, name) VALUES (?, ?, ?)",
        [("2310", "KC", "Kansas City Chiefs"), ("1400", "DEN", "Denver Broncos")],
    )
    conn.execute(
        "INSERT INTO games (game_id, season, season_type, week, game_date, kickoff_time_naive, "
        "home_team_id, away_team_id, home_team_abbr, away_team_abbr, game_status, normalized_at) "
        "VALUES ('2026_01_DEN_KC', 2026, 'REG', 1, '2026-09-14', NULL, "
        "'2310', '1400', 'KC', 'DEN', 'scheduled', '2026-09-14T00:00:00')"
    )
    conn.commit()
    conn.close()

    snapshot = get_schedule(2026, 1)
    game = next(g for g in snapshot.games if g.game_id == "2026_01_DEN_KC")
    assert game.kickoff_timestamp is None
