"""Phase 8A: maps a live odds event's free-text team names (e.g. "Houston Texans", as The
Odds API reports them) to this project's internal `game_id` - using the `teams.name` column
(already an exact-format match for how nflverse and The Odds API both spell full team
names) and the current schedule. Never a hardcoded/guessed abbreviation table.

**Phase 8A provenance correction:** `match_odds_event_to_game` returns an explicit
`EventGameMatch` with a `status` of `"matched"`, `"unmatched"`, or `"ambiguous"` - never just
a bare `game_id | None`. An `"unmatched"` result (a team name that doesn't resolve, or zero
games for that (season, home, away) combination) and an `"ambiguous"` result (MORE THAN ONE
game matches - e.g. a data anomaly, or two teams meeting twice in the same season under an
identical home/away ordering) are both explicit, distinct, non-`"matched"` outcomes. Neither
one ever falls back to guessing a `game_id` - callers must check `status == "matched"` before
trusting `game_id`/`week`/`kickoff_timestamp`, never assume a non-None `game_id` implies a
confident match.
"""

from __future__ import annotations

from dataclasses import dataclass

from nfl_predict.data.db import get_connection, init_schema


def _team_name_to_id(conn) -> dict[str, str]:
    rows = conn.execute("SELECT team_id, name FROM teams").fetchall()
    return {r["name"]: r["team_id"] for r in rows}


@dataclass(frozen=True)
class EventGameMatch:
    status: str  # "matched" | "unmatched" | "ambiguous"
    game_id: str | None
    season: int
    week: int | None
    kickoff_timestamp: str | None
    home_team_id: str | None  # normalized team_id the provider's home_team string resolved to, or None if unresolved
    away_team_id: str | None


def match_odds_event_to_game(home_team: str, away_team: str, season: int) -> EventGameMatch:
    conn = get_connection()
    init_schema(conn)
    try:
        name_to_id = _team_name_to_id(conn)
        home_id = name_to_id.get(home_team)
        away_id = name_to_id.get(away_team)
        if home_id is None or away_id is None:
            return EventGameMatch(status="unmatched", game_id=None, season=season, week=None, kickoff_timestamp=None, home_team_id=home_id, away_team_id=away_id)

        rows = conn.execute(
            "SELECT game_id, week, kickoff_time_naive FROM games WHERE season = ? AND home_team_id = ? AND away_team_id = ?",
            (season, home_id, away_id),
        ).fetchall()
        if not rows:
            return EventGameMatch(status="unmatched", game_id=None, season=season, week=None, kickoff_timestamp=None, home_team_id=home_id, away_team_id=away_id)
        if len(rows) > 1:
            # More than one game shares this exact (season, home, away) combination - a
            # genuine ambiguity (data anomaly, or a rescheduled/duplicated row), never
            # silently resolved by picking the first row.
            return EventGameMatch(status="ambiguous", game_id=None, season=season, week=None, kickoff_timestamp=None, home_team_id=home_id, away_team_id=away_id)

        row = rows[0]
        return EventGameMatch(
            status="matched", game_id=row["game_id"], season=season, week=row["week"],
            kickoff_timestamp=row["kickoff_time_naive"], home_team_id=home_id, away_team_id=away_id,
        )
    finally:
        conn.close()
