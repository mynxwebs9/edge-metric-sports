"""Phase 8A Step 3: current-season schedule retrieval.

Built entirely on the existing Phase 1 ingestion architecture (`nfl_predict.data.ingest`,
the `games` table) - never a manually entered game list. This module only READS the
already-ingested schedule; it does not fetch anything itself (run
`python -m nfl_predict.data.ingest --dataset schedules --seasons <season>` first if a
season isn't ingested yet).

**Known limitation**: Phase 1's `game_status` is binary (`"final"` if both scores are
populated, else `"scheduled"` - see `nfl_predict.data.games.normalize_games`). There is no
distinct postponed/cancelled state in the current schema, so
`postponed_or_cancelled_games` always returns empty until that's added upstream - this is
documented here, not silently pretended away.
"""

from __future__ import annotations

from dataclasses import dataclass

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.repositories import ManifestsRepository

FINAL_STATUS = "final"
SCHEDULED_STATUS = "scheduled"


@dataclass(frozen=True)
class ScheduleGame:
    game_id: str
    season: int
    week: int
    season_type: str
    home_team_id: str
    away_team_id: str
    home_team_abbr: str
    away_team_abbr: str
    kickoff_timestamp: str | None
    game_status: str


@dataclass(frozen=True)
class ScheduleSnapshot:
    season: int
    current_week: int | None
    games: tuple[ScheduleGame, ...]
    source: str
    retrieval_timestamp: str | None


def get_current_season() -> int:
    conn = get_connection()
    init_schema(conn)
    try:
        row = conn.execute("SELECT MAX(season) AS s FROM games").fetchone()
        if row is None or row["s"] is None:
            raise ValueError("No seasons found in the games table - ingest a schedule first.")
        return row["s"]
    finally:
        conn.close()


def _infer_current_week(games: tuple[ScheduleGame, ...]) -> int | None:
    """The current week is the earliest week that still has at least one non-final game -
    the next slate still to be decided. None if every game in the season is final."""
    pending_weeks = sorted({g.week for g in games if g.game_status != FINAL_STATUS})
    return pending_weeks[0] if pending_weeks else None


def get_schedule(season: int, week: int | None = None) -> ScheduleSnapshot:
    conn = get_connection()
    init_schema(conn)
    try:
        query = (
            "SELECT game_id, season, week, season_type, home_team_id, away_team_id, "
            "home_team_abbr, away_team_abbr, kickoff_time_naive, game_status "
            "FROM games WHERE season = ?"
        )
        params: list = [season]
        if week is not None:
            query += " AND week = ?"
            params.append(week)
        query += " ORDER BY week, kickoff_time_naive"
        rows = conn.execute(query, params).fetchall()
        manifest = ManifestsRepository(conn).get_latest_canonical_manifest("nflverse", "schedules", season)
    finally:
        conn.close()

    games = tuple(
        ScheduleGame(
            game_id=r["game_id"], season=r["season"], week=r["week"], season_type=r["season_type"],
            home_team_id=r["home_team_id"], away_team_id=r["away_team_id"],
            home_team_abbr=r["home_team_abbr"], away_team_abbr=r["away_team_abbr"],
            kickoff_timestamp=r["kickoff_time_naive"], game_status=r["game_status"],
        )
        for r in rows
    )

    return ScheduleSnapshot(
        season=season, current_week=_infer_current_week(games) if week is None else week, games=games,
        source="nflverse-data GitHub release, via nflreadpy.load_schedules() (see nfl_predict.data.ingest)",
        retrieval_timestamp=manifest["retrieved_at"] if manifest else None,
    )


def upcoming_games(snapshot: ScheduleSnapshot) -> tuple[ScheduleGame, ...]:
    return tuple(g for g in snapshot.games if g.game_status == SCHEDULED_STATUS)


def completed_games(snapshot: ScheduleSnapshot) -> tuple[ScheduleGame, ...]:
    return tuple(g for g in snapshot.games if g.game_status == FINAL_STATUS)


def postponed_or_cancelled_games(snapshot: ScheduleSnapshot) -> tuple[ScheduleGame, ...]:
    """Always empty today - see module docstring's known limitation."""
    return tuple(g for g in snapshot.games if g.game_status not in (FINAL_STATUS, SCHEDULED_STATUS))
