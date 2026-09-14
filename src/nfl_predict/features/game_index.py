"""The team-game backbone: one row per team per game, chronologically ordered.

This is the skeleton every other feature module joins against. It deliberately carries
game metadata needed for ordering/rest/home-field context (kickoff time, season, week,
opponent, venue) but NEVER carries scores or results in its public output — see
`docs/PHASE2_FEATURE_REPORT.md`'s "targets stay separate" rule and CLAUDE.md principle 2.
Scores are used internally only to know which games are actually completed (so their
play-by-play can be trusted as a source of prior-game stats), never surfaced as a feature.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

import polars as pl

# Public columns - what every other feature module and the final feature tables see.
# No home_score/away_score/winner/game_status here by construction.
TEAM_GAME_INDEX_COLUMNS = [
    "game_id", "season", "season_type", "week", "game_date", "kickoff_time_naive",
    "team_id", "opponent_id", "is_home", "venue", "sort_ts",
]


def _parse_sort_ts(game_date: str, kickoff_time_naive: str | None) -> str:
    """A single sortable string per game: kickoff time when available, else the game date
    at midnight. Used only for chronological ordering within a team's game sequence -
    never exposed as a feature itself."""
    if kickoff_time_naive:
        return kickoff_time_naive
    return f"{game_date}T00:00:00"


def build_team_game_index(conn: sqlite3.Connection, seasons: list[int]) -> pl.DataFrame:
    """One row per (team, game) for every completed-or-scheduled game in `seasons`, expanded
    from the normalized `games` table (both the home and away perspective of each game).

    Ordering is by `sort_ts` (kickoff time, falling back to game date) within each team,
    which is what every rolling-window computation downstream relies on for "prior games."
    """
    placeholders = ",".join("?" for _ in seasons)
    rows = conn.execute(
        f"""SELECT game_id, season, season_type, week, game_date, kickoff_time_naive,
                   home_team_id, away_team_id, venue, game_status, home_score, away_score
            FROM games WHERE season IN ({placeholders})""",
        seasons,
    ).fetchall()

    records = []
    for r in rows:
        sort_ts = _parse_sort_ts(r["game_date"], r["kickoff_time_naive"])
        common = dict(
            game_id=r["game_id"], season=r["season"], season_type=r["season_type"],
            week=r["week"], game_date=r["game_date"], kickoff_time_naive=r["kickoff_time_naive"],
            venue=r["venue"], sort_ts=sort_ts, game_status=r["game_status"],
            home_score=r["home_score"], away_score=r["away_score"],
        )
        records.append({**common, "team_id": r["home_team_id"], "opponent_id": r["away_team_id"], "is_home": True})
        records.append({**common, "team_id": r["away_team_id"], "opponent_id": r["home_team_id"], "is_home": False})

    if not records:
        return pl.DataFrame(schema={c: pl.Utf8 for c in TEAM_GAME_INDEX_COLUMNS})

    df = pl.DataFrame(records)
    return df.sort(["team_id", "sort_ts"])


def add_rest_days(df: pl.DataFrame, max_meaningful_gap_days: int = 30) -> pl.DataFrame:
    """Adds `days_rest`: days since this team's previous game (by `sort_ts`), null if there
    is no previous game or the gap exceeds `max_meaningful_gap_days` (an off-season gap
    between last season's finale and this season's opener isn't "rest" in any meaningful
    football sense, so it's left null rather than reported as e.g. 200 days)."""

    def _parse_dt(ts: str) -> datetime:
        return datetime.fromisoformat(ts)

    out_rows = []
    for team_id, group in df.sort(["team_id", "sort_ts"]).group_by("team_id", maintain_order=True):
        rows = group.to_dicts()
        prev_dt: datetime | None = None
        for row in rows:
            cur_dt = _parse_dt(row["sort_ts"])
            if prev_dt is not None:
                gap = (cur_dt - prev_dt).days
                row["days_rest"] = gap if gap <= max_meaningful_gap_days else None
            else:
                row["days_rest"] = None
            out_rows.append(row)
            prev_dt = cur_dt

    return pl.DataFrame(out_rows)
