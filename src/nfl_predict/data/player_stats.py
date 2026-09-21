"""Reads real per-player weekly box-score stats from the latest canonical, provenance-
tracked raw `nflverse` `player_stats` snapshot (ingested via
`python -m nfl_predict.data.ingest --dataset player_stats --seasons <season>`) - the same
"read the recorded snapshot, never re-fetch, never fabricate" pattern
`nfl_predict.market.snapshot_store` uses for `schedules`.

Used to grade player-prop parlay legs. Deliberately tiny: a whitelist of countable stats, a
name->player_id resolver that refuses to guess, and a stat lookup that distinguishes "this
player has no row" from "this game's stats haven't been ingested yet" - conflating those two
would let a stale snapshot silently grade a leg as a loss.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.repositories import ManifestsRepository

# nflverse `player_stats` column -> plain-English label used in leg descriptions.
PROP_STATS = {
    "passing_yards": "passing yards",
    "rushing_yards": "rushing yards",
    "receiving_yards": "receiving yards",
    "receptions": "receptions",
    "passing_tds": "passing TDs",
    "rushing_tds": "rushing TDs",
    "receiving_tds": "receiving TDs",
}


class PlayerStatsUnavailableError(Exception):
    """No ingested player_stats snapshot exists for the season."""


class PlayerNotFoundError(ValueError):
    """The name matches no player - or more than one - so the caller must be more specific."""


def load_player_week_stats(season: int) -> pl.DataFrame:
    conn = get_connection()
    init_schema(conn)
    try:
        manifest = ManifestsRepository(conn).get_latest_canonical_manifest("nflverse", "player_stats", season)
    finally:
        conn.close()
    if manifest is None:
        raise PlayerStatsUnavailableError(
            f"No ingested player_stats snapshot for season={season} - run "
            f"`python -m nfl_predict.data.ingest --dataset player_stats --seasons {season}` first."
        )
    return pl.read_parquet(manifest["raw_file_path"], memory_map=False)


def resolve_player(stats: pl.DataFrame, name: str, candidate_teams: tuple[str, ...]) -> dict:
    """The one player whose display name matches (case-insensitive, exact) and who has played
    for one of `candidate_teams` (the two teams in the game) this season. Raises rather than
    guessing when there is no match or more than one distinct player."""
    matches = (
        stats.filter(
            (pl.col("player_display_name").str.to_lowercase() == name.strip().lower())
            & pl.col("team").is_in(list(candidate_teams))
        )
        .select(["player_id", "player_display_name", "position", "team"])
        .unique(subset=["player_id"])
        .to_dicts()
    )
    if not matches:
        raise PlayerNotFoundError(
            f"No player named {name!r} on {' or '.join(candidate_teams)} in the ingested stats - "
            "check the spelling (full name as nflverse lists it), or that he has played this season."
        )
    if len(matches) > 1:
        raise PlayerNotFoundError(f"{name!r} matches more than one player on {' / '.join(candidate_teams)}: {matches}")
    return matches[0]


def team_has_stats_for_week(stats: pl.DataFrame, team: str, week: int) -> bool:
    return stats.filter((pl.col("team") == team) & (pl.col("week") == week) & (pl.col("season_type") == "REG")).height > 0


def player_stat_for_week(stats: pl.DataFrame, player_id: str, week: int, stat: str) -> int | None:
    """The player's count for `stat` in `week`, or None if he has no row that week."""
    if stat not in PROP_STATS:
        raise ValueError(f"stat={stat!r} not supported - only {sorted(PROP_STATS)}")
    rows = stats.filter((pl.col("player_id") == player_id) & (pl.col("week") == week) & (pl.col("season_type") == "REG"))
    if rows.height == 0:
        return None
    value = rows[stat][0]
    return None if value is None else int(value)
