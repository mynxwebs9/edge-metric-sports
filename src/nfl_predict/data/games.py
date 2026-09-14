"""Normalized game/schedule table.

Uses nflverse's own `game_id` (format `{season}_{week:02d}_{away_team}_{home_team}`,
constructed by nflverse from immutable inputs — never from a mutable field like the final
score) as this project's stable internal game identifier, per the "prefer a durable upstream
identifier" instruction. We do not reconstruct or reformat it — the value from
`load_schedules()` is stored verbatim.
"""

from __future__ import annotations

import polars as pl

REQUIRED_RAW_COLUMNS = {
    "game_id", "season", "game_type", "week", "gameday", "gametime",
    "home_team", "away_team", "home_score", "away_score", "stadium",
}

# nflverse's `game_type` values, mapped to a coarser season_type. Anything not in this map
# passes through unchanged and is flagged by validation (see validation.py) as an unexpected
# schema/value change worth a human look, rather than silently dropped.
SEASON_TYPE_MAP = {
    "REG": "REG",
    "PRE": "PRE",
    "WC": "POST",
    "DIV": "POST",
    "CON": "POST",
    "SB": "POST",
}


def normalize_games(raw_schedules: pl.DataFrame, abbr_to_team_id: dict[str, str]) -> pl.DataFrame:
    missing = REQUIRED_RAW_COLUMNS - set(raw_schedules.columns)
    if missing:
        raise ValueError(f"raw schedules data is missing required columns: {sorted(missing)}")

    df = raw_schedules.select(
        pl.col("game_id"),
        pl.col("season"),
        pl.col("game_type").alias("raw_game_type"),
        pl.col("week"),
        pl.col("gameday").alias("game_date"),
        pl.col("gametime"),
        pl.col("home_team").alias("home_team_abbr"),
        pl.col("away_team").alias("away_team_abbr"),
        pl.col("home_score"),
        pl.col("away_score"),
        pl.col("stadium").alias("venue"),
    )

    df = df.with_columns(
        pl.col("raw_game_type")
        .replace_strict(SEASON_TYPE_MAP, default=pl.col("raw_game_type"))
        .alias("season_type"),
        pl.col("home_team_abbr")
        .replace_strict(abbr_to_team_id, default=None)
        .alias("home_team_id"),
        pl.col("away_team_abbr")
        .replace_strict(abbr_to_team_id, default=None)
        .alias("away_team_id"),
        # gametime is the nflverse-published local kickoff time string (documented by the
        # source as Eastern Time; nflverse does not publish a UTC offset per game), combined
        # with game_date into a naive (timezone-unaware) ISO-ish string. This is a known
        # limitation carried over from docs/DATA_SOURCES.md's odds-source timezone-ambiguity
        # note - do not treat this as a UTC or venue-local timestamp without resolving that.
        pl.when(pl.col("gametime").is_not_null())
        .then(pl.col("game_date") + pl.lit("T") + pl.col("gametime") + pl.lit(":00"))
        .otherwise(None)
        .alias("kickoff_time_naive"),
        pl.when(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
        .then(pl.lit("final"))
        .otherwise(pl.lit("scheduled"))
        .alias("game_status"),
    )

    return df.select(
        "game_id", "season", "season_type", "week", "game_date", "kickoff_time_naive",
        "home_team_id", "away_team_id", "home_team_abbr", "away_team_abbr",
        "home_score", "away_score", "venue", "game_status",
    )
