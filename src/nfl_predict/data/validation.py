"""Data-quality validation for Phase 1 datasets.

Explicit, never-silent-repair checks. Every issue is classified INFO/WARNING/ERROR/FATAL;
a FATAL issue on a dataset snapshot must prevent that snapshot from being promoted to
normalized data (see `is_promotable`). This module never mutates the data it's checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import polars as pl

Level = Literal["INFO", "WARNING", "ERROR", "FATAL"]


@dataclass(frozen=True)
class ValidationIssue:
    level: Level
    code: str
    message: str
    context: dict[str, Any]


def is_promotable(issues: list[ValidationIssue]) -> bool:
    return not any(issue.level == "FATAL" for issue in issues)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def validate_raw_teams(raw_teams: pl.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required = {"team_abbr", "team_id", "team_name", "team_conf", "team_division"}
    missing_cols = required - set(raw_teams.columns)
    if missing_cols:
        issues.append(ValidationIssue(
            "FATAL", "missing_critical_columns",
            f"raw teams data is missing required columns: {sorted(missing_cols)}",
            {"missing_columns": sorted(missing_cols)},
        ))
        return issues  # nothing else below is checkable without these columns

    ambiguous = (
        raw_teams.group_by("team_abbr")
        .agg(pl.col("team_id").n_unique().alias("n_team_ids"))
        .filter(pl.col("n_team_ids") > 1)
    )
    for row in ambiguous.iter_rows(named=True):
        issues.append(ValidationIssue(
            "FATAL", "ambiguous_team_abbr_mapping",
            f"team_abbr {row['team_abbr']!r} maps to {row['n_team_ids']} different team_ids",
            {"team_abbr": row["team_abbr"]},
        ))

    for col in ("team_abbr", "team_id"):
        n_null = raw_teams[col].null_count()
        if n_null:
            issues.append(ValidationIssue(
                "FATAL", "null_team_identifier",
                f"{n_null} row(s) have a null {col}", {"column": col, "null_count": n_null},
            ))

    return issues


# ---------------------------------------------------------------------------
# Games / schedules
# ---------------------------------------------------------------------------

def validate_normalized_games(games: pl.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    n_null_id = games["game_id"].null_count()
    if n_null_id:
        issues.append(ValidationIssue(
            "FATAL", "missing_game_id", f"{n_null_id} row(s) have a null game_id",
            {"null_count": n_null_id},
        ))

    dupes = games.filter(pl.col("game_id").is_not_null()).group_by("game_id").agg(
        pl.len().alias("n")
    ).filter(pl.col("n") > 1)
    for row in dupes.iter_rows(named=True):
        issues.append(ValidationIssue(
            "FATAL", "duplicate_game_id", f"game_id {row['game_id']!r} appears {row['n']} times",
            {"game_id": row["game_id"], "count": row["n"]},
        ))

    same_team = games.filter(pl.col("home_team_id") == pl.col("away_team_id"))
    for row in same_team.iter_rows(named=True):
        issues.append(ValidationIssue(
            "FATAL", "home_equals_away",
            f"game {row['game_id']!r} has home_team_id == away_team_id", {"game_id": row["game_id"]},
        ))

    unresolved = games.filter(pl.col("home_team_id").is_null() | pl.col("away_team_id").is_null())
    for row in unresolved.iter_rows(named=True):
        issues.append(ValidationIssue(
            "ERROR", "unresolved_team_abbr",
            f"game {row['game_id']!r} has an unresolved team abbreviation "
            f"(home={row['home_team_abbr']!r}, away={row['away_team_abbr']!r})",
            {"game_id": row["game_id"], "home_team_abbr": row["home_team_abbr"], "away_team_abbr": row["away_team_abbr"]},
        ))

    bad_week = games.filter((pl.col("week") < 1) | (pl.col("week") > 22))
    for row in bad_week.iter_rows(named=True):
        issues.append(ValidationIssue(
            "ERROR", "impossible_week", f"game {row['game_id']!r} has implausible week={row['week']}",
            {"game_id": row["game_id"], "week": row["week"]},
        ))

    bad_season = games.filter((pl.col("season") < 1920) | (pl.col("season") > 2100))
    for row in bad_season.iter_rows(named=True):
        issues.append(ValidationIssue(
            "ERROR", "impossible_season", f"game {row['game_id']!r} has implausible season={row['season']}",
            {"game_id": row["game_id"], "season": row["season"]},
        ))

    for row in games.iter_rows(named=True):
        if row["game_date"] is None:
            issues.append(ValidationIssue(
                "ERROR", "missing_game_date", f"game {row['game_id']!r} has no game_date",
                {"game_id": row["game_id"]},
            ))
            continue
        try:
            date.fromisoformat(str(row["game_date"]))
        except ValueError:
            issues.append(ValidationIssue(
                "ERROR", "invalid_game_date",
                f"game {row['game_id']!r} has an unparseable game_date {row['game_date']!r}",
                {"game_id": row["game_id"], "game_date": row["game_date"]},
            ))

    unmapped_types = set(games["season_type"].unique().to_list()) - {"REG", "PRE", "POST"}
    for st in unmapped_types:
        issues.append(ValidationIssue(
            "WARNING", "unrecognized_season_type",
            f"season_type {st!r} was not in the known REG/PRE/POST mapping (raw game_type "
            "passed through unchanged) - check games.SEASON_TYPE_MAP",
            {"season_type": st},
        ))

    return issues


# ---------------------------------------------------------------------------
# Play-by-play
# ---------------------------------------------------------------------------

def validate_pbp(pbp: pl.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if "game_id" not in pbp.columns or "play_id" not in pbp.columns:
        issues.append(ValidationIssue(
            "FATAL", "missing_critical_columns",
            "play-by-play data is missing game_id and/or play_id", {},
        ))
        return issues

    dupes = (
        pbp.group_by(["game_id", "play_id"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    )
    for row in dupes.iter_rows(named=True):
        issues.append(ValidationIssue(
            "ERROR", "duplicate_play_id",
            f"game {row['game_id']!r} play_id {row['play_id']} appears {row['n']} times",
            {"game_id": row["game_id"], "play_id": row["play_id"], "count": row["n"]},
        ))
    return issues


def validate_pbp_schedule_linkage(
    pbp_game_ids: set[str], scheduled_final_game_ids: set[str]
) -> list[ValidationIssue]:
    """`scheduled_final_game_ids` should be the set of game_ids from the normalized games
    table whose game_status == "final" (i.e. the game has actually been played, so
    play-by-play SHOULD exist for it) within the seasons this pbp fetch covers."""
    issues: list[ValidationIssue] = []

    orphaned_pbp = pbp_game_ids - scheduled_final_game_ids
    for game_id in sorted(orphaned_pbp):
        issues.append(ValidationIssue(
            "ERROR", "pbp_game_not_in_schedule",
            f"play-by-play references game_id {game_id!r} with no matching completed "
            "schedule game", {"game_id": game_id},
        ))

    missing_pbp = scheduled_final_game_ids - pbp_game_ids
    for game_id in sorted(missing_pbp):
        issues.append(ValidationIssue(
            "WARNING", "schedule_game_missing_pbp",
            f"completed schedule game {game_id!r} has no matching play-by-play rows",
            {"game_id": game_id},
        ))

    return issues


# ---------------------------------------------------------------------------
# Generic, reusable across datasets
# ---------------------------------------------------------------------------

def validate_critical_columns(dataset_name: str, df: pl.DataFrame, critical_columns: frozenset[str]) -> list[ValidationIssue]:
    missing = critical_columns - set(df.columns)
    if not missing:
        return []
    return [ValidationIssue(
        "FATAL", "missing_critical_columns",
        f"{dataset_name}: missing required columns {sorted(missing)}",
        {"dataset_name": dataset_name, "missing_columns": sorted(missing)},
    )]


def validate_schema_change(
    dataset_name: str, previous_columns: set[str] | None, current_columns: set[str]
) -> list[ValidationIssue]:
    if previous_columns is None:
        return []
    added = current_columns - previous_columns
    removed = previous_columns - current_columns
    issues: list[ValidationIssue] = []
    if added:
        issues.append(ValidationIssue(
            "WARNING", "schema_columns_added",
            f"{dataset_name}: {len(added)} column(s) not seen in the previous snapshot",
            {"columns": sorted(added)},
        ))
    if removed:
        issues.append(ValidationIssue(
            "WARNING", "schema_columns_removed",
            f"{dataset_name}: {len(removed)} column(s) present before are now missing",
            {"columns": sorted(removed)},
        ))
    return issues


def validate_row_count_anomaly(
    dataset_name: str, season: int | None, current_count: int, other_season_counts: list[int]
) -> list[ValidationIssue]:
    """Flags a season whose row count is far from the median of other seasons for the same
    dataset. Needs at least 3 other seasons to have a meaningful median; otherwise silent."""
    if len(other_season_counts) < 3 or current_count == 0:
        return []
    sorted_counts = sorted(other_season_counts)
    median = sorted_counts[len(sorted_counts) // 2]
    if median == 0:
        return []
    ratio = current_count / median
    if ratio < 0.5 or ratio > 1.8:
        return [ValidationIssue(
            "WARNING", "suspicious_row_count",
            f"{dataset_name} season={season}: row count {current_count} is {ratio:.2f}x the "
            f"median of other seasons ({median})",
            {"dataset_name": dataset_name, "season": season, "row_count": current_count, "median_other_seasons": median},
        )]
    return []
