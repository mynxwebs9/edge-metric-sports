"""Normalize a raw snapshot, validate it, and promote it to normalized storage.

"Promotion" means writing to the SQLite normalized tables (teams, games) or the normalized
Parquet store (play-by-play). A FATAL validation issue blocks promotion for that snapshot;
the raw snapshot itself is never affected either way - it was already written immutably
before this runs. See docs/PHASE1_DATA_REPORT.md for what was actually found running this
against real data.
"""

from __future__ import annotations

import sqlite3

import polars as pl

from nfl_predict.data.games import normalize_games
from nfl_predict.data.pbp_store import normalize_and_write_pbp
from nfl_predict.data.repositories import GamesRepository, TeamsRepository, ValidationRepository
from nfl_predict.data.teams import normalize_teams
from nfl_predict.data.validation import (
    ValidationIssue,
    is_promotable,
    validate_normalized_games,
    validate_pbp,
    validate_pbp_schedule_linkage,
    validate_raw_teams,
)
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)


def promote_teams(raw_teams: pl.DataFrame, retrieval_id: str, conn: sqlite3.Connection) -> list[ValidationIssue]:
    issues = validate_raw_teams(raw_teams)
    ValidationRepository(conn).record("teams", None, retrieval_id, issues)
    if not is_promotable(issues):
        logger.error("teams snapshot has FATAL issues; not promoted", extra={"retrieval_id": retrieval_id})
        return issues

    current_abbrs = set(raw_teams["team_abbr"].to_list())
    normalized = normalize_teams(raw_teams, current_abbrs=current_abbrs)
    repo = TeamsRepository(conn)
    repo.upsert_teams(normalized.teams.to_dicts())
    repo.upsert_abbr_aliases(normalized.abbr_aliases.to_dicts())
    logger.info(
        "promoted teams", extra={"retrieval_id": retrieval_id, "n_teams": normalized.teams.height, "n_aliases": normalized.abbr_aliases.height},
    )
    return issues


def promote_games(raw_schedules: pl.DataFrame, retrieval_id: str, conn: sqlite3.Connection) -> list[ValidationIssue]:
    abbr_to_team_id = TeamsRepository(conn).get_abbr_to_team_id()
    if not abbr_to_team_id:
        logger.warning(
            "no team_abbr aliases in the database yet; games will normalize with unresolved team_ids",
            extra={"retrieval_id": retrieval_id},
        )

    normalized_games = normalize_games(raw_schedules, abbr_to_team_id)
    issues = validate_normalized_games(normalized_games)
    seasons_present = sorted(normalized_games["season"].unique().to_list())
    season = seasons_present[0] if len(seasons_present) == 1 else None
    ValidationRepository(conn).record("games", season, retrieval_id, issues)
    if not is_promotable(issues):
        logger.error("games snapshot has FATAL issues; not promoted", extra={"retrieval_id": retrieval_id})
        return issues

    GamesRepository(conn).upsert_games(normalized_games.to_dicts(), retrieval_id)
    logger.info(
        "promoted games", extra={"retrieval_id": retrieval_id, "n_games": normalized_games.height, "seasons": seasons_present},
    )
    return issues


def promote_pbp(raw_pbp: pl.DataFrame, season: int, retrieval_id: str, conn: sqlite3.Connection) -> list[ValidationIssue]:
    issues = validate_pbp(raw_pbp)

    abbr_to_team_id = TeamsRepository(conn).get_abbr_to_team_id()
    games_repo = GamesRepository(conn)
    final_game_ids = games_repo.get_final_game_ids([season])
    pbp_game_ids = set(raw_pbp["game_id"].drop_nulls().unique().to_list()) if "game_id" in raw_pbp.columns else set()
    issues += validate_pbp_schedule_linkage(pbp_game_ids, final_game_ids)

    ValidationRepository(conn).record("pbp", season, retrieval_id, issues)
    if not is_promotable(issues):
        logger.error("pbp snapshot has FATAL issues; not promoted", extra={"retrieval_id": retrieval_id, "season": season})
        return issues

    path = normalize_and_write_pbp(raw_pbp, season, abbr_to_team_id)
    logger.info(
        "promoted pbp", extra={"retrieval_id": retrieval_id, "season": season, "n_rows": raw_pbp.height, "path": str(path)},
    )
    return issues
