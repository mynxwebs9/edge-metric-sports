"""Feature-build CLI.

    python -m nfl_predict.features.build --seasons 2010-2025
    python -m nfl_predict.features.build --seasons 2023-2025
    python -m nfl_predict.features.build --season 2025 --week 10   (diagnostic: prints the
        filtered rows for that game-week, computed with enough trailing context for the
        rolling windows to be real; does not write a feature-store partition, since
        partitions are whole-season and a single week isn't one)

Builds and writes BOTH the team-game and game-level Parquet partitions (feature_version
scoped, one partition per season - see store.py) for every season in `--seasons`. Each
season's write is independent and idempotent: re-running with the same feature_version and
config overwrites that season's partition with the same content (Parquet is not
append-only here; a partition is the CURRENT view for that version/season, matching how the
normalized `games`/`teams` tables in Phase 1 are upserted rather than accumulated).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.provenance import compute_schema_fingerprint
from nfl_predict.data.season_range import DEFAULT_HISTORICAL_SEASONS, parse_season_range
from nfl_predict.features.game_table import build_game_features
from nfl_predict.features.provenance import build_feature_manifest
from nfl_predict.features.store import write_feature_partition
from nfl_predict.features.team_game import build_team_game_features
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _feature_version() -> str:
    from nfl_predict.config import get_feature_engine_config
    return get_feature_engine_config()["feature_version"]


def build_and_write(seasons: list[int]) -> None:
    conn = get_connection()
    init_schema(conn)
    version = _feature_version()

    team_game = build_team_game_features(conn, seasons)
    game = build_game_features(team_game)
    conn.close()

    for season in seasons:
        tg_season = team_game.filter(pl.col("season") == season)
        g_season = game.filter(pl.col("season") == season)
        if tg_season.height == 0:
            logger.warning("no team-game rows for season; skipping partition write", extra={"season": season})
            continue

        tg_manifest = build_feature_manifest(
            tg_season.columns, {c: str(dt) for c, dt in zip(tg_season.columns, tg_season.dtypes)},
            tg_season.height, "team_game", [season], version, PROJECT_ROOT,
        )
        write_feature_partition(tg_season, "team_game", version, season, tg_manifest)

        g_manifest = build_feature_manifest(
            g_season.columns, {c: str(dt) for c, dt in zip(g_season.columns, g_season.dtypes)},
            g_season.height, "game", [season], version, PROJECT_ROOT,
        )
        write_feature_partition(g_season, "game", version, season, g_manifest)

        print(f"season={season}: wrote {tg_season.height} team-game rows, {g_season.height} game rows (feature_version={version})")


def print_single_week(season: int, week: int) -> None:
    conn = get_connection()
    init_schema(conn)
    context_seasons = [s for s in (season - 1, season) if s >= 1999]
    team_game = build_team_game_features(conn, context_seasons)
    conn.close()

    filtered = team_game.filter((pl.col("season") == season) & (pl.col("week") == week))
    if filtered.height == 0:
        print(f"No games found for season={season} week={week}.")
        return
    with pl.Config(tbl_cols=-1, tbl_width_chars=240, tbl_rows=-1):
        print(filtered.select(["game_id", "team_id", "opponent_id", "is_home", "as_of_timestamp", "off_epa_pp_season", "def_epa_pp_allowed_season", "games_played_current_season"]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nfl_predict.features.build")
    parser.add_argument("--seasons", default=None, help=f"Season range spec (default: {DEFAULT_HISTORICAL_SEASONS}).")
    parser.add_argument("--season", type=int, default=None, help="Single-season diagnostic mode (use with --week).")
    parser.add_argument("--week", type=int, default=None, help="Single-week diagnostic mode (use with --season).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.season is not None and args.week is not None:
        print_single_week(args.season, args.week)
        return 0

    seasons = parse_season_range(args.seasons or DEFAULT_HISTORICAL_SEASONS)
    build_and_write(seasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
