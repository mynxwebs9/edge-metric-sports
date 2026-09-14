"""Thin wrapper around nflreadpy's load_* functions.

This is the ONLY module that imports nflreadpy/Polars directly for the purpose of talking to
the network — see docs/ARCHITECTURE.md#tabular-data-representation-boundary. Everything past
`ingest_dataset_season` deals in already-fetched Polars DataFrames or the raw-store's
manifest objects, not in nflreadpy's API surface.

Coverage notes (from actually calling the API, not from trusting docstrings alone — see
docs/PHASE1_DATA_REPORT.md for the full findings):

- nflreadpy validates season ranges itself and raises ValueError for an out-of-range season
  (e.g. requesting snap_counts for 2011, when it only supports 2012+). Passing a LIST that
  mixes valid and invalid seasons raises for the whole call, not just the invalid entries —
  so this module always fetches one season at a time, never a batch list, both to isolate
  failures per season and because it matches the "partition raw storage by season" goal.
- A season inside the documented range can still return zero rows (observed for
  load_snap_counts(seasons=2012), which nflreadpy documents as available "since 2012" but
  which actually has no rows until 2013). This is NOT the same as an out-of-range error —
  it's recorded as a real, successful, empty snapshot, not skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import nflreadpy as nfl
import polars as pl

from nfl_predict.data.games import REQUIRED_RAW_COLUMNS as GAMES_REQUIRED_RAW_COLUMNS
from nfl_predict.data.teams import REQUIRED_RAW_COLUMNS as TEAMS_REQUIRED_RAW_COLUMNS

SOURCE_NAME = "nflverse"

NFLVERSE_DOC_URL = "https://nflreadr.nflverse.com/reference/{fn}.html"


class SeasonNotAvailableError(Exception):
    """Raised when nflreadpy itself reports a season is outside a dataset's supported range."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader_name: str  # matches the nflreadpy function name, for provenance/documentation
    fetch: Callable[[int | None], pl.DataFrame]
    takes_season: bool
    core: bool  # per PHASE 1 CORE DATASETS list vs. the "also investigate" optional list
    documented_min_season: int | None  # from nflreadpy's own docstring, not yet verified
    # Columns that must be present for the fetched frame to be usable at all - verified by
    # actually fetching one season of each dataset and inspecting real columns (not guessed
    # from documentation). Missing any of these is a FATAL validation issue.
    critical_columns: frozenset[str]


def _fetch_teams(_season: int | None) -> pl.DataFrame:
    return nfl.load_teams()


def _fetch_schedules(season: int | None) -> pl.DataFrame:
    return nfl.load_schedules(seasons=season)


def _fetch_pbp(season: int | None) -> pl.DataFrame:
    return nfl.load_pbp(seasons=season)


def _fetch_player_stats(season: int | None) -> pl.DataFrame:
    return nfl.load_player_stats(seasons=season, summary_level="week")


def _fetch_rosters(season: int | None) -> pl.DataFrame:
    return nfl.load_rosters(seasons=season)


def _fetch_rosters_weekly(season: int | None) -> pl.DataFrame:
    return nfl.load_rosters_weekly(seasons=season)


def _fetch_snap_counts(season: int | None) -> pl.DataFrame:
    return nfl.load_snap_counts(seasons=season)


def _fetch_participation(season: int | None) -> pl.DataFrame:
    return nfl.load_participation(seasons=season)


def _fetch_depth_charts(season: int | None) -> pl.DataFrame:
    return nfl.load_depth_charts(seasons=season)


def _fetch_nextgen(stat_type: str) -> Callable[[int | None], pl.DataFrame]:
    def _fetch(season: int | None) -> pl.DataFrame:
        return nfl.load_nextgen_stats(seasons=season, stat_type=stat_type)

    return _fetch


# CORE DATASETS (Phase 1 "at minimum" list). critical_columns verified by actually fetching
# a real season and inspecting the returned frame - see docs/PHASE1_DATA_REPORT.md.
#
# teams/schedules critical_columns are the SAME sets normalize_teams()/normalize_games()
# hard-require (imported from teams.py/games.py, not re-typed here) - this is what makes the
# generic pre-promotion FATAL gate in ingest.py authoritative: if it passes, the
# dataset-specific normalize_*() call downstream is guaranteed not to raise for a missing
# column. Keeping two independently-maintained column lists in sync by hand is exactly the
# kind of drift that would silently reopen that gap.
CORE_DATASET_SPECS: dict[str, DatasetSpec] = {
    "teams": DatasetSpec("teams", "load_teams", _fetch_teams, takes_season=False, core=True, documented_min_season=None,
                          critical_columns=frozenset(TEAMS_REQUIRED_RAW_COLUMNS)),
    "schedules": DatasetSpec("schedules", "load_schedules", _fetch_schedules, takes_season=True, core=True, documented_min_season=None,
                              critical_columns=frozenset(GAMES_REQUIRED_RAW_COLUMNS)),
    "pbp": DatasetSpec("pbp", "load_pbp", _fetch_pbp, takes_season=True, core=True, documented_min_season=1999,
                        critical_columns=frozenset({"game_id", "play_id", "season", "week"})),
    "player_stats": DatasetSpec("player_stats", "load_player_stats", _fetch_player_stats, takes_season=True, core=True, documented_min_season=None,
                                 critical_columns=frozenset({"player_id", "season", "week"})),
    "rosters": DatasetSpec("rosters", "load_rosters", _fetch_rosters, takes_season=True, core=True, documented_min_season=1920,
                            critical_columns=frozenset({"season", "team", "gsis_id"})),
}

# OPTIONAL DATASETS (Phase 1 "also investigate availability" list).
OPTIONAL_DATASET_SPECS: dict[str, DatasetSpec] = {
    "rosters_weekly": DatasetSpec("rosters_weekly", "load_rosters_weekly", _fetch_rosters_weekly, takes_season=True, core=False, documented_min_season=2002,
                                   critical_columns=frozenset({"season", "week", "team", "gsis_id"})),
    "snap_counts": DatasetSpec("snap_counts", "load_snap_counts", _fetch_snap_counts, takes_season=True, core=False, documented_min_season=2012,
                                critical_columns=frozenset({"game_id", "season", "week", "pfr_player_id", "team"})),
    # participation has no season/week columns of its own - it's keyed by nflverse_game_id +
    # play_id and must be joined against pbp/schedules to attach season/week. Documented in
    # docs/PHASE1_DATA_REPORT.md; not treated as a defect, just a real schema characteristic.
    "participation": DatasetSpec("participation", "load_participation", _fetch_participation, takes_season=True, core=False, documented_min_season=2016,
                                  critical_columns=frozenset({"nflverse_game_id", "play_id"})),
    "depth_charts": DatasetSpec("depth_charts", "load_depth_charts", _fetch_depth_charts, takes_season=True, core=False, documented_min_season=2001,
                                 critical_columns=frozenset({"season", "week", "gsis_id"})),
    "nextgen_passing": DatasetSpec("nextgen_passing", "load_nextgen_stats", _fetch_nextgen("passing"), takes_season=True, core=False, documented_min_season=2016,
                                    critical_columns=frozenset({"season", "week"})),
    "nextgen_receiving": DatasetSpec("nextgen_receiving", "load_nextgen_stats", _fetch_nextgen("receiving"), takes_season=True, core=False, documented_min_season=2016,
                                      critical_columns=frozenset({"season", "week"})),
    "nextgen_rushing": DatasetSpec("nextgen_rushing", "load_nextgen_stats", _fetch_nextgen("rushing"), takes_season=True, core=False, documented_min_season=2016,
                                    critical_columns=frozenset({"season", "week"})),
}

ALL_DATASET_SPECS: dict[str, DatasetSpec] = {**CORE_DATASET_SPECS, **OPTIONAL_DATASET_SPECS}


def fetch_dataset_season(spec: DatasetSpec, season: int | None) -> pl.DataFrame:
    """Fetch one dataset for one season (or the whole dataset, for season-less datasets).

    Raises SeasonNotAvailableError if nflreadpy itself rejects the season as out of range —
    callers should treat that as "not available for this season", not as an ingestion
    failure. Any other exception is a real error and propagates.
    """
    try:
        return spec.fetch(season)
    except ValueError as exc:
        if "must be between" in str(exc).lower() or "season" in str(exc).lower():
            raise SeasonNotAvailableError(str(exc)) from exc
        raise


def loader_package_version() -> str:
    return nfl.__version__
