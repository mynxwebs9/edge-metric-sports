"""Relational/normalized state: SQLite locally, Postgres when hosted (Phase 10).

Per docs/ARCHITECTURE.md#storage: this holds relational/tabular state (teams, games,
manifests, validation results); large analytical data (play-by-play) stays in Parquet. This
module is the only place that opens the database connection or runs DDL — everything else
(`src/nfl_predict/data/repositories.py`, `src/nfl_predict/live/schedule_provider.py`) issues
`?`-placeholder or `:name`-placeholder SQL against whatever `get_connection()` returns, and
never needs to know which backend is actually active.

Backend is selected by `NFL_STORAGE_BACKEND` (`sqlite`, the default, or `postgres`), read via
`nfl_predict.config.get_settings()`. Under `postgres`, `get_connection()` returns a
`_PostgresConnectionAdapter` that speaks the same `?` / `:name` placeholder style and the
same `row["column"]` access as `sqlite3.Row` — so every existing call site works completely
unchanged. The one thing that cannot be made backend-agnostic by a placeholder rewrite is
SQLite's `INSERT OR REPLACE` upsert syntax, which has no Postgres equivalent; those specific,
known statements are translated via an explicit, whitespace-normalized lookup table (see
`_INSERT_OR_REPLACE_TRANSLATIONS` below) rather than a general SQL parser, so an
unrecognized "INSERT OR REPLACE" fails loudly instead of being silently mis-translated.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from nfl_predict.config import get_settings

DB_FILENAME = "nfl_predict.sqlite"

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    canonical_abbr TEXT NOT NULL,
    name TEXT NOT NULL,
    nickname TEXT,
    conference TEXT,
    division TEXT
);

CREATE TABLE IF NOT EXISTS team_abbr_aliases (
    abbr TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    week INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    kickoff_time_naive TEXT,
    home_team_id TEXT,
    away_team_id TEXT,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    home_score INTEGER,
    away_score INTEGER,
    venue TEXT,
    game_status TEXT NOT NULL,
    source_retrieval_id TEXT,
    normalized_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);

CREATE TABLE IF NOT EXISTS ingestion_manifests (
    retrieval_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    requested_seasons TEXT NOT NULL,
    loader_function TEXT NOT NULL,
    loader_package_version TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_release_identifier TEXT,
    raw_file_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    column_names TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    duplicate_of_retrieval_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_manifests_dataset ON ingestion_manifests(source_name, dataset_name);

CREATE TABLE IF NOT EXISTS validation_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    retrieval_id TEXT,
    dataset_name TEXT NOT NULL,
    season INTEGER,
    level TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    context TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_validation_dataset ON validation_issues(dataset_name, season);
"""

# Same tables, Postgres dialect: SERIAL replaces INTEGER PRIMARY KEY AUTOINCREMENT; everything
# else is already portable ANSI-ish SQL.
POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    canonical_abbr TEXT NOT NULL,
    name TEXT NOT NULL,
    nickname TEXT,
    conference TEXT,
    division TEXT
);

CREATE TABLE IF NOT EXISTS team_abbr_aliases (
    abbr TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    week INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    kickoff_time_naive TEXT,
    home_team_id TEXT,
    away_team_id TEXT,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    home_score INTEGER,
    away_score INTEGER,
    venue TEXT,
    game_status TEXT NOT NULL,
    source_retrieval_id TEXT,
    normalized_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);

CREATE TABLE IF NOT EXISTS ingestion_manifests (
    retrieval_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    requested_seasons TEXT NOT NULL,
    loader_function TEXT NOT NULL,
    loader_package_version TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_release_identifier TEXT,
    raw_file_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    column_names TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    duplicate_of_retrieval_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_manifests_dataset ON ingestion_manifests(source_name, dataset_name);

CREATE TABLE IF NOT EXISTS validation_issues (
    id SERIAL PRIMARY KEY,
    retrieval_id TEXT,
    dataset_name TEXT NOT NULL,
    season INTEGER,
    level TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    context TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_validation_dataset ON validation_issues(dataset_name, season);
"""

# SQLite's "INSERT OR REPLACE" has no Postgres equivalent - these are the project's only
# three call sites (src/nfl_predict/data/repositories.py), translated explicitly rather than
# via a general SQL parser. Keys are whitespace-normalized (collapsed to single spaces) so
# formatting changes to the source don't silently break the lookup; an un-translated
# "INSERT OR REPLACE" raises rather than being sent to Postgres as invalid SQL.
_INSERT_OR_REPLACE_TRANSLATIONS: dict[str, str] = {
    " ".join("""INSERT OR REPLACE INTO teams
               (team_id, canonical_abbr, name, nickname, conference, division)
               VALUES (:team_id, :canonical_abbr, :name, :nickname, :conference, :division)""".split()):
        " ".join("""INSERT INTO teams (team_id, canonical_abbr, name, nickname, conference, division)
               VALUES (:team_id, :canonical_abbr, :name, :nickname, :conference, :division)
               ON CONFLICT (team_id) DO UPDATE SET
                 canonical_abbr = EXCLUDED.canonical_abbr, name = EXCLUDED.name,
                 nickname = EXCLUDED.nickname, conference = EXCLUDED.conference,
                 division = EXCLUDED.division""".split()),
    " ".join("""INSERT OR REPLACE INTO team_abbr_aliases (abbr, team_id) VALUES (:abbr, :team_id)""".split()):
        " ".join("""INSERT INTO team_abbr_aliases (abbr, team_id) VALUES (:abbr, :team_id)
               ON CONFLICT (abbr) DO UPDATE SET team_id = EXCLUDED.team_id""".split()),
    " ".join("""INSERT OR REPLACE INTO games
               (game_id, season, season_type, week, game_date, kickoff_time_naive,
                home_team_id, away_team_id, home_team_abbr, away_team_abbr,
                home_score, away_score, venue, game_status, source_retrieval_id, normalized_at)
               VALUES (:game_id, :season, :season_type, :week, :game_date, :kickoff_time_naive,
                :home_team_id, :away_team_id, :home_team_abbr, :away_team_abbr,
                :home_score, :away_score, :venue, :game_status, :source_retrieval_id, :normalized_at)""".split()):
        " ".join("""INSERT INTO games
               (game_id, season, season_type, week, game_date, kickoff_time_naive,
                home_team_id, away_team_id, home_team_abbr, away_team_abbr,
                home_score, away_score, venue, game_status, source_retrieval_id, normalized_at)
               VALUES (:game_id, :season, :season_type, :week, :game_date, :kickoff_time_naive,
                :home_team_id, :away_team_id, :home_team_abbr, :away_team_abbr,
                :home_score, :away_score, :venue, :game_status, :source_retrieval_id, :normalized_at)
               ON CONFLICT (game_id) DO UPDATE SET
                 season = EXCLUDED.season, season_type = EXCLUDED.season_type, week = EXCLUDED.week,
                 game_date = EXCLUDED.game_date, kickoff_time_naive = EXCLUDED.kickoff_time_naive,
                 home_team_id = EXCLUDED.home_team_id, away_team_id = EXCLUDED.away_team_id,
                 home_team_abbr = EXCLUDED.home_team_abbr, away_team_abbr = EXCLUDED.away_team_abbr,
                 home_score = EXCLUDED.home_score, away_score = EXCLUDED.away_score,
                 venue = EXCLUDED.venue, game_status = EXCLUDED.game_status,
                 source_retrieval_id = EXCLUDED.source_retrieval_id,
                 normalized_at = EXCLUDED.normalized_at""".split()),
    " ".join("""INSERT OR REPLACE INTO ingestion_manifests
               (retrieval_id, source_name, dataset_name, requested_seasons, loader_function,
                loader_package_version, retrieved_at, source_identifier, source_release_identifier,
                raw_file_path, content_sha256, row_count, column_count, column_names,
                schema_fingerprint, duplicate_of_retrieval_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""".split()):
        " ".join("""INSERT INTO ingestion_manifests
               (retrieval_id, source_name, dataset_name, requested_seasons, loader_function,
                loader_package_version, retrieved_at, source_identifier, source_release_identifier,
                raw_file_path, content_sha256, row_count, column_count, column_names,
                schema_fingerprint, duplicate_of_retrieval_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (retrieval_id) DO UPDATE SET
                 source_name = EXCLUDED.source_name, dataset_name = EXCLUDED.dataset_name,
                 requested_seasons = EXCLUDED.requested_seasons, loader_function = EXCLUDED.loader_function,
                 loader_package_version = EXCLUDED.loader_package_version, retrieved_at = EXCLUDED.retrieved_at,
                 source_identifier = EXCLUDED.source_identifier,
                 source_release_identifier = EXCLUDED.source_release_identifier,
                 raw_file_path = EXCLUDED.raw_file_path, content_sha256 = EXCLUDED.content_sha256,
                 row_count = EXCLUDED.row_count, column_count = EXCLUDED.column_count,
                 column_names = EXCLUDED.column_names, schema_fingerprint = EXCLUDED.schema_fingerprint,
                 duplicate_of_retrieval_id = EXCLUDED.duplicate_of_retrieval_id""".split()),
}

_NAMED_PARAM_RE = re.compile(r":(\w+)")


class UnsupportedPostgresStatementError(Exception):
    """Raised when a SQLite-only statement shape (currently, only an unrecognized
    "INSERT OR REPLACE") is executed against the Postgres backend and has no registered
    translation - fails loudly rather than sending invalid SQL to the server."""


def _translate_for_postgres(sql: str) -> str:
    normalized = " ".join(sql.split())
    if normalized.upper().startswith("INSERT OR REPLACE"):
        translated = _INSERT_OR_REPLACE_TRANSLATIONS.get(normalized)
        if translated is None:
            raise UnsupportedPostgresStatementError(
                f"No registered Postgres translation for this INSERT OR REPLACE statement: {normalized!r}. "
                "Add it to _INSERT_OR_REPLACE_TRANSLATIONS in nfl_predict/data/db.py."
            )
        normalized = translated
    normalized = _NAMED_PARAM_RE.sub(r"%(\1)s", normalized)
    normalized = normalized.replace("?", "%s")
    return normalized


class _PostgresCursorAdapter:
    """Wraps a psycopg cursor so `.fetchone()`/`.fetchall()` rows support `row["column"]`
    access, matching `sqlite3.Row` (every existing call site relies on this)."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def fetchone(self) -> dict | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[dict]:
        return self._cursor.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class _PostgresConnectionAdapter:
    """Speaks `sqlite3.Connection`'s subset of the DB-API used by this codebase
    (`.execute`, `.executemany`, `.commit`, `.close`) against a real psycopg connection, with
    `?` / `:name` placeholders and `INSERT OR REPLACE` transparently translated to Postgres
    equivalents. See module docstring."""

    def __init__(self, database_url: str) -> None:
        import psycopg
        from psycopg.rows import dict_row

        self._conn = psycopg.connect(database_url, row_factory=dict_row)

    def execute(self, sql: str, params: Any = ()) -> _PostgresCursorAdapter:
        translated = _translate_for_postgres(sql)
        cur = self._conn.cursor()
        cur.execute(translated, params)
        return _PostgresCursorAdapter(cur)

    def executemany(self, sql: str, seq_of_params: Any) -> None:
        translated = _translate_for_postgres(sql)
        with self._conn.cursor() as cur:
            cur.executemany(translated, list(seq_of_params))

    def executescript(self, script: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(script)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def db_path() -> Path:
    return get_settings().data_dir / DB_FILENAME


def get_connection() -> sqlite3.Connection | _PostgresConnectionAdapter:
    settings = get_settings()
    if settings.storage_backend == "postgres":
        if not settings.database_url:
            raise ValueError(
                "NFL_STORAGE_BACKEND=postgres but NFL_DATABASE_URL is not set - a Postgres "
                "connection string is required to use the postgres backend."
            )
        return _PostgresConnectionAdapter(settings.database_url)
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection | _PostgresConnectionAdapter) -> None:
    if isinstance(conn, _PostgresConnectionAdapter):
        conn.executescript(POSTGRES_SCHEMA)
    else:
        conn.executescript(SQLITE_SCHEMA)
    conn.commit()
