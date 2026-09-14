"""Repository-style interfaces over the SQLite normalized store.

Nothing outside this module (and db.py) runs raw SQL against normalized state — see
docs/ARCHITECTURE.md#storage. All writes are UPSERTs (INSERT OR REPLACE keyed by the primary
key), so re-running ingestion for the same games/teams is idempotent by construction: it
converges to the same rows rather than duplicating them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from nfl_predict.data.provenance import ProvenanceManifest
from nfl_predict.data.validation import ValidationIssue


class TeamsRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def upsert_teams(self, teams_rows: list[dict]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO teams
               (team_id, canonical_abbr, name, nickname, conference, division)
               VALUES (:team_id, :canonical_abbr, :name, :nickname, :conference, :division)""",
            teams_rows,
        )
        self._conn.commit()

    def upsert_abbr_aliases(self, alias_rows: list[dict]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO team_abbr_aliases (abbr, team_id) VALUES (:abbr, :team_id)",
            alias_rows,
        )
        self._conn.commit()

    def get_all(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM teams ORDER BY team_id").fetchall()

    def get_abbr_to_team_id(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT abbr, team_id FROM team_abbr_aliases").fetchall()
        return {row["abbr"]: row["team_id"] for row in rows}


class GamesRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def upsert_games(self, game_rows: list[dict], retrieval_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for row in game_rows:
            row["source_retrieval_id"] = retrieval_id
            row["normalized_at"] = now
        self._conn.executemany(
            """INSERT OR REPLACE INTO games
               (game_id, season, season_type, week, game_date, kickoff_time_naive,
                home_team_id, away_team_id, home_team_abbr, away_team_abbr,
                home_score, away_score, venue, game_status, source_retrieval_id, normalized_at)
               VALUES (:game_id, :season, :season_type, :week, :game_date, :kickoff_time_naive,
                :home_team_id, :away_team_id, :home_team_abbr, :away_team_abbr,
                :home_score, :away_score, :venue, :game_status, :source_retrieval_id, :normalized_at)""",
            game_rows,
        )
        self._conn.commit()

    def get_by_season(self, season: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM games WHERE season = ? ORDER BY week, game_id", (season,)
        ).fetchall()

    def get_game_ids(self, seasons: list[int] | None = None) -> set[str]:
        if seasons is None:
            rows = self._conn.execute("SELECT game_id FROM games").fetchall()
        else:
            placeholders = ",".join("?" for _ in seasons)
            rows = self._conn.execute(
                f"SELECT game_id FROM games WHERE season IN ({placeholders})", seasons
            ).fetchall()
        return {row["game_id"] for row in rows}

    def get_final_game_ids(self, seasons: list[int]) -> set[str]:
        placeholders = ",".join("?" for _ in seasons)
        rows = self._conn.execute(
            f"SELECT game_id FROM games WHERE season IN ({placeholders}) AND game_status = 'final'",
            seasons,
        ).fetchall()
        return {row["game_id"] for row in rows}

    def count_by_season(self) -> dict[int, int]:
        rows = self._conn.execute(
            "SELECT season, COUNT(*) AS n FROM games GROUP BY season"
        ).fetchall()
        return {row["season"]: row["n"] for row in rows}


class ManifestsRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def record(self, manifest: ProvenanceManifest) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO ingestion_manifests
               (retrieval_id, source_name, dataset_name, requested_seasons, loader_function,
                loader_package_version, retrieved_at, source_identifier, source_release_identifier,
                raw_file_path, content_sha256, row_count, column_count, column_names,
                schema_fingerprint, duplicate_of_retrieval_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                manifest.retrieval_id, manifest.source_name, manifest.dataset_name,
                json.dumps(manifest.requested_seasons), manifest.loader_function,
                manifest.loader_package_version, manifest.retrieved_at, manifest.source_identifier,
                manifest.source_release_identifier, manifest.raw_file_path, manifest.content_sha256,
                manifest.row_count, manifest.column_count, json.dumps(manifest.column_names),
                manifest.schema_fingerprint, manifest.duplicate_of_retrieval_id,
            ),
        )
        self._conn.commit()

    def by_dataset(self, source_name: str, dataset_name: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT * FROM ingestion_manifests WHERE source_name = ? AND dataset_name = ?
               ORDER BY retrieved_at""",
            (source_name, dataset_name),
        ).fetchall()

    def get_latest_canonical_manifest(
        self, source_name: str, dataset_name: str, season: int
    ) -> sqlite3.Row | None:
        """The newest non-duplicate ("canonical") manifest for a single-season request.

        "Newest" means by `retrieved_at`, read from the database - never filesystem
        directory-listing order, which reflects write order, not necessarily chronological
        fetch order (a full multi-season backfill writes every season's snapshot before any
        of them get validated). A dataset/season can have more than one canonical (i.e.
        non-duplicate) manifest if nflverse corrected historical data upstream and it was
        re-fetched - that re-fetch produces a new, different-hash snapshot rather than a
        `duplicate_of_retrieval_id` one, and this method is what picks the authoritative one
        for anything that needs "the" current state of that season (e.g. season-over-season
        schema comparison in `ingest.py`).

        Returns None if no canonical manifest exists for that exact `[season]` request.
        """
        rows = self._conn.execute(
            """SELECT * FROM ingestion_manifests
               WHERE source_name = ? AND dataset_name = ? AND duplicate_of_retrieval_id IS NULL
               ORDER BY retrieved_at DESC""",
            (source_name, dataset_name),
        ).fetchall()
        for row in rows:
            if json.loads(row["requested_seasons"]) == [season]:
                return row
        return None


class ValidationRepository:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def record(
        self, dataset_name: str, season: int | None, retrieval_id: str | None,
        issues: list[ValidationIssue],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (retrieval_id, dataset_name, season, issue.level, issue.code, issue.message,
             json.dumps(issue.context), now)
            for issue in issues
        ]
        self._conn.executemany(
            """INSERT INTO validation_issues
               (retrieval_id, dataset_name, season, level, code, message, context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def by_dataset(self, dataset_name: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM validation_issues WHERE dataset_name = ? ORDER BY created_at",
            (dataset_name,),
        ).fetchall()

    def counts_by_level(self, dataset_name: str | None = None) -> dict[str, int]:
        if dataset_name is None:
            rows = self._conn.execute(
                "SELECT level, COUNT(*) AS n FROM validation_issues GROUP BY level"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT level, COUNT(*) AS n FROM validation_issues WHERE dataset_name = ? GROUP BY level",
                (dataset_name,),
            ).fetchall()
        return {row["level"]: row["n"] for row in rows}
