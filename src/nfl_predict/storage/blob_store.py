"""Phase 10: a single key/value blob abstraction behind every module under `data_dir` that
persists immutable or append-only JSON/Parquet artifacts (research runs, content previews,
the decision log, the published-pick ledger, public model predictions, live odds snapshots,
and the event/game mapping index).

Per docs/ARCHITECTURE.md#storage, the Phase 10 swap must be mechanical, not a rewrite: every
one of those modules already follows the same shape (write bytes once at a deterministic key
and never overwrite them, or append bytes to a growing log) - this module gives that shape one
real interface with two backends, selected by `NFL_STORAGE_BACKEND`:

- `LocalFilesystemBlobStore` ("sqlite" - today's default): identical on-disk layout and
  behavior to what every module already did directly with `pathlib.Path`. Zero behavior
  change for local/dev use.
- `PostgresBlobStore` ("postgres"): the same keys/bytes, stored as rows in one `blobs` table
  in a hosted Postgres instance, for when the FastAPI read API is actually deployed and no
  longer has a durable local disk to read from.

Keys are `/`-separated strings mirroring the paths these modules already used (e.g.
`"research/season=2026/week=1/2026_01_DEN_KC/run_id=abc123/findings.json"`).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from nfl_predict.config import get_settings


class BlobStore(ABC):
    """Every method is byte-exact: callers own their own serialization (JSON, Parquet, ...)
    and hashing; this layer only ever moves bytes in and out by key."""

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Raises FileNotFoundError if `key` does not exist."""

    @abstractmethod
    def write_bytes(self, key: str, content: bytes, *, exist_ok: bool = False) -> None:
        """Raises FileExistsError if `key` already exists and `exist_ok` is False - the
        default, matching every caller's "never overwritten" immutability requirement."""

    @abstractmethod
    def append_bytes(self, key: str, content: bytes) -> None:
        """Creates `key` if absent, else appends to its existing content."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """Every existing key starting with `prefix`, sorted. `prefix` should end with "/"
        for a directory-style listing (every real call site uses it this way)."""


class LocalFilesystemBlobStore(BlobStore):
    def __init__(self) -> None:
        self._root = get_settings().data_dir

    def _path(self, key: str):
        return self._root / key

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, content: bytes, *, exist_ok: bool = False) -> None:
        path = self._path(key)
        if not exist_ok and path.is_file():
            raise FileExistsError(f"Blob already exists at key={key!r} - it is immutable and was not overwritten.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def append_bytes(self, key: str, content: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as f:
            f.write(content)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(
            str(p.relative_to(self._root)).replace("\\", "/")
            for p in base.rglob("*")
            if p.is_file()
        )


class PostgresBlobStore(BlobStore):
    """One `blobs` table holds every key this project persists. Simple by design (a single
    long-lived connection, no pooling) - appropriate for this project's read-heavy,
    low-concurrency reporting API, not a general-purpose high-throughput store.

    The database (or its connection pooler) closes idle connections, and a closed psycopg
    connection never recovers on its own - a real incident: after hours idle every API request
    failed with "the connection is closed" until the process was restarted. So this store
    reconnects: READS retry once on a fresh connection (idempotent, always safe); WRITES only
    reconnect BEFORE sending, when the connection is already known to be dead. A write is never
    blindly retried after an error, because an append whose reply was lost may already have
    happened, and repeating it would duplicate an immutable ledger entry."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS blobs (
        key TEXT PRIMARY KEY,
        content BYTEA NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    def __init__(self, database_url: str) -> None:
        import psycopg  # optional dependency - only imported when the postgres backend is actually selected

        self._psycopg = psycopg
        self._database_url = database_url
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        self._conn = self._psycopg.connect(self._database_url, autocommit=True)
        self._conn.execute(self._SCHEMA)

    def _ensure_connection(self) -> None:
        if self._conn.closed or self._conn.broken:
            with self._lock:
                if self._conn.closed or self._conn.broken:  # another thread may have just reconnected
                    self._connect()

    def _read(self, sql: str, params: tuple):
        """Executes a read; on a dropped connection, reconnects and retries exactly once."""
        self._ensure_connection()
        try:
            return self._conn.execute(sql, params)
        except self._psycopg.OperationalError:
            with self._lock:
                self._connect()
            return self._conn.execute(sql, params)

    def _write(self, sql: str, params: tuple):
        """Executes a write on a live connection - reconnecting only beforehand, never retrying."""
        self._ensure_connection()
        return self._conn.execute(sql, params)

    @staticmethod
    def _escape_like(prefix: str) -> str:
        return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def read_bytes(self, key: str) -> bytes:
        row = self._read("SELECT content FROM blobs WHERE key = %s", (key,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"No blob at key={key!r}")
        return bytes(row[0])

    def write_bytes(self, key: str, content: bytes, *, exist_ok: bool = False) -> None:
        if exist_ok:
            self._write(
                "INSERT INTO blobs (key, content, updated_at) VALUES (%s, %s, now()) "
                "ON CONFLICT (key) DO UPDATE SET content = EXCLUDED.content, updated_at = now()",
                (key, content),
            )
            return
        cur = self._write(
            "INSERT INTO blobs (key, content) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING RETURNING key",
            (key, content),
        )
        if cur.fetchone() is None:
            raise FileExistsError(f"Blob already exists at key={key!r} - it is immutable and was not overwritten.")

    def append_bytes(self, key: str, content: bytes) -> None:
        self._write(
            "INSERT INTO blobs (key, content) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET content = blobs.content || EXCLUDED.content, updated_at = now()",
            (key, content),
        )

    def exists(self, key: str) -> bool:
        row = self._read("SELECT 1 FROM blobs WHERE key = %s LIMIT 1", (key,)).fetchone()
        return row is not None

    def list_keys(self, prefix: str) -> list[str]:
        pattern = self._escape_like(prefix) + "%"
        rows = self._read(
            "SELECT key FROM blobs WHERE key LIKE %s ESCAPE '\\' ORDER BY key", (pattern,)
        ).fetchall()
        return [r[0] for r in rows]


# Keyed by database_url, not a bare no-arg lru_cache: `get_settings()` is itself cached and
# routinely monkeypatched wholesale in tests (a fresh data_dir per test), so this function
# must re-read settings on every call rather than memoizing its own result - a stale bare
# cache would silently keep pointing at a previous test's tmp_path. Postgres connections are
# still reused across calls for the same URL, which is the only part actually worth caching.
_postgres_stores: dict[str, PostgresBlobStore] = {}


def get_blob_store() -> BlobStore:
    settings = get_settings()
    if settings.storage_backend == "postgres":
        if not settings.database_url:
            raise ValueError(
                "NFL_STORAGE_BACKEND=postgres but NFL_DATABASE_URL is not set - a Postgres "
                "connection string is required to use the postgres backend."
            )
        if settings.database_url not in _postgres_stores:
            _postgres_stores[settings.database_url] = PostgresBlobStore(settings.database_url)
        return _postgres_stores[settings.database_url]
    return LocalFilesystemBlobStore()
