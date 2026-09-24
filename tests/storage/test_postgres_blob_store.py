"""Phase 10: `PostgresBlobStore` against a REAL Postgres instance - not mocked, since a mock
would only prove the mock is self-consistent, not that the real SQL is correct.

Skipped unless `NFL_TEST_POSTGRES_URL` points at a real, reachable, disposable Postgres
database (e.g. a local `docker run postgres`, or a free Neon/Supabase instance) - this
sandbox has no Docker daemon available, so these tests have not been run here. Run them once
a real instance is available before relying on the postgres backend for a real deployment:

    NFL_TEST_POSTGRES_URL=postgresql://user:pass@host:5432/dbname pytest tests/storage/test_postgres_blob_store.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NFL_TEST_POSTGRES_URL"),
    reason="Set NFL_TEST_POSTGRES_URL to a real, disposable Postgres database to run this - see module docstring.",
)


@pytest.fixture
def store():
    from nfl_predict.storage.blob_store import PostgresBlobStore

    # A per-test key prefix (not a per-test schema/database) keeps this simple while still
    # never colliding with another test's rows in the same shared test database.
    prefix = f"test-run-{uuid.uuid4().hex}/"
    s = PostgresBlobStore(os.environ["NFL_TEST_POSTGRES_URL"])
    yield s, prefix
    for key in s.list_keys(prefix):
        s._conn.execute("DELETE FROM blobs WHERE key = %s", (key,))
    s._conn.commit()


def test_write_then_read_round_trips(store):
    s, prefix = store
    s.write_bytes(prefix + "a.json", b'{"x": 1}')
    assert s.read_bytes(prefix + "a.json") == b'{"x": 1}'


def test_read_missing_key_raises_file_not_found(store):
    s, prefix = store
    with pytest.raises(FileNotFoundError):
        s.read_bytes(prefix + "nope.json")


def test_write_without_exist_ok_refuses_to_overwrite(store):
    s, prefix = store
    s.write_bytes(prefix + "k.json", b"first")
    with pytest.raises(FileExistsError):
        s.write_bytes(prefix + "k.json", b"second")
    assert s.read_bytes(prefix + "k.json") == b"first"


def test_write_with_exist_ok_overwrites(store):
    s, prefix = store
    s.write_bytes(prefix + "k.json", b"first")
    s.write_bytes(prefix + "k.json", b"second", exist_ok=True)
    assert s.read_bytes(prefix + "k.json") == b"second"


def test_append_creates_then_appends(store):
    s, prefix = store
    s.append_bytes(prefix + "log.jsonl", b"line1\n")
    s.append_bytes(prefix + "log.jsonl", b"line2\n")
    assert s.read_bytes(prefix + "log.jsonl") == b"line1\nline2\n"


def test_list_keys_respects_prefix_and_sql_like_escaping(store):
    s, prefix = store
    # A literal "%" in a key must not act as a SQL LIKE wildcard.
    s.write_bytes(prefix + "50%off/a.json", b"{}")
    s.write_bytes(prefix + "other/b.json", b"{}")
    assert s.list_keys(prefix + "50%off/") == [prefix + "50%off/a.json"]


# ---------------------------------------------------------------- dropped connections
# A real incident: the database closes idle connections, and a closed psycopg connection never
# recovers by itself - after hours idle every API request failed with "the connection is
# closed" until the process was restarted.

def _terminate_from_the_server_side(store_obj):
    import psycopg

    pid = store_obj._conn.info.backend_pid
    with psycopg.connect(os.environ["NFL_TEST_POSTGRES_URL"], autocommit=True) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (pid,))


def test_a_read_after_the_connection_was_closed_reconnects(store):
    s, prefix = store
    s.write_bytes(prefix + "a.json", b"hello")
    s._conn.close()

    assert s.read_bytes(prefix + "a.json") == b"hello"
    assert s.exists(prefix + "a.json") is True


def test_a_read_after_the_server_killed_the_connection_retries_once_on_a_fresh_one(store):
    s, prefix = store
    s.write_bytes(prefix + "a.json", b"hello")
    _terminate_from_the_server_side(s)  # the client doesn't know yet - the first use will find out

    assert s.read_bytes(prefix + "a.json") == b"hello"
    assert s.list_keys(prefix) == [prefix + "a.json"]


def test_a_write_after_the_connection_was_closed_reconnects_before_sending_and_lands_exactly_once(store):
    s, prefix = store
    s.append_bytes(prefix + "log.jsonl", b"one\n")
    s._conn.close()

    s.append_bytes(prefix + "log.jsonl", b"two\n")

    assert s.read_bytes(prefix + "log.jsonl") == b"one\ntwo\n"


def test_a_write_is_never_blindly_retried_after_the_server_kills_the_connection(store):
    """If the connection dies as the write is sent, the caller must find out - repeating an
    append whose reply was lost could duplicate an immutable ledger entry. The NEXT call then
    reconnects normally."""
    import psycopg

    s, prefix = store
    s.append_bytes(prefix + "log.jsonl", b"one\n")
    _terminate_from_the_server_side(s)

    with pytest.raises(psycopg.OperationalError):
        s.append_bytes(prefix + "log.jsonl", b"two\n")

    s.append_bytes(prefix + "log.jsonl", b"three\n")
    assert s.read_bytes(prefix + "log.jsonl") == b"one\nthree\n"  # 'two' was never applied, and nothing was doubled
