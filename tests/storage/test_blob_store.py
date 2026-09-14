"""Phase 10: `LocalFilesystemBlobStore` is the default, always-exercised backend - every
storage module that was refactored onto `BlobStore` (research, content, decisions, the pick
ledger, public predictions, live odds snapshots, the event/game mapping index) relies on
these exact semantics. `PostgresBlobStore` is covered separately in
`test_postgres_blob_store.py`, gated behind a real reachable Postgres instance since this
sandbox has none available.
"""

from __future__ import annotations

import pytest

from nfl_predict.storage.blob_store import LocalFilesystemBlobStore, get_blob_store


def test_write_then_read_round_trips(isolated_data_dir):
    store = get_blob_store()
    assert isinstance(store, LocalFilesystemBlobStore)
    store.write_bytes("a/b/c.json", b'{"x": 1}')
    assert store.read_bytes("a/b/c.json") == b'{"x": 1}'


def test_read_missing_key_raises_file_not_found(isolated_data_dir):
    store = get_blob_store()
    with pytest.raises(FileNotFoundError):
        store.read_bytes("nope.json")


def test_write_without_exist_ok_refuses_to_overwrite(isolated_data_dir):
    store = get_blob_store()
    store.write_bytes("k.json", b"first")
    with pytest.raises(FileExistsError):
        store.write_bytes("k.json", b"second")
    assert store.read_bytes("k.json") == b"first"  # untouched


def test_write_with_exist_ok_overwrites(isolated_data_dir):
    store = get_blob_store()
    store.write_bytes("k.json", b"first")
    store.write_bytes("k.json", b"second", exist_ok=True)
    assert store.read_bytes("k.json") == b"second"


def test_append_creates_then_appends(isolated_data_dir):
    store = get_blob_store()
    store.append_bytes("log.jsonl", b"line1\n")
    store.append_bytes("log.jsonl", b"line2\n")
    assert store.read_bytes("log.jsonl") == b"line1\nline2\n"


def test_exists_reflects_real_state(isolated_data_dir):
    store = get_blob_store()
    assert store.exists("k.json") is False
    store.write_bytes("k.json", b"x")
    assert store.exists("k.json") is True


def test_list_keys_returns_only_keys_under_the_prefix_sorted(isolated_data_dir):
    store = get_blob_store()
    store.write_bytes("research/g1/run_id=b/findings.json", b"{}")
    store.write_bytes("research/g1/run_id=a/findings.json", b"{}")
    store.write_bytes("research/g2/run_id=z/findings.json", b"{}")  # different game - must not appear

    keys = store.list_keys("research/g1/")
    assert keys == sorted(keys)
    assert keys == [
        "research/g1/run_id=a/findings.json",
        "research/g1/run_id=b/findings.json",
    ]


def test_list_keys_on_a_prefix_with_no_data_returns_empty(isolated_data_dir):
    store = get_blob_store()
    assert store.list_keys("nothing/here/") == []


def test_get_blob_store_reflects_a_data_dir_change_not_a_stale_cache(isolated_data_dir, monkeypatch, tmp_path):
    """Regression: `get_blob_store()` used to be a bare `lru_cache()`, which returned a
    store pointing at whichever data_dir was active the FIRST time it was called in the
    process - silently wrong for the second of two tests (or two tenants) that each expect
    their own isolated directory. It must re-read settings on every call."""
    store_a = get_blob_store()
    store_a.write_bytes("k.json", b"in-a")

    other_dir = tmp_path.parent / (tmp_path.name + "-other")
    monkeypatch.setenv("NFL_DATA_DIR", str(other_dir))
    from nfl_predict.config import get_settings
    get_settings.cache_clear()

    store_b = get_blob_store()
    assert store_b.exists("k.json") is False  # a fresh directory, not store_a's data
    store_b.write_bytes("k.json", b"in-b")
    assert store_b.read_bytes("k.json") == b"in-b"
    assert store_a.read_bytes("k.json") == b"in-a"  # original directory untouched
