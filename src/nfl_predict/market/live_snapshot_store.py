"""Phase 5 Step 22: append-only timestamped LIVE odds snapshot storage.

Never overwrites: "BUF -3 at 10:00 AM" must not be replaced by "BUF -4 at 12:00 PM" - both
are stored as separate rows, keyed by `(provider_event_id, bookmaker, fetched_at)`. A second
`append_live_snapshot` call for that exact key raises rather than silently deduplicating or
replacing, so a bug that re-fetches the same instant twice is caught rather than hidden.
This is what lets a future live-operation phase reconstruct opening/pregame/closing
movement and compute real CLV (`clv.py`) once actual multiple snapshots exist for a game -
today, nothing calls this in production (no live prediction pipeline exists yet); it is
exercised only by tests against synthetic snapshots.
"""

from __future__ import annotations

import dataclasses
import io

import polars as pl

from nfl_predict.market.odds_provider import OddsMarketSnapshot
from nfl_predict.storage.blob_store import get_blob_store

LIVE_DIRNAME = "market/live_snapshots"
_SNAPSHOT_FIELDS = [f.name for f in dataclasses.fields(OddsMarketSnapshot)]


class DuplicateSnapshotError(Exception):
    """Raised when a snapshot with the same (provider_event_id, bookmaker, fetched_at) key
    already exists - live snapshots are append-only, never overwritten."""


def _live_snapshot_key(provider_event_id: str) -> str:
    return f"{LIVE_DIRNAME}/event={provider_event_id}/snapshots.parquet"


def _read_parquet_blob(store, key: str) -> pl.DataFrame | None:
    if not store.exists(key):
        return None
    return pl.read_parquet(io.BytesIO(store.read_bytes(key)))


def _write_parquet_blob(store, key: str, df: pl.DataFrame) -> None:
    buffer = io.BytesIO()
    df.write_parquet(buffer)
    store.write_bytes(key, buffer.getvalue(), exist_ok=True)


def append_live_snapshot(snapshot: OddsMarketSnapshot) -> str:
    store = get_blob_store()
    key = _live_snapshot_key(snapshot.provider_event_id)
    new_row = pl.DataFrame([dataclasses.asdict(snapshot)])

    existing = _read_parquet_blob(store, key)
    if existing is not None:
        duplicate = existing.filter(
            (pl.col("bookmaker") == snapshot.bookmaker) & (pl.col("fetched_at") == snapshot.fetched_at)
        )
        if duplicate.height > 0:
            raise DuplicateSnapshotError(
                f"A snapshot already exists for event={snapshot.provider_event_id!r} "
                f"bookmaker={snapshot.bookmaker!r} fetched_at={snapshot.fetched_at!r} - "
                "live snapshots are append-only and are never overwritten."
            )
        combined = pl.concat([existing, new_row], how="diagonal_relaxed")
    else:
        combined = new_row

    _write_parquet_blob(store, key, combined)
    return key


def append_live_snapshot_idempotent(snapshot: OddsMarketSnapshot) -> tuple[str, bool]:
    """Phase 8A Step 9: like `append_live_snapshot`, but re-submitting the EXACT SAME
    snapshot (same key AND same content) is a safe no-op instead of raising
    `DuplicateSnapshotError` - a genuine content change at the same key still raises,
    since that would silently discard information rather than record a real update.
    Returns `(key, was_appended)`."""
    store = get_blob_store()
    key = _live_snapshot_key(snapshot.provider_event_id)
    existing = _read_parquet_blob(store, key)
    if existing is not None:
        duplicate = existing.filter(
            (pl.col("bookmaker") == snapshot.bookmaker) & (pl.col("fetched_at") == snapshot.fetched_at)
        )
        if duplicate.height > 0:
            existing_row = duplicate.row(0, named=True)
            new_row_dict = dataclasses.asdict(snapshot)
            if all(existing_row.get(k) == v for k, v in new_row_dict.items()):
                return key, False  # identical content already recorded - no-op
            raise DuplicateSnapshotError(
                f"A DIFFERENT snapshot already exists for event={snapshot.provider_event_id!r} "
                f"bookmaker={snapshot.bookmaker!r} fetched_at={snapshot.fetched_at!r} - live "
                "snapshots are append-only; this is a genuine conflicting update, not a "
                "harmless re-submission, so it is refused rather than silently discarded."
            )
    appended_key = append_live_snapshot(snapshot)
    return appended_key, True


def read_live_snapshots(provider_event_id: str) -> pl.DataFrame:
    store = get_blob_store()
    key = _live_snapshot_key(provider_event_id)
    existing = _read_parquet_blob(store, key)
    if existing is None:
        return pl.DataFrame(schema={c: pl.Utf8 for c in _SNAPSHOT_FIELDS})
    return existing.sort("fetched_at")
