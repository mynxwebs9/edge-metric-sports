"""Phase 8A Step 28 proof #9: duplicate identical odds snapshots are handled
idempotently (a genuine content conflict at the same key is still refused)."""

from __future__ import annotations

import pytest

from nfl_predict.market.live_snapshot_store import (
    DuplicateSnapshotError,
    append_live_snapshot_idempotent,
    read_live_snapshots,
)
from nfl_predict.market.odds_provider import OddsMarketSnapshot


def _snapshot(home_spread=-2.5) -> OddsMarketSnapshot:
    return OddsMarketSnapshot(
        provider_event_id="evt_test", fetched_at="2026-09-11T10:00:00Z", bookmaker="fixture_book",
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread,
        home_spread_price=-110, away_spread_price=-110, home_moneyline=-140, away_moneyline=120,
        total_line=46.0, over_price=-110, under_price=-110,
    )


def test_resubmitting_the_identical_snapshot_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    _, appended_1 = append_live_snapshot_idempotent(_snapshot())
    _, appended_2 = append_live_snapshot_idempotent(_snapshot())
    assert appended_1 is True
    assert appended_2 is False
    assert read_live_snapshots("evt_test").height == 1  # not duplicated


def test_a_genuinely_different_snapshot_at_the_same_key_still_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    append_live_snapshot_idempotent(_snapshot(home_spread=-2.5))
    with pytest.raises(DuplicateSnapshotError):
        append_live_snapshot_idempotent(_snapshot(home_spread=-4.0))  # same key, different content
