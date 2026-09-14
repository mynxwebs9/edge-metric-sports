"""Phase 5 Step 23 proof #15: live odds snapshots append instead of overwrite."""

from __future__ import annotations

import pytest

from nfl_predict.market.live_snapshot_store import (
    DuplicateSnapshotError,
    append_live_snapshot,
    read_live_snapshots,
)
from nfl_predict.market.odds_provider import OddsMarketSnapshot


def _snapshot(fetched_at: str, home_spread: float) -> OddsMarketSnapshot:
    return OddsMarketSnapshot(
        provider_event_id="evt_test", fetched_at=fetched_at, bookmaker="fixture_book",
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread,
        home_spread_price=-110, away_spread_price=-110,
        home_moneyline=-140, away_moneyline=120, total_line=46.0, over_price=-110, under_price=-110,
    )


def test_appending_a_second_snapshot_does_not_replace_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    append_live_snapshot(_snapshot("2024-09-03T10:00:00Z", -2.5))
    append_live_snapshot(_snapshot("2024-09-05T18:00:00Z", -3.5))

    frame = read_live_snapshots("evt_test")
    assert frame.height == 2
    assert set(frame["home_spread_traditional"].to_list()) == {-2.5, -3.5}
    # The opening snapshot's exact value must still be present, unaltered.
    assert frame.filter(frame["fetched_at"] == "2024-09-03T10:00:00Z")["home_spread_traditional"][0] == -2.5


def test_appending_the_exact_same_key_twice_raises_rather_than_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    append_live_snapshot(_snapshot("2024-09-03T10:00:00Z", -2.5))
    with pytest.raises(DuplicateSnapshotError):
        append_live_snapshot(_snapshot("2024-09-03T10:00:00Z", -4.0))  # same key, different value

    frame = read_live_snapshots("evt_test")
    assert frame.height == 1
    assert frame["home_spread_traditional"][0] == -2.5  # unchanged by the rejected attempt


def test_read_live_snapshots_on_a_nonexistent_event_returns_an_empty_frame_not_none(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    frame = read_live_snapshots("nonexistent_event")
    assert frame.height == 0
