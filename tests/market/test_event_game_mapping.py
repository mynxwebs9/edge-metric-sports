"""Phase 8A provenance correction: the additive, append-only event/game mapping index.
Mirrors `tests/research/test_prospective_ledger.py`'s / `tests/market/test_live_snapshot_store.py`'s
established append-only-plus-hash-verification test pattern.
"""

from __future__ import annotations

import pytest

from nfl_predict.market.event_game_mapping import (
    EventGameMapping,
    EventGameMappingConflictError,
    append_event_game_mapping_idempotent,
    find_provider_event_ids_for_game,
    read_event_game_mappings,
)


def _mapping(provider_event_id="evt_denkc", canonical_game_id="2026_01_DEN_KC", mapping_timestamp="2026-09-14T17:00:00+00:00") -> EventGameMapping:
    return EventGameMapping(
        provider_event_id=provider_event_id, canonical_game_id=canonical_game_id, season=2026, week=1,
        provider_home_team="Kansas City Chiefs", provider_away_team="Denver Broncos",
        home_team_id="2310", away_team_id="1400", kickoff_timestamp="2026-09-14T20:15:00",
        mapping_timestamp=mapping_timestamp, mapping_method="exact_team_name_match_v1",
    )


def test_append_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    appended = append_event_game_mapping_idempotent(_mapping())
    assert appended is True

    rows = read_event_game_mappings(2026, 1)
    assert len(rows) == 1
    assert rows[0]["provider_event_id"] == "evt_denkc"
    assert rows[0]["canonical_game_id"] == "2026_01_DEN_KC"

    assert find_provider_event_ids_for_game(2026, 1, "2026_01_DEN_KC") == ["evt_denkc"]
    assert find_provider_event_ids_for_game(2026, 1, "some_other_game") == []


def test_re_recording_the_identical_mapping_is_a_safe_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    append_event_game_mapping_idempotent(_mapping(mapping_timestamp="2026-09-14T17:00:00+00:00"))
    # A later re-poll re-confirms the same mapping with a NEW mapping_timestamp - still a
    # no-op, and the ORIGINAL timestamp is what's kept.
    appended_again = append_event_game_mapping_idempotent(_mapping(mapping_timestamp="2026-09-14T18:30:00+00:00"))

    assert appended_again is False
    rows = read_event_game_mappings(2026, 1)
    assert len(rows) == 1
    assert rows[0]["mapping_timestamp"] == "2026-09-14T17:00:00+00:00"


def test_a_genuinely_different_mapping_at_the_same_provider_event_id_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    append_event_game_mapping_idempotent(_mapping(provider_event_id="evt_denkc", canonical_game_id="2026_01_DEN_KC"))

    with pytest.raises(EventGameMappingConflictError):
        append_event_game_mapping_idempotent(_mapping(provider_event_id="evt_denkc", canonical_game_id="2026_02_SOME_OTHER_GAME"))

    # The original mapping must be unaffected by the refused conflicting write.
    rows = read_event_game_mappings(2026, 1)
    assert len(rows) == 1
    assert rows[0]["canonical_game_id"] == "2026_01_DEN_KC"


def test_tampering_with_the_persisted_file_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    append_event_game_mapping_idempotent(_mapping())
    data_path = tmp_path / "market" / "event_game_mapping" / "season=2026" / "week=1" / "mappings.jsonl"
    data_path.write_text(data_path.read_text(encoding="utf-8").replace("2026_01_DEN_KC", "2026_01_TAMPERED"), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its recorded hash"):
        read_event_game_mappings(2026, 1)


def test_reading_a_season_week_with_no_mappings_returns_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    assert read_event_game_mappings(2026, 1) == []
    assert find_provider_event_ids_for_game(2026, 1, "2026_01_DEN_KC") == []
