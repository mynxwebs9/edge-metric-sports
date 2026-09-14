"""Phase 6 Step 24 proof #11: outcomes cannot mutate prior research records."""

from __future__ import annotations

import pytest

from nfl_predict.research.prospective_ledger import (
    LedgerEntryAlreadyExistsError,
    ProspectiveLedgerEntry,
    append_ledger_entry,
    join_outcomes,
    read_ledger_entries,
)


def _entry(game_id="g1", run_id="run_a") -> ProspectiveLedgerEntry:
    return ProspectiveLedgerEntry(
        game_id=game_id, season=2026, week=1, research_id="r1", run_id=run_id,
        kickoff_timestamp="2026-09-13T17:00:00+00:00", research_timestamp="2026-09-11T10:00:00+00:00",
        elo_predicted_margin=1.5, elo_home_win_probability=0.55, ridge_predicted_margin=None,
        lightgbm_predicted_margin=None, market_home_spread_traditional=-2.5, market_home_moneyline=-130,
        market_snapshot_timestamp="2026-09-11T09:00:00+00:00", research_classification="NO_MATERIAL_NEW_INFORMATION",
        materiality_level=0, source_urls=("https://example.com",), unresolved_risks=(),
        frozen_at="2026-09-11T10:05:00+00:00",
    )


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.research.prospective_ledger.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_append_then_read_round_trips(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    append_ledger_entry(_entry())
    entries = read_ledger_entries(2026, 1)
    assert len(entries) == 1
    assert entries[0]["game_id"] == "g1"


def test_duplicate_game_run_id_raises(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    append_ledger_entry(_entry())
    with pytest.raises(LedgerEntryAlreadyExistsError):
        append_ledger_entry(_entry())


def test_multiple_runs_for_the_same_game_both_persist(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    append_ledger_entry(_entry(run_id="thursday"))
    append_ledger_entry(_entry(run_id="friday"))
    entries = read_ledger_entries(2026, 1)
    assert {e["run_id"] for e in entries} == {"thursday", "friday"}


def test_join_outcomes_does_not_mutate_the_stored_ledger(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    append_ledger_entry(_entry())
    before = read_ledger_entries(2026, 1)

    joined = join_outcomes(before, outcomes_by_game_id={"g1": {"actual_margin": 7.0}})
    assert joined[0]["outcome"] == {"actual_margin": 7.0}
    assert "outcome" not in before[0]  # the object returned by read_ledger_entries is untouched

    after = read_ledger_entries(2026, 1)
    assert after == before  # re-reading from disk shows the ledger file itself is untouched
    assert "outcome" not in after[0]


def test_tampering_with_the_ledger_file_is_detected_on_read(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    append_ledger_entry(_entry())
    from nfl_predict.research.prospective_ledger import _ledger_paths

    data_path, _ = _ledger_paths(2026, 1)
    with data_path.open("a", encoding="utf-8") as f:
        f.write('{"game_id": "sneaky_injected_row"}\n')

    with pytest.raises(ValueError, match="does not match its recorded hash"):
        read_ledger_entries(2026, 1)
