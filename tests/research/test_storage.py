"""Phase 6 Step 24 proofs #9-10: research versioning and multiple runs append rather than
overwrite."""

from __future__ import annotations

import pytest

from nfl_predict.research.storage import (
    ResearchRunAlreadyExistsError,
    list_research_runs,
    read_research_run,
    verify_research_run_integrity,
    write_research_run,
)


def _patch_settings(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_write_research_run_creates_all_four_files(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    run_key = write_research_run(
        season=2026, week=1, game_id="2026_01_CHI_CAR", run_id="20260911T100000Z",
        input_packet={"game_id": "2026_01_CHI_CAR"}, findings_or_failure={"classification": "NO_MATERIAL_NEW_INFORMATION"},
        findings_kind="findings",
    )
    run_dir = tmp_path / run_key  # run_key is a blob-store key (str), not a Path - resolve against the local backend's root to inspect the on-disk files directly
    assert (run_dir / "input.json").is_file()
    assert (run_dir / "findings.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert not (run_dir / "evaluation.json").is_file()  # not provided


def test_a_second_write_with_the_same_run_id_raises(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    write_research_run(season=2026, week=1, game_id="g1", run_id="run_a", input_packet={}, findings_or_failure={}, findings_kind="findings")
    with pytest.raises(ResearchRunAlreadyExistsError):
        write_research_run(season=2026, week=1, game_id="g1", run_id="run_a", input_packet={}, findings_or_failure={}, findings_kind="findings")


def test_multiple_runs_for_the_same_game_are_all_preserved(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    write_research_run(season=2026, week=1, game_id="g1", run_id="thursday_run", input_packet={"day": "thu"}, findings_or_failure={}, findings_kind="findings")
    write_research_run(season=2026, week=1, game_id="g1", run_id="friday_run", input_packet={"day": "fri"}, findings_or_failure={}, findings_kind="findings")
    write_research_run(season=2026, week=1, game_id="g1", run_id="sunday_run", input_packet={"day": "sun"}, findings_or_failure={}, findings_kind="findings")

    runs = list_research_runs(2026, 1, "g1")
    assert runs == ["friday_run", "sunday_run", "thursday_run"]  # all three preserved
    assert read_research_run(2026, 1, "g1", "thursday_run")["input"]["day"] == "thu"
    assert read_research_run(2026, 1, "g1", "friday_run")["input"]["day"] == "fri"


def test_verify_research_run_integrity_detects_tampering(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    run_key = write_research_run(season=2026, week=1, game_id="g1", run_id="run_a", input_packet={"x": 1}, findings_or_failure={"y": 2}, findings_kind="findings")
    assert verify_research_run_integrity(2026, 1, "g1", "run_a") is True

    (tmp_path / run_key / "findings.json").write_text('{"y": 999}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its recorded hash"):
        verify_research_run_integrity(2026, 1, "g1", "run_a")


def test_read_research_run_on_nonexistent_run_raises_file_not_found(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        read_research_run(2026, 1, "nonexistent", "no_run")
