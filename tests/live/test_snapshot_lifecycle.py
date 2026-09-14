"""Phase 8A Step 28 proofs #3, #16-18: post-kickoff data cannot enter/become a pregame
snapshot, multiple pregame runs preserve history, and latest-valid-snapshot selection
works."""

from __future__ import annotations

import pytest

from nfl_predict.live.snapshot_lifecycle import (
    PregameRunAlreadyExistsError,
    PregameRunRecord,
    list_pregame_runs,
    select_latest_valid_pregame_snapshot,
    write_pregame_run,
)

KICKOFF = "2026-09-13T17:00:00+00:00"


def _record(run_id, run_timestamp, market_ts=None, research_ts=None, market_available=True, research_available=True, kickoff=KICKOFF) -> PregameRunRecord:
    return PregameRunRecord(
        game_id="g1", season=2026, week=1, run_id=run_id, run_timestamp=run_timestamp, kickoff_timestamp=kickoff,
        elo_available=True, ridge_available=True, lightgbm_available=True, logistic_available=True,
        market_available=market_available, market_snapshot_timestamp=market_ts or run_timestamp,
        research_available=research_available, research_classification="NO_MATERIAL_NEW_INFORMATION",
        research_timestamp=research_ts or run_timestamp, payload={"elo_margin": 3.0},
    )


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.live.snapshot_lifecycle.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_multiple_pregame_runs_are_all_preserved(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    write_pregame_run(_record("thursday", "2026-09-10T12:00:00+00:00"))
    write_pregame_run(_record("friday", "2026-09-11T12:00:00+00:00"))
    write_pregame_run(_record("sunday_morning", "2026-09-13T09:00:00+00:00"))
    runs = list_pregame_runs(2026, 1, "g1")
    assert runs == ["friday", "sunday_morning", "thursday"]


def test_a_run_id_cannot_be_overwritten(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    write_pregame_run(_record("thursday", "2026-09-10T12:00:00+00:00"))
    with pytest.raises(PregameRunAlreadyExistsError):
        write_pregame_run(_record("thursday", "2026-09-10T12:00:00+00:00"))


def test_a_post_kickoff_run_can_never_be_selected_as_the_current_pregame_snapshot(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    # A "run" whose own timestamp is AFTER kickoff (e.g. a bug, or a postgame research
    # pass mislabeled) must never become the "current pregame" snapshot.
    write_pregame_run(_record("valid_pregame", "2026-09-13T10:00:00+00:00"))
    write_pregame_run(_record("bad_postkickoff", "2026-09-13T20:00:00+00:00"))  # after 17:00 kickoff

    selected = select_latest_valid_pregame_snapshot(2026, 1, "g1", now="2026-09-13T21:00:00+00:00")
    assert selected["run_id"] == "valid_pregame"


def test_latest_valid_snapshot_picks_the_most_recent_qualifying_run(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    write_pregame_run(_record("thursday", "2026-09-10T12:00:00+00:00"))
    write_pregame_run(_record("friday", "2026-09-11T12:00:00+00:00"))
    write_pregame_run(_record("sunday_morning", "2026-09-13T09:00:00+00:00"))

    selected = select_latest_valid_pregame_snapshot(2026, 1, "g1", now="2026-09-13T10:00:00+00:00")
    assert selected["run_id"] == "sunday_morning"


def test_stale_market_disqualifies_a_run_from_being_selected(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    # Market snapshot is 30 hours old relative to "now" - exceeds the 24h default max.
    write_pregame_run(_record("stale_market_run", "2026-09-12T12:00:00+00:00", market_ts="2026-09-11T00:00:00+00:00"))
    selected = select_latest_valid_pregame_snapshot(2026, 1, "g1", now="2026-09-12T13:00:00+00:00")
    assert selected is None


def test_missing_research_disqualifies_a_run_when_research_is_required(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    write_pregame_run(_record("no_research", "2026-09-12T12:00:00+00:00", research_available=False))
    selected = select_latest_valid_pregame_snapshot(2026, 1, "g1", now="2026-09-12T12:30:00+00:00", require_research=True)
    assert selected is None


def test_older_invalid_snapshots_remain_readable_for_audit(tmp_path, monkeypatch):
    from nfl_predict.live.snapshot_lifecycle import read_pregame_run

    _patch(monkeypatch, tmp_path)
    write_pregame_run(_record("bad_postkickoff", "2026-09-13T20:00:00+00:00"))
    # Not selectable as current, but still fully readable - nothing was deleted.
    record = read_pregame_run(2026, 1, "g1", "bad_postkickoff")
    assert record["run_id"] == "bad_postkickoff"
