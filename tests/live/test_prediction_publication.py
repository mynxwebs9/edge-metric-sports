"""Phase 8A: publication-state lifecycle - superseding never mutates the original record."""

from __future__ import annotations

import pytest

from nfl_predict.live.prediction_publication import (
    PublicationState,
    PublicModelPrediction,
    PublicPredictionAlreadyExistsError,
    publish_model_prediction,
    read_model_prediction,
    supersede_previous_predictions,
)


def _pred(prediction_id, game_id="g1") -> PublicModelPrediction:
    return PublicModelPrediction(
        prediction_id=prediction_id, game_id=game_id, season=2026, week=1, generated_at="2026-09-11T20:00:00+00:00",
        elo_home_win_probability=0.6, elo_predicted_margin=3.0, ridge_predicted_margin=2.5, lightgbm_predicted_margin=2.8,
        model_agreement_all_agree=True, model_agreement_dispersion=0.5, predicted_winner="home",
    )


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())


def test_publish_then_read_round_trips(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_model_prediction(_pred("run1"), PublicationState.PUBLISHED)
    record = read_model_prediction("g1", "run1")
    assert record["state"] == "PUBLISHED"
    assert record["elo_predicted_margin"] == 3.0


def test_cannot_republish_the_same_prediction_id(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_model_prediction(_pred("run1"), PublicationState.PUBLISHED)
    with pytest.raises(PublicPredictionAlreadyExistsError):
        publish_model_prediction(_pred("run1"), PublicationState.PUBLISHED)


def test_superseding_never_mutates_the_original_record_bytes(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    publish_model_prediction(_pred("run1"), PublicationState.PUBLISHED)
    path_before = (tmp_path / "live" / "public_predictions" / "g1" / "run1.json")
    bytes_before = path_before.read_bytes()

    publish_model_prediction(_pred("run2"), PublicationState.PUBLISHED)
    superseded = supersede_previous_predictions("g1", "run2")

    assert superseded == ["run1"]
    assert path_before.read_bytes() == bytes_before  # original file untouched

    old_record = read_model_prediction("g1", "run1")
    assert old_record["state"] == "SUPERSEDED"
    assert old_record["superseded_by"] == "run2"

    new_record = read_model_prediction("g1", "run2")
    assert new_record["state"] == "PUBLISHED"  # the newest one is not superseded


def test_a_raw_iso_8601_timestamp_prediction_id_is_filesystem_safe_on_windows(tmp_path, monkeypatch):
    """Regression test: prediction_id is commonly a raw `_now_iso()` timestamp, e.g.
    '2026-09-11T22:58:09.419768+00:00', whose colons are illegal in Windows filenames.
    Publishing and reading back must work, and the record's own field must still carry the
    true, unmodified ISO string (only the on-disk filename is sanitized)."""
    _patch(monkeypatch, tmp_path)
    timestamp_id = "2026-09-11T22:58:09.419768+00:00"
    key = publish_model_prediction(_pred(timestamp_id), PublicationState.PUBLISHED)
    path = tmp_path / key  # key is a blob-store key (str); resolve against the local backend's root to inspect the file
    assert path.is_file()
    assert ":" not in path.name

    record = read_model_prediction("g1", timestamp_id)
    assert record["prediction_id"] == timestamp_id
    assert record["state"] == "PUBLISHED"
