"""Phase 8A Step 28 proof #19: results cannot mutate original predictions - results are
joined separately, never fabricated for an unplayed game."""

from __future__ import annotations

import pytest

from nfl_predict.live.prediction_publication import PublicationState, PublicModelPrediction, publish_model_prediction
from nfl_predict.live.results_ingestion import (
    ConflictingGradedResultError,
    compute_graded_result,
    read_graded_result,
    write_graded_result,
)


def test_an_unplayed_game_produces_no_graded_result():
    assert compute_graded_result("2026_01_DEN_KC") is None  # real upcoming game, not yet played


def test_a_completed_game_produces_a_real_graded_result():
    # 2 real games were already completed in the ingested 2026 season (see Phase 8A setup).
    from nfl_predict.data.db import get_connection, init_schema

    conn = get_connection()
    init_schema(conn)
    row = conn.execute("SELECT game_id FROM games WHERE season = 2026 AND game_status = 'final' LIMIT 1").fetchone()
    conn.close()
    assert row is not None, "expected at least one completed 2026 game for this test"

    result = compute_graded_result(row["game_id"])
    assert result is not None
    assert result.home_margin == result.home_score - result.away_score


def test_writing_a_graded_result_never_touches_a_published_prediction_record(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.storage.blob_store.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    pred = PublicModelPrediction(
        prediction_id="run1", game_id="g1", season=2026, week=1, generated_at="2026-09-11T20:00:00+00:00",
        elo_home_win_probability=0.6, elo_predicted_margin=3.0, ridge_predicted_margin=2.5, lightgbm_predicted_margin=2.8,
        model_agreement_all_agree=True, model_agreement_dispersion=0.5, predicted_winner="home",
    )
    publish_model_prediction(pred, PublicationState.PUBLISHED)
    pred_path = tmp_path / "live" / "public_predictions" / "g1" / "run1.json"
    bytes_before = pred_path.read_bytes()

    from nfl_predict.live.results_ingestion import GradedResult

    write_graded_result(GradedResult(game_id="g1", season=2026, week=1, home_score=27, away_score=20, home_margin=7, total_points=47, home_win=True, is_tie=False, graded_at="2026-09-13T23:00:00+00:00"))

    assert pred_path.read_bytes() == bytes_before  # untouched by results ingestion


def test_a_conflicting_result_at_the_same_game_id_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    from nfl_predict.live.results_ingestion import GradedResult

    write_graded_result(GradedResult(game_id="g1", season=2026, week=1, home_score=27, away_score=20, home_margin=7, total_points=47, home_win=True, is_tie=False, graded_at="t1"))
    with pytest.raises(ConflictingGradedResultError):
        write_graded_result(GradedResult(game_id="g1", season=2026, week=1, home_score=10, away_score=10, home_margin=0, total_points=20, home_win=None, is_tie=True, graded_at="t2"))


def test_resubmitting_the_identical_result_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    from nfl_predict.live.results_ingestion import GradedResult

    r = GradedResult(game_id="g1", season=2026, week=1, home_score=27, away_score=20, home_margin=7, total_points=47, home_win=True, is_tie=False, graded_at="t1")
    write_graded_result(r)
    write_graded_result(r)  # does not raise
    assert read_graded_result("g1")["home_score"] == 27
