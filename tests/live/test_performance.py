"""Phase 8A Step 28 proof #28: current-season performance excludes ungraded games; and
categories (ALL_MODEL_PREDICTIONS vs BEST_BETS/LEANS) are never mixed."""

from __future__ import annotations

from nfl_predict.live.performance import compute_all_model_predictions_performance
from nfl_predict.live.results_ingestion import GradedResult, write_graded_result


def _pred(game_id, predicted_winner="home", elo_prob=0.7, elo_margin=5.0) -> dict:
    return {"game_id": game_id, "predicted_winner": predicted_winner, "elo_home_win_probability": elo_prob, "elo_predicted_margin": elo_margin}


def test_ungraded_games_are_excluded_from_the_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())

    write_graded_result(GradedResult(game_id="g1", season=2026, week=1, home_score=27, away_score=20, home_margin=7, total_points=47, home_win=True, is_tie=False, graded_at="t"))
    # g2 has NO graded result (not yet played).
    predictions = [_pred("g1"), _pred("g2")]

    perf = compute_all_model_predictions_performance(predictions)
    assert perf.n_graded == 1
    assert perf.n_ungraded_excluded == 1
    assert perf.winner_accuracy == 1.0  # g1: predicted home, home won


def test_a_tied_game_is_excluded_not_counted_as_wrong(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    write_graded_result(GradedResult(game_id="g1", season=2026, week=1, home_score=20, away_score=20, home_margin=0, total_points=40, home_win=None, is_tie=True, graded_at="t"))
    perf = compute_all_model_predictions_performance([_pred("g1")])
    assert perf.n_graded == 0
    assert perf.n_ungraded_excluded == 1


def test_margin_mae_and_brier_are_computed_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    write_graded_result(GradedResult(game_id="g1", season=2026, week=1, home_score=27, away_score=20, home_margin=7, total_points=47, home_win=True, is_tie=False, graded_at="t"))
    perf = compute_all_model_predictions_performance([_pred("g1", elo_prob=0.7, elo_margin=5.0)])
    assert perf.margin_mae == 2.0  # |5.0 - 7| = 2.0
    assert perf.brier_score == (0.7 - 1.0) ** 2


def test_all_ungraded_returns_none_metrics_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("nfl_predict.live.results_ingestion.get_settings", lambda: type("S", (), {"data_dir": tmp_path, "storage_backend": "sqlite", "database_url": None})())
    perf = compute_all_model_predictions_performance([_pred("g_never_played")])
    assert perf.n_graded == 0
    assert perf.winner_accuracy is None
    assert perf.brier_score is None
    assert perf.margin_mae is None
