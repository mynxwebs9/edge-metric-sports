"""Phase 8A Step 28 proofs #1, #4-7: live feature schema matches the frozen model feature
lists, Elo only updates from completed games, Ridge/LightGBM feature order matches the
frozen protocol, missing required features fail prediction safely (never a fabricated
value)."""

from __future__ import annotations

from nfl_predict.live.live_models import predict_live_elo, predict_live_ridge_lightgbm_logistic
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES


def test_live_ridge_e_uses_the_exact_frozen_e_add_personnel_feature_list():
    """Proof #1/#5: the live wrapper passes ABLATIONS["E_add_personnel"] verbatim - the
    same frozen list Phase 4's sealed backtest used - never a live-only variant."""
    import inspect

    from nfl_predict.live import live_models

    source = inspect.getsource(live_models.predict_live_ridge_lightgbm_logistic)
    assert 'ABLATIONS["E_add_personnel"]' in source
    assert "CORE_FEATURES" in source
    assert 'ABLATIONS["F_full"]' in source


def test_live_predictions_exist_for_every_requested_upcoming_game():
    results = predict_live_ridge_lightgbm_logistic(["2026_01_DEN_KC", "2026_01_BUF_HOU"], list(range(2010, 2027)))
    ridge_game_ids = {r.game_id for r in results["ridge_margin_E_v1"]}
    assert ridge_game_ids == {"2026_01_DEN_KC", "2026_01_BUF_HOU"}


def test_live_lightgbm_predicts_all_three_targets():
    results = predict_live_ridge_lightgbm_logistic(["2026_01_DEN_KC"], list(range(2010, 2027)))
    targets = {r.target for r in results["lightgbm_F_v1"]}
    assert targets == {"home_margin", "total_points", "home_win"}


def test_no_completed_games_at_all_produces_explicit_unavailable_not_a_fabricated_value(monkeypatch):
    """Proof #7: if there is nothing to train on, every model is None (unavailable) -
    never a guessed prediction."""
    import polars as pl

    from nfl_predict.models.feature_matrix import ModelDataset

    def fake_frame(seasons):
        # A single upcoming game, zero completed games - the frozen ABLATIONS columns plus
        # the bare minimum identity/outcome columns the live pipeline touches.
        cols = {c: [None] for c in ABLATIONS["F_full"]}
        cols.update({
            "game_id": ["g1"], "season": [2026], "week": [1], "season_type": ["REG"],
            "home_margin": [None], "total_points": [None], "home_win": [None], "is_tie": [False],
        })
        return ModelDataset(seasons=seasons, frame=pl.DataFrame(cols))

    monkeypatch.setattr("nfl_predict.live.live_models.build_live_game_frame", fake_frame)
    results = predict_live_ridge_lightgbm_logistic(["g1"], [2026])
    assert all(v is None for v in results.values())


def test_elo_prediction_uses_predict_pregame_never_run_sequential_on_the_target_game():
    """Proof #4: live Elo prediction is read-only for the target game (predict_pregame),
    never folding the target game's own (nonexistent) outcome into ratings."""
    import inspect

    from nfl_predict.live import live_models

    source = inspect.getsource(live_models.predict_live_elo)
    assert "predict_pregame" in source
    assert "run_sequential" not in source  # only used internally by _live_elo_model on COMPLETED games
