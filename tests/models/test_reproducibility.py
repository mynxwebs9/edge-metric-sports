"""Training twice with the same data/config/seed must produce materially identical results."""

from __future__ import annotations

import numpy as np

from nfl_predict.models import boosted, linear_models
from nfl_predict.models.elo import EloConfig, EloModel
from nfl_predict.models.feature_matrix import ABLATIONS, load_game_dataset, select_features


def test_ridge_margin_is_reproducible():
    ds = load_game_dataset([2020, 2021, 2022])
    X, _ = select_features(ds, ABLATIONS["B_off_def_efficiency"])
    y = ds.frame["home_margin"].to_numpy()

    fitted_1 = linear_models.fit_ridge(X, y, alpha=5.0)
    fitted_2 = linear_models.fit_ridge(X, y, alpha=5.0)

    assert fitted_1.coefficients == fitted_2.coefficients
    assert fitted_1.intercept == fitted_2.intercept

    pred_1 = linear_models.predict(fitted_1, X)
    pred_2 = linear_models.predict(fitted_2, X)
    np.testing.assert_array_equal(pred_1, pred_2)


def test_logistic_win_is_reproducible():
    from nfl_predict.models.feature_matrix import without_ties

    ds = without_ties(load_game_dataset([2020, 2021, 2022]))
    X, _ = select_features(ds, ABLATIONS["B_off_def_efficiency"])
    y = ds.frame["home_win"].to_numpy().astype(float)

    fitted_1 = linear_models.fit_logistic(X, y, C=1.0)
    fitted_2 = linear_models.fit_logistic(X, y, C=1.0)
    assert fitted_1.coefficients == fitted_2.coefficients


def test_lightgbm_margin_is_reproducible():
    ds = load_game_dataset([2020, 2021, 2022])
    X, _ = select_features(ds, ABLATIONS["B_off_def_efficiency"])
    y = ds.frame["home_margin"].to_numpy()

    fitted_1 = boosted.fit_lgbm_regressor(X, y, target="home_margin")
    fitted_2 = boosted.fit_lgbm_regressor(X, y, target="home_margin")

    pred_1 = boosted.predict(fitted_1, X)
    pred_2 = boosted.predict(fitted_2, X)
    np.testing.assert_array_almost_equal(pred_1, pred_2, decimal=10)
    assert fitted_1.feature_importances == fitted_2.feature_importances


def test_elo_is_fully_deterministic():
    from nfl_predict.models.targets import build_targets
    from nfl_predict.data.db import get_connection, init_schema
    from nfl_predict.models.elo import sort_games_chronologically

    conn = get_connection()
    init_schema(conn)
    targets = sort_games_chronologically(build_targets(conn, [2020, 2021]))
    conn.close()

    model_1 = EloModel(EloConfig(k_factor=20, home_field_advantage=50))
    history_1 = model_1.run_sequential(targets)
    model_2 = EloModel(EloConfig(k_factor=20, home_field_advantage=50))
    history_2 = model_2.run_sequential(targets)

    assert history_1["expected_home_win_prob"].to_list() == history_2["expected_home_win_prob"].to_list()
    assert model_1.ratings == model_2.ratings
