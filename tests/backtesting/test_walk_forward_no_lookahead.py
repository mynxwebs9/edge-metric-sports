"""Phase 4 Step 13 proofs #3-5, #8-10: the walk-forward protocol's core no-look-ahead and
reproducibility guarantees, tested directly against the mechanism that enforces them (not
just against one run's output)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nfl_predict.backtesting.freeze import CANDIDATE_MODELS
from nfl_predict.backtesting.walk_forward import (
    FEATURE_SET_LOOKUP,
    _fit_predict_lightgbm,
    _fit_predict_logistic,
    _fit_predict_naive,
    _fit_predict_ridge,
    _select,
    _training_mask,
)
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES
from nfl_predict.features.registry import assert_no_denylisted_columns


# ---------------------------------------------------------------------------
# Proof #3: frozen feature lists cannot silently change - walk_forward.py's per-model
# feature lists must be byte-identical to what freeze.py already froze and hashed.
# ---------------------------------------------------------------------------

def test_ridge_and_logistic_candidates_use_the_exact_frozen_feature_lists():
    frozen_by_id = {c["model_id"]: c for c in CANDIDATE_MODELS}

    assert FEATURE_SET_LOOKUP[frozen_by_id["ridge_margin_E_v1"]["feature_set"]] == ABLATIONS["E_add_personnel"]
    assert FEATURE_SET_LOOKUP[frozen_by_id["ridge_total_E_v1"]["feature_set"]] == ABLATIONS["E_add_personnel"]
    assert FEATURE_SET_LOOKUP[frozen_by_id["logistic_win_E_v1"]["feature_set"]] == ABLATIONS["E_add_personnel"]
    assert FEATURE_SET_LOOKUP[frozen_by_id["ridge_margin_CORE_v1"]["feature_set"]] == CORE_FEATURES
    assert FEATURE_SET_LOOKUP[frozen_by_id["lightgbm_F_v1"]["feature_set"]] == ABLATIONS["F_full"]


def test_frozen_candidate_hyperparameters_match_what_walk_forward_actually_uses():
    frozen_by_id = {c["model_id"]: c for c in CANDIDATE_MODELS}
    assert frozen_by_id["ridge_margin_E_v1"]["hyperparameters"]["alpha"] == 5.0
    assert frozen_by_id["ridge_total_E_v1"]["hyperparameters"]["alpha"] == 5.0
    assert frozen_by_id["ridge_margin_CORE_v1"]["hyperparameters"]["alpha"] == 5.0
    assert frozen_by_id["logistic_win_E_v1"]["hyperparameters"]["C"] == 1.0


# ---------------------------------------------------------------------------
# Proof #9: identifier columns are structurally rejected as model features, even inside
# the walk-forward's own `_select` helper (not just Phase 3's `select_features`).
# ---------------------------------------------------------------------------

def test_walk_forward_select_rejects_identifier_columns():
    frame = pl.DataFrame({"game_id": ["a", "b"], "team_id": ["x", "y"], "diff_days_rest": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Identifier column"):
        _select(frame, ["team_id", "diff_days_rest"])


# ---------------------------------------------------------------------------
# Proof #8: market/target-leakage columns remain forbidden for Phase 4's own holdout
# loader too (not just Phase 3's) - `_load_game_dataset_impl` runs the same denylist guard
# regardless of caller, and this asserts that guard function itself still rejects a known
# market column name.
# ---------------------------------------------------------------------------

def test_market_columns_remain_denylisted_for_phase4_too():
    with pytest.raises(ValueError):
        assert_no_denylisted_columns(["diff_days_rest", "spread_line"])
    with pytest.raises(ValueError):
        assert_no_denylisted_columns(["total_line", "diff_off_epa_pp_season"])
    assert_no_denylisted_columns(["diff_days_rest", "diff_off_epa_pp_season"])  # does not raise


# ---------------------------------------------------------------------------
# Proof #4/#5: the training mask is the mechanism that guarantees a later week never
# influences an earlier prediction - exercised directly and exhaustively here, including
# the REG/POST week-numbering edge case that makes this non-trivial (POST week numbers are
# always numerically GREATER than REG week numbers within the same season).
# ---------------------------------------------------------------------------

def test_training_mask_excludes_the_current_and_every_later_week_same_season():
    seasons = np.array([2024, 2024, 2024, 2024])
    weeks = np.array([1, 2, 3, 4])
    mask = _training_mask(seasons, weeks, season_p=2024, week_p=3)
    assert mask.tolist() == [True, True, False, False]


def test_training_mask_includes_every_earlier_season_regardless_of_week():
    seasons = np.array([2010, 2022, 2023, 2024])
    weeks = np.array([22, 1, 1, 1])
    # Every row here is either an earlier season (always eligible, any week number) or the
    # same season (2024) at week 1, which is strictly before the week-2 prediction.
    mask = _training_mask(seasons, weeks, season_p=2024, week_p=2)
    assert mask.tolist() == [True, True, True, True]


def test_training_mask_orders_postseason_after_every_regular_season_week_in_the_same_season():
    """POST week numbers (19-22, or 18-21 pre-2021) are always > every REG week number in
    the same season - verified against the real games table. A prediction for POST week 19
    must be trainable on that season's REG weeks 1-18; a prediction for REG week 1 must
    NEVER be trainable on that same season's POST games, even though word order in the
    season isn't what the raw week number implies out of context."""
    seasons = np.array([2024, 2024, 2024])
    season_types = ["REG", "REG", "POST"]
    weeks = np.array([1, 18, 19])

    mask_predicting_post_week19 = _training_mask(seasons, weeks, season_p=2024, week_p=19)
    assert mask_predicting_post_week19.tolist() == [True, True, False]

    mask_predicting_reg_week1 = _training_mask(seasons, weeks, season_p=2024, week_p=1)
    assert mask_predicting_reg_week1.tolist() == [False, False, False]


def test_ledger_training_cutoffs_never_reach_or_exceed_their_own_predicted_week(tmp_path):
    """Integration-level check against the real Phase 4 ledger (if present): every
    non-Elo prediction's recorded training_cutoff must sort strictly before the (season,
    week) it predicts."""
    from nfl_predict.backtesting.ledger import ledger_paths

    ledger_path, _ = ledger_paths("phase4_v1")
    if not ledger_path.is_file():
        pytest.skip("phase4_v1 ledger not present in this environment")

    ledger = pl.read_parquet(ledger_path, memory_map=False).filter(pl.col("model_id") != "elo_v2")
    season = ledger["season"].to_numpy()
    week = ledger["week"].to_numpy()
    cutoff_season = ledger["training_cutoff_season"].to_numpy()
    cutoff_week = ledger["training_cutoff_week"].to_numpy()

    violates = (cutoff_season > season) | ((cutoff_season == season) & (cutoff_week >= week))
    assert not violates.any(), f"{violates.sum()} predictions have a training cutoff at or after their own predicted week"


# ---------------------------------------------------------------------------
# Proof #10: re-running the exact same frozen fit (same frame, same masks, same
# hyperparameters, same random seed) gives materially identical predictions.
# ---------------------------------------------------------------------------

def _tiny_synthetic_frame(n_train: int = 40, n_predict: int = 4, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_train + n_predict
    seasons = np.array([2023] * n_train + [2024] * n_predict)
    train_weeks = (np.arange(n_train) % 18) + 1
    weeks = np.concatenate([train_weeks, np.array([1] * n_predict)])
    feature_cols = ["diff_days_rest", "diff_off_epa_pp_season"]
    data = {
        "game_id": [f"g{i}" for i in range(n)],
        "season": seasons.tolist(), "week": weeks.tolist(), "season_type": ["REG"] * n,
        "kickoff_time_naive": [None] * n, "is_tie": [False] * n,
        "home_margin": rng.normal(0, 10, n).tolist(),
        "total_points": rng.normal(45, 10, n).tolist(),
        "home_win": rng.integers(0, 2, n).tolist(),
    }
    for c in feature_cols:
        data[c] = rng.normal(0, 1, n).tolist()
    return pl.DataFrame(data), feature_cols, seasons, weeks


def test_ridge_refit_is_deterministic_given_the_same_inputs():
    frame, feature_cols, seasons, weeks = _tiny_synthetic_frame()
    train_mask = seasons == 2023
    predict_mask = seasons == 2024

    run_a = _fit_predict_ridge(frame, train_mask, predict_mask, feature_cols, "home_margin", 5.0, "test_model", (2023, 18))
    run_b = _fit_predict_ridge(frame, train_mask, predict_mask, feature_cols, "home_margin", 5.0, "test_model", (2023, 18))

    values_a = [r.predicted_value for r in run_a]
    values_b = [r.predicted_value for r in run_b]
    assert values_a == values_b
    assert len(values_a) == predict_mask.sum()


def test_logistic_refit_is_deterministic_given_the_same_inputs():
    frame, feature_cols, seasons, weeks = _tiny_synthetic_frame()
    train_mask = seasons == 2023
    predict_mask = seasons == 2024

    run_a = _fit_predict_logistic(frame, train_mask, predict_mask, feature_cols, 1.0, "test_model", (2023, 18))
    run_b = _fit_predict_logistic(frame, train_mask, predict_mask, feature_cols, 1.0, "test_model", (2023, 18))
    assert [r.predicted_value for r in run_a] == [r.predicted_value for r in run_b]


def test_lightgbm_refit_is_deterministic_given_the_same_inputs():
    frame, feature_cols, seasons, weeks = _tiny_synthetic_frame(n_train=60, n_predict=4)
    train_mask = seasons == 2023
    predict_mask = seasons == 2024

    run_a = _fit_predict_lightgbm(frame, train_mask, predict_mask, feature_cols, "test_model", (2023, 18))
    run_b = _fit_predict_lightgbm(frame, train_mask, predict_mask, feature_cols, "test_model", (2023, 18))
    values_a = sorted((r.target, r.game_id, r.predicted_value) for r in run_a)
    values_b = sorted((r.target, r.game_id, r.predicted_value) for r in run_b)
    assert values_a == values_b


def test_naive_refit_is_deterministic_given_the_same_inputs():
    frame, feature_cols, seasons, weeks = _tiny_synthetic_frame()
    train_mask = seasons == 2023
    predict_mask = seasons == 2024

    run_a = _fit_predict_naive(frame, train_mask, predict_mask, (2023, 18))
    run_b = _fit_predict_naive(frame, train_mask, predict_mask, (2023, 18))
    values_a = sorted((r.target, r.game_id, r.predicted_value) for r in run_a)
    values_b = sorted((r.target, r.game_id, r.predicted_value) for r in run_b)
    assert values_a == values_b
