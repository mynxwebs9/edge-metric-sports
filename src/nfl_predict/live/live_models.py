"""Phase 8A Step 5-7: live model predictions for upcoming games, using the exact frozen
Phase 4 training protocol - feature lists, hyperparameters, random seeds - never silently
changed, never retrained on sportsbook information.

**Ridge/LightGBM/Logistic**: reuse `nfl_predict.backtesting.walk_forward`'s own
`_fit_predict_ridge`/`_fit_predict_lightgbm`/`_fit_predict_logistic` helpers verbatim - the
same functions Phase 4's sealed backtest used - with the training mask covering every
completed game up to now and the predict mask covering the requested upcoming game(s). This
IS "periodic retraining using completed games only, per the frozen protocol" (Step 5), not
a new or different model; every prediction still carries `artifact_hash`/`feature_names_hash`
(the walk-forward helpers already compute these).

**Elo**: reuses `nfl_predict.research.input_packet`'s live Elo continuation (frozen
k_factor=40, home_field_advantage=45, verified against the Phase 4 freeze manifest) -
already built and tested in Phase 6.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nfl_predict.backtesting.walk_forward import (
    PredictionRecord,
    _fit_predict_lightgbm,
    _fit_predict_logistic,
    _fit_predict_ridge,
)
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES
from nfl_predict.research.input_packet import (
    FROZEN_ELO_MARGIN_INTERCEPT,
    FROZEN_ELO_MARGIN_SLOPE,
    _live_elo_model,
)
from nfl_predict.models.elo import apply_margin_transform

from nfl_predict.live.live_data import build_live_game_frame, completed_game_mask

MARGIN_MODEL_IDS = ("elo_v2", "ridge_margin_E_v1", "ridge_margin_CORE_v1", "lightgbm_F_v1")
WIN_MODEL_IDS = ("elo_v2", "logistic_win_E_v1", "lightgbm_F_v1")


def _last_completed_week(frame: pl.DataFrame, mask: np.ndarray) -> tuple[int, int]:
    if not mask.any():
        return (0, 0)
    seasons = frame["season"].to_numpy()[mask]
    weeks = frame["week"].to_numpy()[mask]
    idx = np.lexsort((weeks, seasons))
    return (int(seasons[idx[-1]]), int(weeks[idx[-1]]))


def predict_live_elo(game_ids: list[str], seasons: list[int]) -> list[PredictionRecord]:
    """Elo's own ratings are continuous/sequential across ALL seasons - `seasons` here is
    only used to build the frame that contains `game_ids`' rows; Elo's model state itself
    is computed from every completed game in the `games` table, not scoped to `seasons`."""
    frame = build_live_game_frame(seasons).frame
    target_rows = frame.filter(pl.col("game_id").is_in(game_ids))
    if target_rows.height == 0:
        return []

    model = _live_elo_model()
    records = []
    for row in target_rows.iter_rows(named=True):
        pred = model.predict_pregame(row["home_team_id"], row["away_team_id"], row["season"])
        predicted_margin = apply_margin_transform(np.array([pred["elo_diff_pre"]]), FROZEN_ELO_MARGIN_SLOPE, FROZEN_ELO_MARGIN_INTERCEPT)[0]
        common = dict(
            game_id=row["game_id"], season=row["season"], week=row["week"], season_type=row["season_type"],
            kickoff_timestamp=row.get("kickoff_time_naive"), model_id="elo_v2", model_version="v1", feature_version=None,
            training_cutoff_season=0, training_cutoff_week=0, training_row_count=0, artifact_hash=None, feature_names_hash=None,
        )
        records.append(PredictionRecord(target="home_margin", predicted_value=float(predicted_margin), **common))
        records.append(PredictionRecord(target="home_win", predicted_value=float(pred["expected_home_win_prob"]), **common))
    return records


def predict_live_ridge_lightgbm_logistic(game_ids: list[str], seasons: list[int]) -> dict[str, list[PredictionRecord] | None]:
    """Returns a dict keyed by model_id -> list[PredictionRecord], or None if that model's
    prediction could not be made safely (e.g. zero completed games to train on yet) - never
    a fabricated/guessed value."""
    frame = build_live_game_frame(seasons).frame
    train_mask = completed_game_mask(frame).to_numpy()
    predict_mask = frame["game_id"].is_in(game_ids).to_numpy()

    if predict_mask.sum() == 0:
        return {mid: None for mid in ("ridge_margin_E_v1", "ridge_margin_CORE_v1", "logistic_win_E_v1", "lightgbm_F_v1")}

    training_cutoff = _last_completed_week(frame, train_mask)
    results: dict[str, list[PredictionRecord] | None] = {}

    if train_mask.sum() == 0:
        # No completed games at all yet to train on (e.g. the very start of a season with
        # zero finished games) - every one of these models is honestly unavailable, not
        # substituted with a guess.
        return {mid: None for mid in ("ridge_margin_E_v1", "ridge_margin_CORE_v1", "logistic_win_E_v1", "lightgbm_F_v1")}

    results["ridge_margin_E_v1"] = _fit_predict_ridge(frame, train_mask, predict_mask, ABLATIONS["E_add_personnel"], "home_margin", 5.0, "ridge_margin_E_v1", training_cutoff)
    results["ridge_margin_CORE_v1"] = _fit_predict_ridge(frame, train_mask, predict_mask, CORE_FEATURES, "home_margin", 5.0, "ridge_margin_CORE_v1", training_cutoff)
    results["logistic_win_E_v1"] = _fit_predict_logistic(frame, train_mask, predict_mask, ABLATIONS["E_add_personnel"], 1.0, "logistic_win_E_v1", training_cutoff)
    results["lightgbm_F_v1"] = _fit_predict_lightgbm(frame, train_mask, predict_mask, ABLATIONS["F_full"], "lightgbm_F_v1", training_cutoff)
    return results
