"""Phase 4 Step 2: the walk-forward weekly retraining/prediction protocol.

Simulates realistic weekly operation across 2024-2025 instead of training once and
predicting two whole seasons statically: for every (season, week) in the sealed holdout,
in chronological order, each non-Elo candidate is retrained from scratch on every completed
game strictly BEFORE that week (development 2010-2022, validation 2023, and every holdout
week already "played" earlier in this same walk) and then predicts only that week's games.
2024 Week 1 trains through 2023; 2024 Week 2 additionally sees completed 2024 Week 1; and so
on into 2025. No later week's games are ever visible to an earlier week's prediction - this
is enforced structurally by `_training_mask` (a game qualifies for training only if its
(season, week) sorts strictly before the week being predicted), not by trusting call order.

Elo is handled differently, deliberately: `EloModel.run_sequential` is ALREADY a
walk-forward process (it returns PRE-game ratings/win-probability for every game, updating
only after each game's own outcome is folded in), so it is run exactly ONCE over the entire
chronological game sequence from 2010 through 2025 rather than being "retrained" 40 times
over identical historical games - retraining it weekly would recompute the exact same
sequential update and waste time for zero behavioral difference. Its home-margin transform
(elo_diff_pre -> predicted margin) IS fit only once, from development-period Elo history
only (2010-2022) exactly as Phase 3 did - a fixed derived constant from already-frozen
inputs, not a new tunable parameter and not fit on holdout data.

Requires the Phase 4 freeze to be complete (`holdout_guard.assert_freeze_complete`) and
explicitly unseals 2024-2025 targets before anything else happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from nfl_predict.logging_conf import get_logger
from nfl_predict.models import baseline_naive, boosted, linear_models
from nfl_predict.models.elo import EloConfig, EloModel, apply_margin_transform, fit_margin_transform, sort_games_chronologically
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES, FEATURE_VERSION, ModelDataset, select_features
from nfl_predict.models.split import DEVELOPMENT_SEASONS, SEALED_HOLDOUT_SEASONS, VALIDATION_SEASON

from nfl_predict.backtesting.holdout_guard import (
    assert_freeze_complete,
    load_holdout_game_dataset,
    unseal_and_write_holdout_targets,
)

logger = get_logger(__name__)

ALL_SEASONS = DEVELOPMENT_SEASONS + [VALIDATION_SEASON] + SEALED_HOLDOUT_SEASONS
FEATURE_SET_LOOKUP = {**ABLATIONS, "CORE": CORE_FEATURES}

_RIDGE_MODEL_IDS = {"ridge_margin_E_v1": "home_margin", "ridge_total_E_v1": "total_points", "ridge_margin_CORE_v1": "home_margin"}
_LOGISTIC_MODEL_IDS = {"logistic_win_E_v1": "home_win"}
_LIGHTGBM_MODEL_ID = "lightgbm_F_v1"
_NAIVE_MODEL_ID = "naive_v1"
_ELO_MODEL_ID = "elo_v2"


@dataclass(frozen=True)
class PredictionRecord:
    game_id: str
    season: int
    week: int
    season_type: str
    kickoff_timestamp: str | None
    model_id: str
    model_version: str
    feature_version: str | None
    target: str
    predicted_value: float
    training_cutoff_season: int
    training_cutoff_week: int
    training_row_count: int
    artifact_hash: str | None
    feature_names_hash: str | None


def _training_mask(seasons: np.ndarray, weeks: np.ndarray, season_p: int, week_p: int) -> np.ndarray:
    """A game is training-eligible for the prediction made at (season_p, week_p) iff it
    sorts strictly earlier - either an earlier season, or the same season at a strictly
    earlier week number. Within a season, POST week numbers (19-22, or 18-21 pre-2021) are
    always numerically greater than every REG week number (1-18/17) - verified against the
    real `games` table - so comparing raw week numbers alone correctly orders REG before
    POST without needing season_type in the comparison."""
    return (seasons < season_p) | ((seasons == season_p) & (weeks < week_p))


def _holdout_week_sequence(seasons: np.ndarray, weeks: np.ndarray) -> list[tuple[int, int]]:
    sealed_mask = np.isin(seasons, SEALED_HOLDOUT_SEASONS)
    pairs = sorted(set(zip(seasons[sealed_mask].tolist(), weeks[sealed_mask].tolist())))
    return pairs


def _artifact_hash(obj) -> str:
    import hashlib
    import io
    import joblib

    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _feature_names_hash(names: list[str]) -> str:
    import hashlib

    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()


def _select(frame_slice: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Routes every feature slice through the same identifier/non-numeric/missing-column
    guard Phase 3 uses (`select_features`), rather than a bare `.select()` - defense in
    depth against a future refactor accidentally handing a team_id column to a model."""
    X, _ = select_features(ModelDataset(seasons=[], frame=frame_slice), feature_cols)
    return X


def _records_from_predictions(
    frame_slice: pl.DataFrame, model_id: str, target: str, predicted: np.ndarray,
    training_cutoff: tuple[int, int], n_train: int, artifact_hash: str | None, feature_names_hash: str | None,
) -> list[PredictionRecord]:
    game_ids = frame_slice["game_id"].to_list()
    seasons = frame_slice["season"].to_list()
    weeks = frame_slice["week"].to_list()
    season_types = frame_slice["season_type"].to_list()
    kickoffs = frame_slice["kickoff_time_naive"].to_list()
    return [
        PredictionRecord(
            game_id=game_ids[i], season=seasons[i], week=weeks[i], season_type=season_types[i],
            kickoff_timestamp=kickoffs[i], model_id=model_id, model_version="v1", feature_version=FEATURE_VERSION,
            target=target, predicted_value=float(predicted[i]),
            training_cutoff_season=training_cutoff[0], training_cutoff_week=training_cutoff[1],
            training_row_count=n_train, artifact_hash=artifact_hash, feature_names_hash=feature_names_hash,
        )
        for i in range(len(game_ids))
    ]


def _run_elo(full_frame: pl.DataFrame, elo_config: EloConfig) -> list[PredictionRecord]:
    targets_cols = full_frame.select([
        "game_id", "season", "week", "season_type", "kickoff_time_naive", "game_date",
        "home_team_id", "away_team_id", "home_margin", "home_win", "is_tie",
    ])
    full_sorted = sort_games_chronologically(targets_cols)
    model = EloModel(elo_config)
    history = model.run_sequential(full_sorted)

    dev_mask = full_sorted["season"].is_in(DEVELOPMENT_SEASONS).to_numpy()
    dev_history = history.filter(pl.Series(dev_mask))
    dev_actual_margin = full_sorted.filter(pl.Series(dev_mask))["home_margin"].to_numpy()
    slope, intercept = fit_margin_transform(dev_history, dev_actual_margin)
    logger.info("Elo margin transform (development-only, frozen): slope=%.4f intercept=%.4f", slope, intercept)

    # `history` is built by `run_sequential` iterating `full_sorted` row-by-row in that exact
    # order, so the two frames are already positionally aligned - filtering both by the same
    # boolean mask (rather than joining on game_id) keeps that alignment explicit and avoids
    # depending on a join's row-order guarantees.
    holdout_mask = full_sorted["season"].is_in(SEALED_HOLDOUT_SEASONS).to_numpy()
    holdout_history = history.filter(pl.Series(holdout_mask))
    holdout_meta = full_sorted.filter(pl.Series(holdout_mask))

    win_records = _records_from_predictions(
        holdout_meta, _ELO_MODEL_ID, "home_win", holdout_history["expected_home_win_prob"].to_numpy(),
        training_cutoff=(0, 0), n_train=0, artifact_hash=None, feature_names_hash=None,
    )
    margin_pred = apply_margin_transform(holdout_history["elo_diff_pre"].to_numpy(), slope, intercept)
    margin_records = _records_from_predictions(
        holdout_meta, _ELO_MODEL_ID, "home_margin", margin_pred,
        training_cutoff=(0, 0), n_train=0, artifact_hash=None, feature_names_hash=None,
    )
    # Elo has no fixed-batch "training_cutoff"/"n_train" the way the other models do - its
    # state at any point IS every prior game's outcome, applied sequentially. The (0, 0)/0
    # sentinels above are intentional and documented (not fabricated non-zero values); the
    # game's own (season, week) already identifies its exact point in the sequence.
    return win_records + margin_records


def run_walk_forward() -> list[PredictionRecord]:
    manifest = assert_freeze_complete()
    elo_entry = next(c for c in manifest["candidate_models"] if c["model_id"] == _ELO_MODEL_ID)
    elo_config = EloConfig(
        k_factor=elo_entry["hyperparameters"]["k_factor"],
        home_field_advantage=elo_entry["hyperparameters"]["home_field_advantage"],
        season_regression_fraction=elo_entry["hyperparameters"]["season_regression_fraction"],
        initial_rating=elo_entry["hyperparameters"]["initial_rating"],
    )

    logger.info("Unsealing 2024-2025 targets (freeze verified complete)")
    unseal_and_write_holdout_targets()

    dataset = load_holdout_game_dataset(ALL_SEASONS)
    frame = dataset.frame
    if not set(SEALED_HOLDOUT_SEASONS) <= set(frame["season"].unique().to_list()):
        raise RuntimeError("Holdout seasons missing from the loaded dataset after unsealing - refusing to proceed.")

    seasons_arr = frame["season"].to_numpy()
    weeks_arr = frame["week"].to_numpy()
    holdout_weeks = _holdout_week_sequence(seasons_arr, weeks_arr)
    logger.info("Walk-forward will predict %d holdout weeks (%d holdout games total)", len(holdout_weeks), int(np.isin(seasons_arr, SEALED_HOLDOUT_SEASONS).sum()))

    return _run_walk_forward_with_elo_config(frame, seasons_arr, weeks_arr, holdout_weeks, elo_config)


def _fit_predict_ridge(frame: pl.DataFrame, train_mask: np.ndarray, predict_mask: np.ndarray, feature_cols: list[str], target: str, alpha: float, model_id: str, training_cutoff: tuple[int, int]) -> list[PredictionRecord]:
    train_frame = frame.filter(pl.Series(train_mask))
    predict_frame = frame.filter(pl.Series(predict_mask))
    X_train = _select(train_frame, feature_cols)
    X_predict = _select(predict_frame, feature_cols)
    y_train = train_frame[target].to_numpy()

    fitted = linear_models.fit_ridge(X_train, y_train, alpha=alpha, target=target)
    predicted = linear_models.predict(fitted, X_predict)
    return _records_from_predictions(
        predict_frame, model_id, target, predicted, training_cutoff, len(y_train),
        _artifact_hash(fitted.pipeline), _feature_names_hash(feature_cols),
    )


def _fit_predict_logistic(frame: pl.DataFrame, train_mask: np.ndarray, predict_mask: np.ndarray, feature_cols: list[str], C: float, model_id: str, training_cutoff: tuple[int, int]) -> list[PredictionRecord]:
    train_frame = frame.filter(pl.Series(train_mask) & (~pl.col("is_tie")))
    predict_frame = frame.filter(pl.Series(predict_mask) & (~pl.col("is_tie")))
    if predict_frame.height == 0:
        return []
    X_train = _select(train_frame, feature_cols)
    X_predict = _select(predict_frame, feature_cols)
    y_train = train_frame["home_win"].to_numpy().astype(float)

    fitted = linear_models.fit_logistic(X_train, y_train, C=C, target="home_win")
    predicted = linear_models.predict(fitted, X_predict)
    return _records_from_predictions(
        predict_frame, model_id, "home_win", predicted, training_cutoff, len(y_train),
        _artifact_hash(fitted.pipeline), _feature_names_hash(feature_cols),
    )


def _fit_predict_lightgbm(frame: pl.DataFrame, train_mask: np.ndarray, predict_mask: np.ndarray, feature_cols: list[str], model_id: str, training_cutoff: tuple[int, int]) -> list[PredictionRecord]:
    train_frame = frame.filter(pl.Series(train_mask))
    predict_frame = frame.filter(pl.Series(predict_mask))
    X_train = _select(train_frame, feature_cols)
    X_predict = _select(predict_frame, feature_cols)
    records: list[PredictionRecord] = []

    for target in ("home_margin", "total_points"):
        y_train = train_frame[target].to_numpy()
        fitted = boosted.fit_lgbm_regressor(X_train, y_train, target=target)
        predicted = boosted.predict(fitted, X_predict)
        records += _records_from_predictions(
            predict_frame, model_id, target, predicted, training_cutoff, len(y_train),
            _artifact_hash(fitted.model), _feature_names_hash(feature_cols),
        )

    train_nt = train_frame.filter(~pl.col("is_tie"))
    predict_nt = predict_frame.filter(~pl.col("is_tie"))
    if predict_nt.height > 0:
        X_train_nt = _select(train_nt, feature_cols)
        X_predict_nt = _select(predict_nt, feature_cols)
        y_win_train = train_nt["home_win"].to_numpy()
        clf = lgb.LGBMClassifier(**boosted.DEFAULT_PARAMS)
        clf.fit(X_train_nt.to_pandas(), y_win_train)
        pred_win = clf.predict_proba(X_predict_nt.to_pandas())[:, 1]
        records += _records_from_predictions(
            predict_nt, model_id, "home_win", pred_win, training_cutoff, len(y_win_train),
            _artifact_hash(clf), _feature_names_hash(feature_cols),
        )
    return records


def _fit_predict_naive(frame: pl.DataFrame, train_mask: np.ndarray, predict_mask: np.ndarray, training_cutoff: tuple[int, int]) -> list[PredictionRecord]:
    train_frame = frame.filter(pl.Series(train_mask))
    predict_frame = frame.filter(pl.Series(predict_mask))
    naive = baseline_naive.fit_naive_baseline(train_frame)
    n = predict_frame.height
    records = []
    records += _records_from_predictions(predict_frame, _NAIVE_MODEL_ID, "home_margin", naive.predict_margin(n), training_cutoff, naive.n_games_fit, None, None)
    records += _records_from_predictions(predict_frame, _NAIVE_MODEL_ID, "total_points", naive.predict_total(n), training_cutoff, naive.n_games_fit, None, None)
    records += _records_from_predictions(predict_frame, _NAIVE_MODEL_ID, "home_win", naive.predict_home_win_prob(n), training_cutoff, naive.n_games_fit, None, None)
    return records


def _last_training_week(seasons: np.ndarray, weeks: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    if not mask.any():
        return (0, 0)
    idx = np.lexsort((weeks[mask], seasons[mask]))
    return (int(seasons[mask][idx[-1]]), int(weeks[mask][idx[-1]]))


def _run_walk_forward_with_elo_config(frame: pl.DataFrame, seasons_arr: np.ndarray, weeks_arr: np.ndarray, holdout_weeks: list[tuple[int, int]], elo_config: EloConfig) -> list[PredictionRecord]:
    all_records: list[PredictionRecord] = []
    all_records += _run_elo(frame, elo_config)

    for season_p, week_p in holdout_weeks:
        train_mask = _training_mask(seasons_arr, weeks_arr, season_p, week_p)
        predict_mask = (seasons_arr == season_p) & (weeks_arr == week_p)
        if predict_mask.sum() == 0:
            continue
        training_cutoff = _last_training_week(seasons_arr, weeks_arr, train_mask)

        all_records += _fit_predict_naive(frame, train_mask, predict_mask, training_cutoff)
        all_records += _fit_predict_ridge(frame, train_mask, predict_mask, FEATURE_SET_LOOKUP["E_add_personnel"], "home_margin", 5.0, "ridge_margin_E_v1", training_cutoff)
        all_records += _fit_predict_ridge(frame, train_mask, predict_mask, FEATURE_SET_LOOKUP["E_add_personnel"], "total_points", 5.0, "ridge_total_E_v1", training_cutoff)
        all_records += _fit_predict_ridge(frame, train_mask, predict_mask, CORE_FEATURES, "home_margin", 5.0, "ridge_margin_CORE_v1", training_cutoff)
        all_records += _fit_predict_logistic(frame, train_mask, predict_mask, FEATURE_SET_LOOKUP["E_add_personnel"], 1.0, "logistic_win_E_v1", training_cutoff)
        all_records += _fit_predict_lightgbm(frame, train_mask, predict_mask, FEATURE_SET_LOOKUP["F_full"], "lightgbm_F_v1", training_cutoff)

        logger.info("Walk-forward week done: season=%d week=%d n_predict=%d training_cutoff=%s", season_p, week_p, int(predict_mask.sum()), training_cutoff)

    return all_records
