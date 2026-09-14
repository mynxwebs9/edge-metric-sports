"""Phase 4 Step 8: does the Phase 3 uncertainty estimate generalize to the sealed
2024-2025 holdout?

Phase 3's uncertainty diagnostic (`nfl_predict.models.uncertainty`) fit a Ridge model on
F_full features, using a 2010-2020 FIT chunk and a 2021-2022 OOS-CHECK chunk (both within
development), and reported residual_std=13.06 (margin) / 14.26 (total), residual_mean
~=0 (margin) / -1.17 (total) - see docs/PHASE3_MODEL_REPORT.md. Those are FROZEN numbers;
this module does not recompute or adjust them. It only asks whether holdout residuals from
the SAME model spec (Ridge, alpha=5.0, F_full features) look like they came from the same
distribution the frozen sigma describes.

None of Phase 4's 7 declared candidate models is exactly "Ridge on F_full" (the frozen
candidates use E_add_personnel for Ridge and F_full only for LightGBM) - so this module runs
that one additional, already-fully-specified model spec through the same walk-forward
protocol purely for this like-for-like check. This is NOT a new candidate for Step 10's
model-selection decision; it exists only to answer Step 8's question honestly instead of
comparing Phase 3's F_full-based sigma against a different feature set's residuals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from nfl_predict.logging_conf import get_logger
from nfl_predict.models import uncertainty
from nfl_predict.models.feature_matrix import ABLATIONS
from nfl_predict.models.split import SEALED_HOLDOUT_SEASONS
from nfl_predict.models.targets import read_targets

from nfl_predict.backtesting.holdout_guard import assert_freeze_complete, load_holdout_game_dataset
from nfl_predict.backtesting.walk_forward import (
    ALL_SEASONS,
    PredictionRecord,
    _fit_predict_ridge,
    _holdout_week_sequence,
    _last_training_week,
    _training_mask,
)

logger = get_logger(__name__)

# Frozen in docs/PHASE3_MODEL_REPORT.md / data/reports/phase3_results.json - never recomputed here.
PHASE3_FROZEN_RESIDUAL_STD = {"home_margin": 13.059063938633658, "total_points": 14.264568098186846}
PHASE3_FROZEN_RESIDUAL_MEAN = {"home_margin": -0.015030027892653383, "total_points": -1.1701057669174368}
UNCERTAINTY_CHECK_MODEL_IDS = {"home_margin": "ridge_margin_F_full_uncertainty_check_v1", "total_points": "ridge_total_F_full_uncertainty_check_v1"}


def _run_ridge_ffull_walk_forward() -> list[PredictionRecord]:
    assert_freeze_complete()
    dataset = load_holdout_game_dataset(ALL_SEASONS)
    frame = dataset.frame
    seasons_arr = frame["season"].to_numpy()
    weeks_arr = frame["week"].to_numpy()
    holdout_weeks = _holdout_week_sequence(seasons_arr, weeks_arr)

    records: list[PredictionRecord] = []
    for season_p, week_p in holdout_weeks:
        train_mask = _training_mask(seasons_arr, weeks_arr, season_p, week_p)
        predict_mask = (seasons_arr == season_p) & (weeks_arr == week_p)
        if predict_mask.sum() == 0:
            continue
        cutoff = _last_training_week(seasons_arr, weeks_arr, train_mask)
        records += _fit_predict_ridge(frame, train_mask, predict_mask, ABLATIONS["F_full"], "home_margin", 5.0, UNCERTAINTY_CHECK_MODEL_IDS["home_margin"], cutoff)
        records += _fit_predict_ridge(frame, train_mask, predict_mask, ABLATIONS["F_full"], "total_points", 5.0, UNCERTAINTY_CHECK_MODEL_IDS["total_points"], cutoff)
    return records


def run_uncertainty_validation() -> dict:
    records = _run_ridge_ffull_walk_forward()
    pred_frame = pl.DataFrame([asdict(r) for r in records])
    outcomes = read_targets(SEALED_HOLDOUT_SEASONS).select(["game_id", "home_margin", "total_points"])
    joined = pred_frame.join(outcomes, on="game_id", how="inner")

    results = {}
    for target in ("home_margin", "total_points"):
        sub = joined.filter(pl.col("target") == target)
        actual = sub[target].to_numpy()
        predicted = sub["predicted_value"].to_numpy()
        residual = actual - predicted  # actual - predicted, matching Phase 3's train.py convention
        diag = uncertainty.compute_residual_diagnostics(residual, predicted)

        results[target] = {
            "n": diag.n,
            "holdout_residual_std": diag.residual_std,
            "holdout_residual_mean": diag.residual_mean,
            "phase3_frozen_residual_std": PHASE3_FROZEN_RESIDUAL_STD[target],
            "phase3_frozen_residual_mean": PHASE3_FROZEN_RESIDUAL_MEAN[target],
            "residual_std_ratio_holdout_over_frozen": diag.residual_std / PHASE3_FROZEN_RESIDUAL_STD[target],
            "interval_calibration_using_frozen_sigma": _coverage_using_frozen_sigma(residual, PHASE3_FROZEN_RESIDUAL_STD[target]),
            "holdout_interval_calibration_own_sigma": diag.interval_calibration,
            "holdout_empirical_quantiles": diag.empirical_quantiles,
            "holdout_heteroskedasticity": diag.heteroskedasticity_by_bucket,
        }

    # Early- vs later-season behavior (Step 8 explicitly asks for this split).
    for target in ("home_margin", "total_points"):
        sub = joined.filter(pl.col("target") == target)
        early = sub.filter(pl.col("week") <= 4)
        later = sub.filter(pl.col("week") > 4)
        results[target]["early_season_weeks_1_4"] = _summarize_residuals(early, target)
        results[target]["later_season_weeks_5_plus"] = _summarize_residuals(later, target)

    return results


def _summarize_residuals(sub: pl.DataFrame, target: str) -> dict:
    if sub.height == 0:
        return {"n": 0}
    residual = sub[target].to_numpy() - sub["predicted_value"].to_numpy()
    return {"n": len(residual), "residual_mean": float(np.mean(residual)), "residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else None}


def _coverage_using_frozen_sigma(residual: np.ndarray, frozen_sigma: float) -> list[dict]:
    """Nominal interval coverage IF we had trusted the Phase-3-frozen sigma going into the
    holdout (a Normal interval built from frozen_sigma, not from these holdout residuals) -
    this is the actual test of whether the frozen estimate would have served well live."""
    from scipy.stats import norm

    rows = []
    for level in (0.5, 0.8, 0.9, 0.95):
        z = norm.ppf(0.5 + level / 2)
        half_width = z * frozen_sigma
        covered = float(np.mean((residual >= -half_width) & (residual <= half_width)))
        rows.append({"nominal_coverage": level, "half_width_from_frozen_sigma": half_width, "observed_coverage": covered, "n": len(residual)})
    return rows
