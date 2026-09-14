"""Phase 4 Step B: investigate the ~1-point negative total-points bias found in Phase 3's
uncertainty diagnostics (fit 2010-2020, checked on 2021-2022).

Uses ONLY data through 2023. Runs several walk-forward fit/check windows within that range
to see whether the bias is global (present in every window), era-related (changes
magnitude/sign over time), or tied to the training/check periods' underlying scoring
environment - not just the single window Phase 3 happened to check. Also checks a
different feature set (CORE vs F_full) to see if the bias is feature-set-specific
(model-spec) or persists regardless (more likely a genuine target/era characteristic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_predict.models import linear_models
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES, load_game_dataset, select_features

# (fit_seasons, check_seasons) - all strictly within 2010-2023, each check window
# immediately following its fit window, mimicking a walk-forward re-fit at different points
# in history.
WALK_FORWARD_WINDOWS: list[tuple[list[int], list[int]]] = [
    (list(range(2010, 2016)), [2016, 2017, 2018]),
    (list(range(2010, 2019)), [2019, 2020]),
    (list(range(2010, 2021)), [2021, 2022]),  # Phase 3's original check window
    (list(range(2010, 2023)), [2023]),
]


@dataclass(frozen=True)
class BiasWindowResult:
    fit_seasons: list[int]
    check_seasons: list[int]
    feature_set: str
    n_check: int
    bias: float               # mean(predicted - actual)
    fit_mean_total: float     # mean(total_points) in the fit window
    check_mean_total: float   # mean(total_points) in the check window
    naive_reversion_bias: float  # what the bias would be from pure reversion-to-fit-mean alone
    bias_by_week_bucket: list[dict]
    bias_by_season_type: list[dict]


def _week_bucket(week: int, season_type: str) -> str:
    if season_type == "POST":
        return "POST"
    if week <= 4:
        return "weeks_1_4"
    if week <= 10:
        return "weeks_5_10"
    return "weeks_11_plus"


def _run_one_window(fit_seasons: list[int], check_seasons: list[int], feature_set_name: str, feature_cols: list[str]) -> BiasWindowResult:
    fit_ds = load_game_dataset(fit_seasons)
    check_ds = load_game_dataset(check_seasons)

    X_fit, _ = select_features(fit_ds, feature_cols)
    X_check, _ = select_features(check_ds, feature_cols)
    y_fit = fit_ds.frame["total_points"].to_numpy()
    y_check = check_ds.frame["total_points"].to_numpy()

    fitted = linear_models.fit_ridge(X_fit, y_fit, alpha=5.0, target="total_points")
    pred_check = linear_models.predict(fitted, X_check)
    resid = pred_check - y_check
    bias = float(np.mean(resid))

    fit_mean = float(np.mean(y_fit))
    check_mean = float(np.mean(y_check))
    naive_reversion_bias = fit_mean - check_mean  # predicting the fit-period mean for every game

    meta = check_ds.frame.select(["season_type", "week"]).to_pandas()
    meta["week_bucket"] = [_week_bucket(w, st) for w, st in zip(meta["week"], meta["season_type"])]

    by_week = []
    for bucket in sorted(meta["week_bucket"].unique()):
        mask = (meta["week_bucket"] == bucket).to_numpy()
        if mask.sum() == 0:
            continue
        by_week.append({"bucket": bucket, "n": int(mask.sum()), "bias": float(np.mean(resid[mask]))})

    by_type = []
    for st in sorted(meta["season_type"].unique()):
        mask = (meta["season_type"] == st).to_numpy()
        if mask.sum() == 0:
            continue
        by_type.append({"season_type": st, "n": int(mask.sum()), "bias": float(np.mean(resid[mask]))})

    return BiasWindowResult(
        fit_seasons=fit_seasons, check_seasons=check_seasons, feature_set=feature_set_name,
        n_check=len(y_check), bias=bias, fit_mean_total=fit_mean, check_mean_total=check_mean,
        naive_reversion_bias=naive_reversion_bias, bias_by_week_bucket=by_week, bias_by_season_type=by_type,
    )


def run_bias_investigation() -> list[BiasWindowResult]:
    results = []
    for fit_seasons, check_seasons in WALK_FORWARD_WINDOWS:
        results.append(_run_one_window(fit_seasons, check_seasons, "F_full", ABLATIONS["F_full"]))
    # Model-spec check: re-run the Phase 3 original window with CORE features instead of F_full.
    results.append(_run_one_window(list(range(2010, 2021)), [2021, 2022], "CORE", CORE_FEATURES))
    return results
