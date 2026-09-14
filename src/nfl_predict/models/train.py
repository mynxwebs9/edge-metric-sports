"""Phase 3 training/evaluation orchestrator.

    python -m nfl_predict.models.train

Trains every Baseline/Model described in docs/PHASE3_MODEL_REPORT.md on
DEVELOPMENT_SEASONS (2010-2022), evaluates on VALIDATION_SEASON (2023) only, saves model
artifacts, and writes machine-readable results to data/reports/. Never touches
SEALED_HOLDOUT_SEASONS (2024-2025) - see src/nfl_predict/models/split.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from nfl_predict.config import PROJECT_ROOT
from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.logging_conf import get_logger
from nfl_predict.models import baseline_naive, boosted, elo, linear_models, metrics as M, uncertainty
from nfl_predict.models.artifact import save_model_artifact
from nfl_predict.models.feature_matrix import (
    ABLATIONS,
    CORE_FEATURES,
    load_game_dataset,
    select_features,
    without_ties,
)
from nfl_predict.models.split import DEVELOPMENT_SEASONS, VALIDATION_SEASON
from nfl_predict.models.targets import build_targets

logger = get_logger(__name__)
MODEL_VERSION = "v1"

FEATURE_SETS = {**ABLATIONS, "CORE": CORE_FEATURES}


def _week_bucket(week: int, season_type: str) -> str:
    if season_type == "POST":
        return "POST"
    if week <= 4:
        return "REG_weeks_1_4"
    if week <= 10:
        return "REG_weeks_5_10"
    return "REG_weeks_11_plus"


def run() -> dict:
    conn = get_connection()
    init_schema(conn)
    dev_targets_raw = build_targets(conn, DEVELOPMENT_SEASONS)
    val_targets_raw = build_targets(conn, [VALIDATION_SEASON])
    conn.close()

    dev_ds = load_game_dataset(DEVELOPMENT_SEASONS)
    val_ds = load_game_dataset([VALIDATION_SEASON])

    results: dict = {"linear": [], "boosted": [], "naive": {}, "elo": {}, "uncertainty": {}, "predictions": []}

    # ---------------- Baseline 0: naive ----------------
    naive = baseline_naive.fit_naive_baseline(dev_ds.frame)
    val_n = val_ds.frame.height
    naive_margin_pred = naive.predict_margin(val_n)
    naive_total_pred = naive.predict_total(val_n)
    naive_win_pred = naive.predict_home_win_prob(val_n)
    results["naive"] = {
        "params": {"home_margin_mean": naive.home_margin_mean, "total_points_mean": naive.total_points_mean, "home_win_rate": naive.home_win_rate, "n_train": naive.n_games_fit},
        "margin": M.regression_metrics(val_ds.frame["home_margin"].to_numpy(), naive_margin_pred),
        "total": M.regression_metrics(val_ds.frame["total_points"].to_numpy(), naive_total_pred),
    }
    val_no_tie_mask = ~val_ds.frame["is_tie"].to_numpy()
    results["naive"]["win"] = M.win_probability_metrics(
        val_ds.frame["home_win"].to_numpy()[val_no_tie_mask].astype(float), naive_win_pred[val_no_tie_mask]
    )

    # ---------------- Baseline 1: Elo ----------------
    dev_sorted = elo.sort_games_chronologically(dev_targets_raw)
    best_config, grid_results = elo.grid_search_elo_params(dev_sorted)
    logger.info("elo grid search complete", extra={"best_k": best_config.k_factor, "best_home_adv": best_config.home_field_advantage})

    full_targets_raw = build_targets(get_connection(), DEVELOPMENT_SEASONS + [VALIDATION_SEASON])
    full_sorted = elo.sort_games_chronologically(full_targets_raw)
    model = elo.EloModel(best_config)
    elo_history = model.run_sequential(full_sorted)
    slope, intercept = elo.fit_margin_transform(
        elo_history.filter(pl.col("season").is_in(DEVELOPMENT_SEASONS)),
        dev_sorted["home_margin"].to_numpy(),
    )
    val_elo_hist = elo_history.filter(pl.col("season") == VALIDATION_SEASON)
    val_sorted_targets = full_sorted.filter(pl.col("season") == VALIDATION_SEASON)
    val_elo_margin_pred = elo.apply_margin_transform(val_elo_hist["elo_diff_pre"].to_numpy(), slope, intercept)
    val_elo_win_pred = val_elo_hist["expected_home_win_prob"].to_numpy()
    val_no_tie_mask2 = ~val_sorted_targets["is_tie"].to_numpy()

    results["elo"] = {
        "params": {"k_factor": best_config.k_factor, "home_field_advantage": best_config.home_field_advantage, "margin_slope": slope, "margin_intercept": intercept},
        "grid_search": grid_results,
        "margin": M.regression_metrics(val_sorted_targets["home_margin"].to_numpy(), val_elo_margin_pred),
        "win": M.win_probability_metrics(val_sorted_targets["home_win"].to_numpy()[val_no_tie_mask2].astype(float), val_elo_win_pred[val_no_tie_mask2]),
    }

    # ---------------- Baselines 2-4: Ridge margin/total, Logistic win, per feature set ----------------
    dev_ds_no_tie = without_ties(dev_ds)
    val_ds_no_tie = without_ties(val_ds)

    for fs_name, cols in FEATURE_SETS.items():
        X_dev, _ = select_features(dev_ds, cols)
        X_val, _ = select_features(val_ds, cols)
        y_margin_dev = dev_ds.frame["home_margin"].to_numpy()
        y_total_dev = dev_ds.frame["total_points"].to_numpy()

        ridge_margin = linear_models.fit_ridge(X_dev, y_margin_dev, alpha=5.0, target="home_margin")
        pred_margin = linear_models.predict(ridge_margin, X_val)
        ridge_total = linear_models.fit_ridge(X_dev, y_total_dev, alpha=5.0, target="total_points")
        pred_total = linear_models.predict(ridge_total, X_val)

        X_dev_nt, _ = select_features(dev_ds_no_tie, cols)
        X_val_nt, _ = select_features(val_ds_no_tie, cols)
        y_win_dev = dev_ds_no_tie.frame["home_win"].to_numpy().astype(float)
        y_win_val = val_ds_no_tie.frame["home_win"].to_numpy().astype(float)
        logit = linear_models.fit_logistic(X_dev_nt, y_win_dev, C=1.0, target="home_win")
        pred_win = linear_models.predict(logit, X_val_nt)

        entry = {
            "feature_set": fs_name, "n_features": len(cols),
            "margin_metrics": M.regression_metrics(val_ds.frame["home_margin"].to_numpy(), pred_margin),
            "total_metrics": M.regression_metrics(val_ds.frame["total_points"].to_numpy(), pred_total),
            "win_metrics": M.win_probability_metrics(y_win_val, pred_win),
            "margin_coefficients": ridge_margin.coefficients,
            "win_coefficients": logit.coefficients,
        }
        results["linear"].append(entry)

        save_model_artifact(
            ridge_margin.pipeline, model_id=f"ridge_margin_{fs_name}", model_type="ridge", model_version=MODEL_VERSION,
            family="margin", feature_version="v1", feature_names=cols, training_seasons=DEVELOPMENT_SEASONS,
            validation_season=VALIDATION_SEASON, training_row_count=ridge_margin.n_train,
            hyperparameters={"alpha": 5.0}, preprocessing="median-impute+indicator, standardize (fit on train only)",
            target="home_margin", metrics=entry["margin_metrics"], project_root=PROJECT_ROOT,
        )

        if fs_name == "F_full":
            results["predictions"].append({
                "feature_set": fs_name, "model": "ridge",
                "game_id": val_ds.frame["game_id"].to_list(), "pred_margin": pred_margin.tolist(), "pred_total": pred_total.tolist(),
            })

    # ---------------- Model 5: LightGBM (F_full only) ----------------
    X_dev_full, cols_full = select_features(dev_ds, ABLATIONS["F_full"])
    X_val_full, _ = select_features(val_ds, ABLATIONS["F_full"])
    lgbm_margin = boosted.fit_lgbm_regressor(X_dev_full, dev_ds.frame["home_margin"].to_numpy(), target="home_margin")
    pred_margin_lgbm = boosted.predict(lgbm_margin, X_val_full)
    lgbm_total = boosted.fit_lgbm_regressor(X_dev_full, dev_ds.frame["total_points"].to_numpy(), target="total_points")
    pred_total_lgbm = boosted.predict(lgbm_total, X_val_full)

    import lightgbm as lgb
    X_dev_full_nt, _ = select_features(dev_ds_no_tie, ABLATIONS["F_full"])
    X_val_full_nt, _ = select_features(val_ds_no_tie, ABLATIONS["F_full"])
    lgbm_win = lgb.LGBMClassifier(**{k: v for k, v in boosted.DEFAULT_PARAMS.items()})
    lgbm_win.fit(X_dev_full_nt.to_pandas(), dev_ds_no_tie.frame["home_win"].to_numpy())
    pred_win_lgbm = lgbm_win.predict_proba(X_val_full_nt.to_pandas())[:, 1]

    results["boosted"] = {
        "params": boosted.DEFAULT_PARAMS,
        "margin_metrics": M.regression_metrics(val_ds.frame["home_margin"].to_numpy(), pred_margin_lgbm),
        "total_metrics": M.regression_metrics(val_ds.frame["total_points"].to_numpy(), pred_total_lgbm),
        "win_metrics": M.win_probability_metrics(val_ds_no_tie.frame["home_win"].to_numpy().astype(float), pred_win_lgbm),
        "margin_importance": lgbm_margin.feature_importances,
    }
    save_model_artifact(
        lgbm_margin.model, model_id="lgbm_margin_F_full", model_type="lightgbm", model_version=MODEL_VERSION,
        family="margin", feature_version="v1", feature_names=cols_full, training_seasons=DEVELOPMENT_SEASONS,
        validation_season=VALIDATION_SEASON, training_row_count=lgbm_margin.n_train,
        hyperparameters=lgbm_margin.params, preprocessing="none (LightGBM native NaN handling)",
        target="home_margin", metrics=results["boosted"]["margin_metrics"], project_root=PROJECT_ROOT,
    )

    # ---------------- Calibration (best win model = logistic F_full) ----------------
    f_full_entry = next(e for e in results["linear"] if e["feature_set"] == "F_full")
    X_val_nt_f, _ = select_features(val_ds_no_tie, ABLATIONS["F_full"])
    X_dev_nt_f, _ = select_features(dev_ds_no_tie, ABLATIONS["F_full"])
    logit_f = linear_models.fit_logistic(X_dev_nt_f, dev_ds_no_tie.frame["home_win"].to_numpy().astype(float), target="home_win")
    pred_win_f = linear_models.predict(logit_f, X_val_nt_f)
    results["calibration_logistic_F_full"] = M.calibration_bins(val_ds_no_tie.frame["home_win"].to_numpy().astype(float), pred_win_f)
    results["calibration_lgbm_F_full"] = M.calibration_bins(val_ds_no_tie.frame["home_win"].to_numpy().astype(float), pred_win_lgbm)

    # ---------------- Season-type / week-range breakdowns (Ridge F_full margin, Logistic F_full win) ----------------
    val_meta = val_ds.frame.select(["game_id", "season_type", "week"]).to_pandas()
    val_meta["week_bucket"] = [_week_bucket(w, st) for w, st in zip(val_meta["week"], val_meta["season_type"])]
    ridge_margin_f = linear_models.fit_ridge(X_dev_full, dev_ds.frame["home_margin"].to_numpy(), alpha=5.0, target="home_margin")
    pred_margin_f_full = linear_models.predict(ridge_margin_f, X_val_full)
    actual_margin = val_ds.frame["home_margin"].to_numpy()

    breakdown = []
    for seg_col, seg_name in [("season_type", "season_type"), ("week_bucket", "week_bucket")]:
        for seg_val in val_meta[seg_col].unique():
            mask = (val_meta[seg_col] == seg_val).to_numpy()
            breakdown.append({
                "segment_type": seg_name, "segment_value": str(seg_val),
                **M.regression_metrics(actual_margin[mask], pred_margin_f_full[mask]),
            })
    results["margin_breakdown_F_full"] = breakdown

    # ---------------- Uncertainty diagnostics (margin & total, ridge F_full) ----------------
    fit_ds = load_game_dataset(uncertainty.UNCERTAINTY_FIT_SEASONS)
    oos_ds = load_game_dataset(uncertainty.UNCERTAINTY_OOS_CHECK_SEASONS)
    X_fit, _ = select_features(fit_ds, ABLATIONS["F_full"])
    X_oos, _ = select_features(oos_ds, ABLATIONS["F_full"])

    margin_model_u = linear_models.fit_ridge(X_fit, fit_ds.frame["home_margin"].to_numpy(), alpha=5.0, target="home_margin")
    margin_pred_oos = linear_models.predict(margin_model_u, X_oos)
    margin_resid_oos = oos_ds.frame["home_margin"].to_numpy() - margin_pred_oos
    margin_diag = uncertainty.compute_residual_diagnostics(margin_resid_oos, margin_pred_oos)

    total_model_u = linear_models.fit_ridge(X_fit, fit_ds.frame["total_points"].to_numpy(), alpha=5.0, target="total_points")
    total_pred_oos = linear_models.predict(total_model_u, X_oos)
    total_resid_oos = oos_ds.frame["total_points"].to_numpy() - total_pred_oos
    total_diag = uncertainty.compute_residual_diagnostics(total_resid_oos, total_pred_oos)

    results["uncertainty"] = {
        "margin": {"n": margin_diag.n, "residual_std": margin_diag.residual_std, "residual_mean": margin_diag.residual_mean,
                   "empirical_quantiles": margin_diag.empirical_quantiles, "interval_calibration": margin_diag.interval_calibration,
                   "heteroskedasticity": margin_diag.heteroskedasticity_by_bucket},
        "total": {"n": total_diag.n, "residual_std": total_diag.residual_std, "residual_mean": total_diag.residual_mean,
                  "empirical_quantiles": total_diag.empirical_quantiles, "interval_calibration": total_diag.interval_calibration,
                  "heteroskedasticity": total_diag.heteroskedasticity_by_bucket},
    }

    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    results = run()
    out_dir = PROJECT_ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase3_results.json").write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")

    rows = []
    for e in results["linear"]:
        rows.append({"model": "ridge/logistic", "feature_set": e["feature_set"], "n_features": e["n_features"],
                      "margin_mae": e["margin_metrics"]["mae"], "margin_rmse": e["margin_metrics"]["rmse"],
                      "total_mae": e["total_metrics"]["mae"], "total_rmse": e["total_metrics"]["rmse"],
                      "win_brier": e["win_metrics"]["brier"], "win_log_loss": e["win_metrics"]["log_loss"],
                      "win_auc": e["win_metrics"]["roc_auc"], "win_accuracy": e["win_metrics"]["accuracy"],
                      "n_val": e["margin_metrics"]["n"]})
    rows.append({"model": "naive", "feature_set": "-", "n_features": 0,
                 "margin_mae": results["naive"]["margin"]["mae"], "margin_rmse": results["naive"]["margin"]["rmse"],
                 "total_mae": results["naive"]["total"]["mae"], "total_rmse": results["naive"]["total"]["rmse"],
                 "win_brier": results["naive"]["win"]["brier"], "win_log_loss": results["naive"]["win"]["log_loss"],
                 "win_auc": results["naive"]["win"]["roc_auc"], "win_accuracy": results["naive"]["win"]["accuracy"],
                 "n_val": results["naive"]["margin"]["n"]})
    rows.append({"model": "elo", "feature_set": "-", "n_features": 0,
                 "margin_mae": results["elo"]["margin"]["mae"], "margin_rmse": results["elo"]["margin"]["rmse"],
                 "total_mae": None, "total_rmse": None,
                 "win_brier": results["elo"]["win"]["brier"], "win_log_loss": results["elo"]["win"]["log_loss"],
                 "win_auc": results["elo"]["win"]["roc_auc"], "win_accuracy": results["elo"]["win"]["accuracy"],
                 "n_val": results["elo"]["margin"]["n"]})
    rows.append({"model": "lightgbm", "feature_set": "F_full", "n_features": len(ABLATIONS["F_full"]),
                 "margin_mae": results["boosted"]["margin_metrics"]["mae"], "margin_rmse": results["boosted"]["margin_metrics"]["rmse"],
                 "total_mae": results["boosted"]["total_metrics"]["mae"], "total_rmse": results["boosted"]["total_metrics"]["rmse"],
                 "win_brier": results["boosted"]["win_metrics"]["brier"], "win_log_loss": results["boosted"]["win_metrics"]["log_loss"],
                 "win_auc": results["boosted"]["win_metrics"]["roc_auc"], "win_accuracy": results["boosted"]["win_metrics"]["accuracy"],
                 "n_val": results["boosted"]["margin_metrics"]["n"]})

    pd.DataFrame(rows).to_csv(out_dir / "phase3_model_comparison.csv", index=False)
    pd.DataFrame(results["margin_breakdown_F_full"]).to_csv(out_dir / "phase3_segment_breakdown.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
