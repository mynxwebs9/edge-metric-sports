"""Evaluation metrics for Phase 3. No ROI, no ATS, no market comparison - see
docs/PHASE3_MODEL_REPORT.md. Every metric function reports its own sample size; none of
this module hides how many rows a number is based on.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, mean bias (pred - actual), and Pearson correlation. `n` is always
    reported alongside - a metric on 12 games is not the same as one on 1,200."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "corr": None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else None
    return {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "corr": corr}


def win_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Brier score, log loss, ROC-AUC (secondary), accuracy at 0.5 (secondary). Calibration
    quality matters more than raw accuracy - see `calibration_bins` for the reliability
    table, reported separately since a single number can't show miscalibration shape."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"n": 0, "brier": None, "log_loss": None, "roc_auc": None, "accuracy": None}
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    brier = float(brier_score_loss(y_true, y_prob_clipped))
    try:
        ll = float(log_loss(y_true, y_prob_clipped, labels=[0, 1]))
    except ValueError:
        ll = None
    try:
        auc = float(roc_auc_score(y_true, y_prob_clipped)) if len(set(y_true.tolist())) > 1 else None
    except ValueError:
        auc = None
    accuracy = float(np.mean((y_prob_clipped >= 0.5).astype(float) == y_true))
    return {"n": n, "brier": brier, "log_loss": ll, "roc_auc": auc, "accuracy": accuracy}


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, bin_edges: list[float] | None = None) -> list[dict]:
    """Reliability table: predicted-probability bucket -> observed win rate, WITH counts.
    Default bins are 10-point buckets from 0 to 1. Tiny bins are reported (with their small
    n visible) rather than hidden - never overinterpreted, per the Phase 3 brief."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if bin_edges is None:
        bin_edges = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        n = int(mask.sum())
        rows.append({
            "bin_low": lo, "bin_high": hi, "n": n,
            "mean_predicted": float(y_prob[mask].mean()) if n else None,
            "observed_rate": float(y_true[mask].mean()) if n else None,
        })
    return rows


def residuals_by_bucket(y_true: np.ndarray, y_pred: np.ndarray, n_buckets: int = 5) -> list[dict]:
    """Error magnitude bucketed by predicted-value magnitude - for inspecting whether a
    margin/total model's error grows for more extreme predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return []
    order = np.argsort(np.abs(y_pred))
    buckets = np.array_split(order, min(n_buckets, n))
    rows = []
    for b in buckets:
        if len(b) == 0:
            continue
        err = y_pred[b] - y_true[b]
        rows.append({
            "n": len(b),
            "pred_abs_range": (float(np.abs(y_pred[b]).min()), float(np.abs(y_pred[b]).max())),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
        })
    return rows
