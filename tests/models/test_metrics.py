"""Sanity checks on the evaluation metric functions themselves."""

from __future__ import annotations

import numpy as np

from nfl_predict.models.metrics import calibration_bins, regression_metrics, win_probability_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([3.0, -5.0, 10.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["bias"] == 0
    assert abs(m["corr"] - 1.0) < 1e-9
    assert m["n"] == 3


def test_regression_metrics_reports_bias_direction():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    m = regression_metrics(y_true, y_pred)
    assert m["bias"] == 2.0
    assert m["mae"] == 2.0


def test_win_probability_metrics_perfect_calibration_gives_low_brier():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    m = win_probability_metrics(y_true, y_prob)
    assert m["brier"] < 1e-6
    assert m["accuracy"] == 1.0


def test_win_probability_metrics_reports_n():
    m = win_probability_metrics(np.array([]), np.array([]))
    assert m["n"] == 0
    assert m["brier"] is None


def test_calibration_bins_report_counts_and_observed_rate():
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.85, 0.1, 0.15, 0.6])
    bins = calibration_bins(y_true, y_prob, bin_edges=[0.0, 0.5, 1.0])
    low_bin = next(b for b in bins if b["bin_low"] == 0.0)
    high_bin = next(b for b in bins if b["bin_low"] == 0.5)
    assert low_bin["n"] == 2
    assert low_bin["observed_rate"] == 0.0
    assert high_bin["n"] == 3
    assert abs(high_bin["observed_rate"] - 1.0) < 1e-9
