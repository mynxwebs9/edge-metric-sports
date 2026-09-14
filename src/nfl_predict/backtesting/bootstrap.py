"""Phase 4 Step 6: bootstrap resampling at the game level for confidence intervals on
MAE/RMSE/Brier DIFFERENCES between two models scored on the identical holdout games.

Resampling is at the GAME level (each bootstrap draw resamples whole games with
replacement, never individual model predictions independently) - the point of Step 6 is to
respect that model comparisons on the same 570 games are paired/dependent observations, not
570 independent trials for each model separately. A confidence interval that excludes zero
is evidence a metric difference is unlikely to be pure noise; one that straddles zero is not
- this module only computes the interval, it never asserts "model X is better."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANDOM_SEED = 42
N_BOOTSTRAP = 2000


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _brier(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((pred - actual) ** 2))


METRIC_FUNCS = {"mae": _mae, "rmse": _rmse, "brier": _brier}


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    n_games: int
    n_bootstrap: int
    point_estimate_diff: float  # metric(candidate) - metric(baseline) on the REAL (unresampled) data
    ci_low: float
    ci_high: float
    ci_level: float
    excludes_zero: bool
    proportion_resamples_candidate_better: float  # fraction of resamples where candidate beat baseline


def bootstrap_metric_difference(
    actual: np.ndarray, pred_candidate: np.ndarray, pred_baseline: np.ndarray,
    metric: str = "mae", n_bootstrap: int = N_BOOTSTRAP, ci_level: float = 0.95, seed: int = RANDOM_SEED,
) -> BootstrapResult:
    actual = np.asarray(actual, dtype=float)
    pred_candidate = np.asarray(pred_candidate, dtype=float)
    pred_baseline = np.asarray(pred_baseline, dtype=float)
    n = len(actual)
    if n == 0:
        raise ValueError("Cannot bootstrap a metric difference on zero games")
    metric_fn = METRIC_FUNCS[metric]

    point_diff = metric_fn(actual, pred_candidate) - metric_fn(actual, pred_baseline)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric_fn(actual[idx], pred_candidate[idx]) - metric_fn(actual[idx], pred_baseline[idx])

    alpha = 1 - ci_level
    lo, hi = (float(x) for x in np.quantile(diffs, [alpha / 2, 1 - alpha / 2]))

    return BootstrapResult(
        metric=metric, n_games=n, n_bootstrap=n_bootstrap, point_estimate_diff=point_diff,
        ci_low=lo, ci_high=hi, ci_level=ci_level, excludes_zero=(lo > 0) or (hi < 0),
        proportion_resamples_candidate_better=float(np.mean(diffs < 0)),
    )
