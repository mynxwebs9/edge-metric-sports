"""Baseline 0: naive historical-constant model.

No team-specific information at all - exists purely as the floor every other model must
beat. Every constant is computed from TRAINING (development) data only; validation-season
outcomes are never used to compute them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class NaiveBaseline:
    home_margin_mean: float
    total_points_mean: float
    home_win_rate: float  # excludes ties from both numerator and denominator
    n_games_fit: int

    def predict_margin(self, n: int) -> np.ndarray:
        return np.full(n, self.home_margin_mean)

    def predict_total(self, n: int) -> np.ndarray:
        return np.full(n, self.total_points_mean)

    def predict_home_win_prob(self, n: int) -> np.ndarray:
        return np.full(n, self.home_win_rate)


def fit_naive_baseline(train_targets: pl.DataFrame) -> NaiveBaseline:
    non_tied = train_targets.filter(~pl.col("is_tie"))
    home_win_rate = float(non_tied["home_win"].cast(pl.Float64).mean()) if non_tied.height else 0.5
    return NaiveBaseline(
        home_margin_mean=float(train_targets["home_margin"].mean()),
        total_points_mean=float(train_targets["total_points"].mean()),
        home_win_rate=home_win_rate,
        n_games_fit=train_targets.height,
    )
