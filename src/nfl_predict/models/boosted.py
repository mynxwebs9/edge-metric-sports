"""Model 5: one restrained nonlinear comparison model.

**Chosen: LightGBM**, over XGBoost or HistGradientBoosting, because (a) it's already a
declared project dependency (`pyproject.toml`'s `modeling` extra), (b) it handles missing
values natively (no imputation needed, unlike the Ridge/logistic pipelines - useful given
Phase 2's deliberate nulls for early-season/pre-2013/pre-2011 features), and (c) it trains
fast enough on ~3,500 training rows that a small comparison sweep is cheap. This is a
comparison model, not a replacement for the linear baselines until it demonstrably earns
that on validation metrics (docs/MODEL_SPEC.md's "complexity must earn its place" rule).

Hyperparameters are FIXED and PREDETERMINED, not grid-searched - deliberately conservative
(shallow trees, low learning rate, moderate leaf/sample minimums) to resist overfitting a
~3,500-row training set, per the Phase 3 brief's "do not launch a massive hyperparameter
search" instruction. No tuning against the validation season occurred.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

RANDOM_SEED = 42

DEFAULT_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.03,
    max_depth=4,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=RANDOM_SEED,
    verbosity=-1,
)


@dataclass
class FittedBoostedModel:
    model: object
    feature_names: list[str]
    feature_importances: dict[str, float]
    target: str
    n_train: int
    params: dict


def fit_lgbm_regressor(X_train: pl.DataFrame, y_train: np.ndarray, target: str, params: dict | None = None) -> FittedBoostedModel:
    params = {**DEFAULT_PARAMS, **(params or {})}
    X_pd: pd.DataFrame = X_train.to_pandas()
    model = lgb.LGBMRegressor(**params)
    model.fit(X_pd, y_train)
    importances = dict(zip(X_pd.columns, model.feature_importances_.tolist()))
    return FittedBoostedModel(model=model, feature_names=list(X_pd.columns), feature_importances=importances, target=target, n_train=len(y_train), params=params)


def predict(fitted: FittedBoostedModel, X: pl.DataFrame) -> np.ndarray:
    X_pd = X.to_pandas()[fitted.feature_names]
    return fitted.model.predict(X_pd)
