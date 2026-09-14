"""Baselines 2-4: regularized linear models (Ridge margin, Ridge total, logistic win prob).

All preprocessing (imputation, scaling) is fitted ONLY on training data via a proper
scikit-learn Pipeline - `pipeline.fit(X_train, ...)` then `pipeline.predict(X_val)`, never
a global impute-then-split. Missingness indicators are preserved (SimpleImputer
`add_indicator=True`) rather than discarded, since which features are missing can itself be
informative (e.g. a team with no snap_counts-era personnel data yet).

Inputs are pandas DataFrames (not polars/numpy) specifically so scikit-learn's
`get_feature_names_out()` can be used to recover which coefficient belongs to which
original feature (or which missingness indicator) after fitting - this is what makes the
coefficient report reviewable rather than a bare unlabeled array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def _to_pandas(X: pl.DataFrame) -> pd.DataFrame:
    return X.to_pandas()


def build_ridge_pipeline(alpha: float = 5.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
    ])


def build_logistic_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_SEED)),
    ])


@dataclass
class FittedLinearModel:
    pipeline: Pipeline
    feature_names_in: list[str]
    feature_names_out: list[str]  # after imputer's missingness indicators
    coefficients: dict[str, float]
    intercept: float
    target: str
    n_train: int


def fit_ridge(X_train: pl.DataFrame, y_train: np.ndarray, alpha: float = 5.0, target: str = "home_margin") -> FittedLinearModel:
    pipeline = build_ridge_pipeline(alpha)
    X_pd = _to_pandas(X_train)
    pipeline.fit(X_pd, y_train)
    names_out = list(pipeline.named_steps["imputer"].get_feature_names_out(X_pd.columns))
    coefs = pipeline.named_steps["ridge"].coef_
    return FittedLinearModel(
        pipeline=pipeline, feature_names_in=list(X_pd.columns), feature_names_out=names_out,
        coefficients=dict(zip(names_out, coefs.tolist())),
        intercept=float(pipeline.named_steps["ridge"].intercept_),
        target=target, n_train=len(y_train),
    )


def fit_logistic(X_train: pl.DataFrame, y_train: np.ndarray, C: float = 1.0, target: str = "home_win") -> FittedLinearModel:
    pipeline = build_logistic_pipeline(C)
    X_pd = _to_pandas(X_train)
    pipeline.fit(X_pd, y_train)
    names_out = list(pipeline.named_steps["imputer"].get_feature_names_out(X_pd.columns))
    coefs = pipeline.named_steps["logreg"].coef_[0]
    return FittedLinearModel(
        pipeline=pipeline, feature_names_in=list(X_pd.columns), feature_names_out=names_out,
        coefficients=dict(zip(names_out, coefs.tolist())),
        intercept=float(pipeline.named_steps["logreg"].intercept_[0]),
        target=target, n_train=len(y_train),
    )


def predict(fitted: FittedLinearModel, X: pl.DataFrame) -> np.ndarray:
    X_pd = _to_pandas(X)
    if list(X_pd.columns) != fitted.feature_names_in:
        X_pd = X_pd[fitted.feature_names_in]
    if hasattr(fitted.pipeline, "predict_proba"):
        return fitted.pipeline.predict_proba(X_pd)[:, 1]
    return fitted.pipeline.predict(X_pd)
