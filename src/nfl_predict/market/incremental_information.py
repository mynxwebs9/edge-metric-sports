"""Phase 5 Steps 12-13 (combined - Step 12's "start simple" market-aware comparison IS
Step 13's incremental-information test): does an independent model add predictive
information beyond the market?

Fits simple, interpretable OLS regressions - never a complex boosted market-aware model,
per Step 12's explicit "start simple and interpretable" instruction:

    MARKET ONLY:       actual_margin ~ market_implied_margin
    MARKET + <MODEL>:  actual_margin ~ market_implied_margin + model_predicted_margin

evaluated OUT-OF-SAMPLE via a two-way split ACROSS the two holdout seasons (fit on 2024,
evaluate on 2025, and vice versa) - the only genuine out-of-sample split available without
either touching development-period data (which has no walk-forward-generated, genuinely
out-of-sample model predictions to reuse) or refitting/reselecting anything against
2024-2025 outcomes, which Step 15 explicitly forbids treating as validation of a
threshold/model developed on that same data.

This is a plain linear-regression diagnostic, not a new frozen candidate model, and its
result never feeds back into `nfl_predict.models` or `nfl_predict.backtesting`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

FIT_EVAL_SPLITS = ((2024, 2025), (2025, 2024))


@dataclass(frozen=True)
class IncrementalInfoResult:
    model_id: str
    fit_season: int
    eval_season: int
    n_fit: int
    n_eval: int
    market_only_mae: float
    market_plus_model_mae: float
    improvement_mae: float  # market_only_mae - market_plus_model_mae; positive means the model helped
    market_plus_model_coefficients: dict


def _fit_and_eval(train_market, train_model_pred, train_actual, test_market, test_model_pred, test_actual) -> tuple[float, float, dict]:
    market_only = LinearRegression().fit(train_market.reshape(-1, 1), train_actual)
    market_only_pred = market_only.predict(test_market.reshape(-1, 1))
    market_only_mae = float(np.mean(np.abs(market_only_pred - test_actual)))

    X_train = np.column_stack([train_market, train_model_pred])
    X_test = np.column_stack([test_market, test_model_pred])
    combined = LinearRegression().fit(X_train, train_actual)
    combined_pred = combined.predict(X_test)
    combined_mae = float(np.mean(np.abs(combined_pred - test_actual)))

    coefs = {
        "market_coef": float(combined.coef_[0]), "model_coef": float(combined.coef_[1]),
        "intercept": float(combined.intercept_),
    }
    return market_only_mae, combined_mae, coefs


def run_incremental_information_test(
    model_id: str, market_implied_margin: np.ndarray, model_predicted_margin: np.ndarray,
    actual_margin: np.ndarray, seasons: np.ndarray,
) -> list[IncrementalInfoResult]:
    market_implied_margin = np.asarray(market_implied_margin, dtype=float)
    model_predicted_margin = np.asarray(model_predicted_margin, dtype=float)
    actual_margin = np.asarray(actual_margin, dtype=float)
    seasons = np.asarray(seasons)

    results = []
    for fit_season, eval_season in FIT_EVAL_SPLITS:
        fit_mask = seasons == fit_season
        eval_mask = seasons == eval_season
        market_only_mae, combined_mae, coefs = _fit_and_eval(
            market_implied_margin[fit_mask], model_predicted_margin[fit_mask], actual_margin[fit_mask],
            market_implied_margin[eval_mask], model_predicted_margin[eval_mask], actual_margin[eval_mask],
        )
        results.append(IncrementalInfoResult(
            model_id=model_id, fit_season=fit_season, eval_season=eval_season,
            n_fit=int(fit_mask.sum()), n_eval=int(eval_mask.sum()),
            market_only_mae=market_only_mae, market_plus_model_mae=combined_mae,
            improvement_mae=market_only_mae - combined_mae, market_plus_model_coefficients=coefs,
        ))
    return results
