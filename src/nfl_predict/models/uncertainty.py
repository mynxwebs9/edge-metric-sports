"""Predictive uncertainty diagnostics (docs/MODEL_SPEC.md's "Predictive uncertainty"
section) - Phase 3 begins this, it does not finish it. No fixed/arbitrary standard
deviation is assumed; everything here is computed from genuinely out-of-sample residuals
within DEVELOPMENT data (2010-2022), never from the validation season (2023) or the sealed
holdout.

Method: split development chronologically into a FIT chunk (2010-2020) and an OOS-CHECK
chunk (2021-2022, still entirely within development, never validation/holdout). A model is
fit on the FIT chunk only, predicts on OOS-CHECK, and residuals (actual - predicted) from
that genuinely-unseen-by-the-fit chunk are what every diagnostic below is built from. This
is a Phase 3 diagnostic, not the final Phase 4 calibration - documented as preliminary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNCERTAINTY_FIT_SEASONS = list(range(2010, 2021))     # 2010-2020
UNCERTAINTY_OOS_CHECK_SEASONS = [2021, 2022]           # still development, held out from the fit chunk only


@dataclass
class ResidualDiagnostics:
    n: int
    residual_std: float          # candidate parametric (Normal) sigma
    residual_mean: float         # should be near 0; a nonzero value flags fit bias
    empirical_quantiles: dict[str, float]  # e.g. "p10" -> residual value at the 10th percentile
    interval_calibration: list[dict]       # nominal coverage vs. observed coverage, WITH counts
    heteroskedasticity_by_bucket: list[dict]  # residual std by |predicted value| bucket


def compute_residual_diagnostics(residuals: np.ndarray, predicted: np.ndarray, interval_levels: list[float] | None = None) -> ResidualDiagnostics:
    residuals = np.asarray(residuals, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(residuals)
    interval_levels = interval_levels or [0.5, 0.8, 0.9, 0.95]

    residual_std = float(np.std(residuals, ddof=1)) if n > 1 else float("nan")
    residual_mean = float(np.mean(residuals)) if n else float("nan")

    quantile_points = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    empirical_quantiles = {f"p{int(q * 100)}": float(np.quantile(residuals, q)) for q in quantile_points} if n else {}

    interval_calibration = []
    from scipy.stats import norm

    for level in interval_levels:
        z = norm.ppf(0.5 + level / 2)
        half_width_normal = z * residual_std
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        half_width_empirical_lo = float(np.quantile(residuals, lo_q))
        half_width_empirical_hi = float(np.quantile(residuals, hi_q))

        covered_normal = float(np.mean((residuals >= -half_width_normal) & (residuals <= half_width_normal))) if n else None
        covered_empirical = float(np.mean((residuals >= half_width_empirical_lo) & (residuals <= half_width_empirical_hi))) if n else None

        interval_calibration.append({
            "nominal_coverage": level, "n": n,
            "normal_half_width": half_width_normal, "normal_observed_coverage": covered_normal,
            "empirical_bounds": (half_width_empirical_lo, half_width_empirical_hi), "empirical_observed_coverage": covered_empirical,
        })

    hetero = []
    if n >= 20:
        order = np.argsort(np.abs(predicted))
        buckets = np.array_split(order, 4)
        for b in buckets:
            if len(b) < 2:
                continue
            hetero.append({
                "n": len(b),
                "abs_pred_range": (float(np.abs(predicted[b]).min()), float(np.abs(predicted[b]).max())),
                "residual_std": float(np.std(residuals[b], ddof=1)),
            })

    return ResidualDiagnostics(
        n=n, residual_std=residual_std, residual_mean=residual_mean,
        empirical_quantiles=empirical_quantiles, interval_calibration=interval_calibration,
        heteroskedasticity_by_bucket=hetero,
    )
