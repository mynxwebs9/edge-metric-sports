"""Phase 4 Step A: Elo K-factor/home-advantage re-investigation.

Phase 3 selected K=30 from a grid of {10,15,20,25,30} - the edge of the searched range,
meaning the search boundary may have constrained the result. This module expands the grid
conservatively and re-checks, using ONLY data through 2023 (development 2010-2022 +
validation 2023) - never 2024-2025. Two temporal checks are used, not one:

1. The SAME methodology Phase 3 used (sequential log-loss over development 2010-2022 only),
   for direct comparability with the Phase 3 result.
2. A genuine out-of-sample confirmatory check: apply the winning (dev-selected) config to
   2023 (fit continues sequentially from development into 2023, as Elo always does) and
   report ITS log-loss too - this is not used to pick the parameters (that would just be
   validation-season tuning under a different name), only to confirm the dev-selected
   config isn't badly miscalibrated on the one holdout-adjacent season available pre-Phase-4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from nfl_predict.models.elo import EloConfig, EloModel, sort_games_chronologically

# Expanded, but still small and predetermined - per the Phase 4 brief's "documented range,
# do not perform an enormous optimizer search" instruction. Extends past the old edge (30)
# in both directions is unnecessary (10 was never near winning); extending upward is what
# actually tests whether the boundary was real.
K_GRID: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0)
HOME_ADV_GRID: tuple[float, ...] = (25.0, 35.0, 45.0, 55.0, 65.0)


@dataclass(frozen=True)
class EloTuningResult:
    selected_config: EloConfig
    grid_results: list[dict]
    is_k_on_grid_boundary: bool
    is_home_adv_on_grid_boundary: bool
    confirmatory_2023_log_loss: float
    dev_log_loss_of_selected: float


def _dev_log_loss(dev_sorted: pl.DataFrame, config: EloConfig) -> float:
    model = EloModel(config)
    history = model.run_sequential(dev_sorted)
    not_tied = ~dev_sorted["is_tie"].to_numpy()
    y_true = dev_sorted["home_win"].to_numpy()[not_tied].astype(float)
    y_prob = np.clip(history["expected_home_win_prob"].to_numpy()[not_tied], 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def run_elo_tuning(
    dev_targets: pl.DataFrame,
    validation_targets: pl.DataFrame,
    k_grid: tuple[float, ...] = K_GRID,
    home_adv_grid: tuple[float, ...] = HOME_ADV_GRID,
) -> EloTuningResult:
    dev_sorted = sort_games_chronologically(dev_targets)

    grid_results = []
    for k in k_grid:
        for home_adv in home_adv_grid:
            config = EloConfig(k_factor=k, home_field_advantage=home_adv)
            ll = _dev_log_loss(dev_sorted, config)
            grid_results.append({"k_factor": k, "home_field_advantage": home_adv, "dev_log_loss": ll})

    best = min(grid_results, key=lambda r: r["dev_log_loss"])
    selected = EloConfig(k_factor=best["k_factor"], home_field_advantage=best["home_field_advantage"])

    # Confirmatory check: continue the SAME sequential model from development into 2023,
    # scored on 2023 only. Not used to select parameters - selection already happened above,
    # using development data exclusively.
    combined_sorted = sort_games_chronologically(pl.concat([dev_targets, validation_targets], how="diagonal_relaxed"))
    model = EloModel(selected)
    full_history = model.run_sequential(combined_sorted)
    val_mask = (combined_sorted["season"] == validation_targets["season"][0]).to_numpy()
    val_not_tied = val_mask & (~combined_sorted["is_tie"].to_numpy())
    y_true_val = combined_sorted["home_win"].to_numpy()[val_not_tied].astype(float)
    y_prob_val = np.clip(full_history["expected_home_win_prob"].to_numpy()[val_not_tied], 1e-6, 1 - 1e-6)
    confirmatory_ll = float(log_loss(y_true_val, y_prob_val, labels=[0, 1]))

    return EloTuningResult(
        selected_config=selected,
        grid_results=grid_results,
        is_k_on_grid_boundary=(best["k_factor"] in (min(k_grid), max(k_grid))),
        is_home_adv_on_grid_boundary=(best["home_field_advantage"] in (min(home_adv_grid), max(home_adv_grid))),
        confirmatory_2023_log_loss=confirmatory_ll,
        dev_log_loss_of_selected=best["dev_log_loss"],
    )
