"""Phase 4 Step 4 (holdout scoring) + Step 5 (baseline comparison) + Step 7 (calibration).

Reads the ALREADY-PERSISTED prediction ledger (`ledger.read_prediction_ledger`, which
verifies the ledger's hash before returning anything) and joins it with real 2024-2025
outcomes - which are only ever read here, AFTER predictions exist, never before and never
used to regenerate or adjust a stored prediction. This module never writes to the ledger.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import polars as pl

from nfl_predict.logging_conf import get_logger
from nfl_predict.models import metrics as M
from nfl_predict.models.split import SEALED_HOLDOUT_SEASONS
from nfl_predict.models.targets import TARGET_COLUMNS, read_targets

from nfl_predict.backtesting.freeze import SUBGROUP_DEFINITIONS
from nfl_predict.backtesting.holdout_guard import assert_freeze_complete
from nfl_predict.backtesting.ledger import read_prediction_ledger

logger = get_logger(__name__)

REGRESSION_TARGETS = {"home_margin", "total_points"}
WIN_TARGET = "home_win"

# Every sophisticated model is compared against these two baselines (Step 5).
BASELINE_MODEL_IDS = ("naive_v1", "elo_v2")


def _week_bucket(week: int, season_type: str) -> str | None:
    """None for POST (it has its own subgroup, `season_type_POST` - week number alone isn't
    a meaningful REG-style bucket for the playoffs) - mirrors bias_investigation.py's
    identical helper for consistency across Phase 4 modules."""
    if season_type == "POST":
        return None
    if week <= 4:
        return "weeks_1_4"
    if week <= 10:
        return "weeks_5_10"
    return "weeks_11_plus"


def load_scored_frame(run_id: str) -> pl.DataFrame:
    """The single join point: verified predictions (ledger) + outcomes (targets, read only
    now that predictions already exist) -> one frame with both, for every metric below to
    slice from. Requires the freeze to still verify (defense in depth, not load-bearing here
    since the ledger's own hash check already proves predictions were persisted first)."""
    assert_freeze_complete()
    predictions = read_prediction_ledger(run_id)
    outcomes = read_targets(SEALED_HOLDOUT_SEASONS).select(
        ["game_id", "home_margin", "total_points", "home_win", "is_tie"]
    )
    joined = predictions.join(outcomes, on="game_id", how="inner", suffix="_actual")
    if joined.height != predictions.height:
        raise RuntimeError(
            f"Scoring join dropped rows: {predictions.height} predictions but only "
            f"{joined.height} matched an outcome - every persisted holdout prediction must "
            "have a real completed-game outcome to join against."
        )
    week_buckets = [
        _week_bucket(w, st) for w, st in zip(joined["week"].to_list(), joined["season_type"].to_list())
    ]
    return joined.with_columns(pl.Series("week_bucket", week_buckets))


def _actual_for_target(frame: pl.DataFrame, target: str) -> np.ndarray:
    if target == WIN_TARGET:
        return frame["home_win"].to_numpy().astype(float)
    return frame[target].to_numpy()


def _subgroup_mask(frame: pl.DataFrame, subgroup: str) -> np.ndarray:
    season = frame["season"].to_numpy()
    season_type = frame["season_type"].to_list()
    week_bucket = frame["week_bucket"].to_list()
    if subgroup == "season_2024":
        return season == 2024
    if subgroup == "season_2025":
        return season == 2025
    if subgroup == "combined_2024_2025":
        return np.isin(season, SEALED_HOLDOUT_SEASONS)
    if subgroup == "season_type_REG":
        return np.array([st == "REG" for st in season_type])
    if subgroup == "season_type_POST":
        return np.array([st == "POST" for st in season_type])
    if subgroup in ("weeks_1_4", "weeks_5_10", "weeks_11_plus"):
        return np.array([wb == subgroup for wb in week_bucket])
    raise ValueError(f"Unknown subgroup {subgroup!r} - not in the Step E frozen SUBGROUP_DEFINITIONS")


def _score_one(frame: pl.DataFrame, target: str) -> dict:
    predicted = frame["predicted_value"].to_numpy()
    actual = _actual_for_target(frame, target)
    if target == WIN_TARGET:
        not_tied = ~frame["is_tie"].to_numpy()
        metrics = M.win_probability_metrics(actual[not_tied], predicted[not_tied])
        metrics["calibration_bins"] = M.calibration_bins(actual[not_tied], predicted[not_tied])
        metrics["n_ties_excluded"] = int((~not_tied).sum())
    else:
        metrics = M.regression_metrics(actual, predicted)
    return metrics


def score_holdout(run_id: str) -> dict:
    scored = load_scored_frame(run_id)
    model_target_pairs = scored.select(["model_id", "target"]).unique().sort(["model_id", "target"]).iter_rows()

    results: dict = {}
    for model_id, target in model_target_pairs:
        model_frame = scored.filter((pl.col("model_id") == model_id) & (pl.col("target") == target))
        entry = {
            "overall_2024_2025": _score_one(model_frame, target),
            "by_subgroup": {},
        }
        for subgroup in SUBGROUP_DEFINITIONS:
            mask = _subgroup_mask(model_frame, subgroup)
            sub_frame = model_frame.filter(pl.Series(mask))
            entry["by_subgroup"][subgroup] = _score_one(sub_frame, target) if sub_frame.height else {"n": 0}
        results.setdefault(model_id, {})[target] = entry

    return results


def paired_predictions(scored: pl.DataFrame, model_id: str, baseline_id: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Aligns model_id's and baseline_id's predictions for `target` on the SAME set of
    games (inner-joined on game_id, so a model that skipped a tie for home_win is handled
    correctly by construction rather than by a separate tie filter) - returns
    (actual, pred_candidate, pred_baseline, game_ids), ready for `bootstrap.py`."""
    candidate = scored.filter((pl.col("model_id") == model_id) & (pl.col("target") == target))
    baseline = scored.filter((pl.col("model_id") == baseline_id) & (pl.col("target") == target))
    actual_col = "home_win" if target == WIN_TARGET else target

    joined = candidate.select(["game_id", "predicted_value", actual_col]).join(
        baseline.select(["game_id", "predicted_value"]), on="game_id", suffix="_baseline"
    )
    actual = joined[actual_col].to_numpy().astype(float)
    return actual, joined["predicted_value"].to_numpy(), joined["predicted_value_baseline"].to_numpy(), joined["game_id"].to_list()


_METRIC_KEY_BY_TARGET = {"home_margin": "mae", "total_points": "mae", "home_win": "brier"}


def compare_to_baselines(scoring_results: dict) -> dict:
    """Step 5: every non-baseline model's overall-2024-2025 metric compared against BOTH
    baselines on the SAME target, reported as an absolute difference (candidate - baseline)
    - negative means the candidate beat that baseline on this metric. No claim of
    superiority is made here; Step 6's bootstrap CIs are what determine whether a
    difference is distinguishable from noise."""
    comparisons: dict = {}
    for model_id, targets in scoring_results.items():
        if model_id in BASELINE_MODEL_IDS:
            continue
        for target, entry in targets.items():
            metric_key = _METRIC_KEY_BY_TARGET[target]
            candidate_value = entry["overall_2024_2025"].get(metric_key)
            if candidate_value is None:
                continue
            for baseline_id in BASELINE_MODEL_IDS:
                baseline_entry = scoring_results.get(baseline_id, {}).get(target)
                if baseline_entry is None:
                    continue
                baseline_value = baseline_entry["overall_2024_2025"].get(metric_key)
                if baseline_value is None:
                    continue
                comparisons.setdefault(model_id, {})[f"{target}_vs_{baseline_id}"] = {
                    "metric": metric_key,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "absolute_difference": candidate_value - baseline_value,
                }
    return comparisons
