"""Phase 8A Steps 21-22: current-season prospective performance tracking.

`ALL_MODEL_PREDICTIONS` accuracy (winner accuracy, Brier score, margin MAE) is computed
here, joining `prediction_publication` records with `results_ingestion`'s graded results -
strictly for games that HAVE a graded result; an ungraded (not-yet-played) game's prediction
is excluded from every denominator, never counted as a loss or an unknown. `BEST_BETS`/
`LEANS` win/loss/units/ROI reuse Phase 7's own `nfl_predict.decision.records` functions
directly - never recomputed or duplicated here. The three categories are never mixed.

Per Step 22: this module never backfills a "current-season Best Bets record" from Phase 5's
historical analysis - the official Best Bets record begins only with real, prospectively
published picks (Phase 7's `pick_ledger`), which is empty until one is actually published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_predict.live.results_ingestion import read_graded_result


@dataclass(frozen=True)
class AllModelPredictionsPerformance:
    n_graded: int
    n_ungraded_excluded: int
    winner_accuracy: float | None
    brier_score: float | None
    margin_mae: float | None


def compute_all_model_predictions_performance(predictions: list[dict]) -> AllModelPredictionsPerformance:
    """`predictions`: a list of already-read `prediction_publication` records (each with
    `game_id`, `elo_home_win_probability`, `elo_predicted_margin`, `predicted_winner`)."""
    graded_rows = []
    n_ungraded = 0
    for pred in predictions:
        result = read_graded_result(pred["game_id"])
        if result is None or result.get("is_tie"):
            n_ungraded += 1  # not yet played, OR a tie (no winner to score accuracy against)
            continue
        graded_rows.append((pred, result))

    if not graded_rows:
        return AllModelPredictionsPerformance(n_graded=0, n_ungraded_excluded=n_ungraded, winner_accuracy=None, brier_score=None, margin_mae=None)

    correct = []
    briers = []
    margin_errors = []
    for pred, result in graded_rows:
        actual_home_win = bool(result["home_win"])
        if pred.get("predicted_winner") is not None:
            predicted_home_win = pred["predicted_winner"] == "home"
            correct.append(predicted_home_win == actual_home_win)
        if pred.get("elo_home_win_probability") is not None:
            briers.append((pred["elo_home_win_probability"] - float(actual_home_win)) ** 2)
        if pred.get("elo_predicted_margin") is not None:
            margin_errors.append(abs(pred["elo_predicted_margin"] - result["home_margin"]))

    return AllModelPredictionsPerformance(
        n_graded=len(graded_rows), n_ungraded_excluded=n_ungraded,
        winner_accuracy=float(np.mean(correct)) if correct else None,
        brier_score=float(np.mean(briers)) if briers else None,
        margin_mae=float(np.mean(margin_errors)) if margin_errors else None,
    )
