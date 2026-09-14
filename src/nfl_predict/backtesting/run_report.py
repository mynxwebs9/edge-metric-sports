"""Phase 4 Steps 5-9 + 12 orchestrator: scoring, baseline comparison, bootstrap CIs,
uncertainty validation, and error analysis, all in one place, written to
data/reports/phase4_results.json for docs/PHASE4_BACKTEST_REPORT.md to cite from.

    python -m nfl_predict.backtesting.run_report

Requires data/backtests/predictions/run_id=<run_id>/predictions.parquet to already exist
(`nfl_predict.backtesting.run_backtest`). Reads the ledger and outcomes; writes nothing back
to either - only to data/reports/.
"""

from __future__ import annotations

import json

import numpy as np

from nfl_predict.config import PROJECT_ROOT
from nfl_predict.logging_conf import get_logger

from nfl_predict.backtesting.bootstrap import bootstrap_metric_difference
from nfl_predict.backtesting.error_analysis import run_error_analysis
from nfl_predict.backtesting.scoring import (
    BASELINE_MODEL_IDS,
    _METRIC_KEY_BY_TARGET,
    compare_to_baselines,
    load_scored_frame,
    paired_predictions,
    score_holdout,
)
from nfl_predict.backtesting.uncertainty_validation import run_uncertainty_validation

logger = get_logger(__name__)

RUN_ID = "phase4_v1"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool,)):
        return o
    return str(o)


def run_bootstrap_comparisons(run_id: str, scoring_results: dict) -> dict:
    scored = load_scored_frame(run_id)
    comparisons = {}
    for model_id, targets in scoring_results.items():
        if model_id in BASELINE_MODEL_IDS:
            continue
        for target in targets:
            metric = _METRIC_KEY_BY_TARGET[target]
            for baseline_id in BASELINE_MODEL_IDS:
                if target not in scoring_results.get(baseline_id, {}):
                    continue
                actual, pred_c, pred_b, _ = paired_predictions(scored, model_id, baseline_id, target)
                result = bootstrap_metric_difference(actual, pred_c, pred_b, metric=metric)
                comparisons.setdefault(model_id, {})[f"{target}_vs_{baseline_id}"] = {
                    "metric": result.metric, "n_games": result.n_games, "n_bootstrap": result.n_bootstrap,
                    "point_estimate_diff": result.point_estimate_diff, "ci_low": result.ci_low, "ci_high": result.ci_high,
                    "ci_level": result.ci_level, "excludes_zero": result.excludes_zero,
                    "proportion_resamples_candidate_better": result.proportion_resamples_candidate_better,
                }
    # Baseline-vs-baseline sanity comparison (Elo vs naive), for report context.
    actual, pred_elo, pred_naive, _ = paired_predictions(scored, "elo_v2", "naive_v1", "home_margin")
    r = bootstrap_metric_difference(actual, pred_elo, pred_naive, metric="mae")
    comparisons["elo_v2"] = {"home_margin_vs_naive_v1": {
        "metric": r.metric, "n_games": r.n_games, "n_bootstrap": r.n_bootstrap,
        "point_estimate_diff": r.point_estimate_diff, "ci_low": r.ci_low, "ci_high": r.ci_high,
        "ci_level": r.ci_level, "excludes_zero": r.excludes_zero,
        "proportion_resamples_candidate_better": r.proportion_resamples_candidate_better,
    }}
    return comparisons


def run(run_id: str = RUN_ID) -> dict:
    logger.info("Step 4: holdout scoring")
    scoring_results = score_holdout(run_id)

    logger.info("Step 5: baseline comparison")
    baseline_comparisons = compare_to_baselines(scoring_results)

    logger.info("Step 6: bootstrap confidence intervals")
    bootstrap_comparisons = run_bootstrap_comparisons(run_id, scoring_results)

    logger.info("Step 8: uncertainty validation")
    uncertainty_results = run_uncertainty_validation()

    logger.info("Step 9: error analysis")
    error_analysis_results = run_error_analysis(run_id)

    return {
        "run_id": run_id,
        "scoring": scoring_results,
        "baseline_comparisons": baseline_comparisons,
        "bootstrap_comparisons": bootstrap_comparisons,
        "uncertainty_validation": uncertainty_results,
        "error_analysis": error_analysis_results,
    }


def main() -> int:
    results = run()
    out_dir = PROJECT_ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase4_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    print(json.dumps({"output_path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
