"""Phase 5 Steps 5-13/19 orchestrator: runs the full market-benchmark analysis against the
frozen Phase 4 predictions and writes data/reports/phase5_results.json.

    python -m nfl_predict.market.run_market_report

Requires the market_snapshot layer (`run_market_ingest`) and the Phase 4 ledger
(`nfl_predict.backtesting.run_backtest`) to already exist. Reads both; writes to neither -
only to data/reports/.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import polars as pl

from nfl_predict.config import PROJECT_ROOT
from nfl_predict.logging_conf import get_logger

from nfl_predict.backtesting.scoring import load_scored_frame
from nfl_predict.market.ats import build_ats_bets, summarize_ats_bets
from nfl_predict.market.benchmark import game_level_frame, run_market_benchmark
from nfl_predict.market.clv import clv_available_for_historical_data
from nfl_predict.market.error_review import select_review_cases
from nfl_predict.market.incremental_information import run_incremental_information_test
from nfl_predict.market.moneyline_edge import build_moneyline_bets, moneyline_edge_bucket_report, summarize_moneyline_bets
from nfl_predict.market.spread_edge import spread_edge_bucket_report

logger = get_logger(__name__)

RUN_ID = "phase4_v1"
MARGIN_CANDIDATE_MODEL_IDS = ["elo_v2", "ridge_margin_E_v1", "ridge_margin_CORE_v1", "lightgbm_F_v1"]
WIN_CANDIDATE_MODEL_IDS = ["elo_v2", "logistic_win_E_v1", "lightgbm_F_v1"]

# Totals: Phase 4 found the independent total-points model does not beat naive. Per Step 14,
# no totals betting strategy is built - only the market benchmark (already computed in
# run_market_benchmark) is reported. This constant documents that conclusion for the report.
TOTALS_CONCLUSION = "DEFERRED / INSUFFICIENT INDEPENDENT SIGNAL"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run(run_id: str = RUN_ID) -> dict:
    logger.info("Step 5: market benchmark")
    benchmark_results = run_market_benchmark(run_id)

    scored = load_scored_frame(run_id)
    market_frame = game_level_frame(run_id)
    market_implied_margin = -market_frame["home_spread_traditional"].to_numpy()
    seasons = market_frame["season"].to_numpy()
    actual_margin = market_frame["home_margin"].to_numpy()

    logger.info("Steps 6-8, 10: spread disagreement, ATS, price-aware ROI, bucket report")
    ats_by_model = {}
    spread_edge_by_model = {}
    all_ats_bets_by_model = {}
    for model_id in MARGIN_CANDIDATE_MODEL_IDS:
        preds = scored.filter((pl.col("model_id") == model_id) & (pl.col("target") == "home_margin")).select(["game_id", "predicted_value"])
        frame = market_frame.join(preds, on="game_id", how="inner")
        bets = build_ats_bets(model_id, frame["predicted_value"].to_numpy(), frame)
        all_ats_bets_by_model[model_id] = bets
        ats_by_model[model_id] = summarize_ats_bets(bets)
        spread_edge_by_model[model_id] = spread_edge_bucket_report(bets)

    logger.info("Step 9: moneyline edge analysis")
    moneyline_by_model = {}
    moneyline_bucket_by_model = {}
    for model_id in WIN_CANDIDATE_MODEL_IDS:
        preds = scored.filter((pl.col("model_id") == model_id) & (pl.col("target") == "home_win")).select(["game_id", "predicted_value"])
        frame = market_frame.join(preds, on="game_id", how="inner")
        bets = build_moneyline_bets(model_id, frame["predicted_value"].to_numpy(), frame)
        moneyline_by_model[model_id] = summarize_moneyline_bets(bets)
        moneyline_bucket_by_model[model_id] = moneyline_edge_bucket_report(bets)

    logger.info("Steps 12-13: incremental information test")
    incremental_by_model = {}
    for model_id in MARGIN_CANDIDATE_MODEL_IDS:
        preds = scored.filter((pl.col("model_id") == model_id) & (pl.col("target") == "home_margin")).select(["game_id", "predicted_value"])
        frame = market_frame.join(preds, on="game_id", how="inner")
        results = run_incremental_information_test(model_id, market_implied_margin, frame["predicted_value"].to_numpy(), actual_margin, seasons)
        incremental_by_model[model_id] = [asdict(r) for r in results]

    logger.info("Step 11: CLV availability check")
    clv_summary = {
        "available_for_historical_data": clv_available_for_historical_data(),
        "reason": "The historical market_snapshot layer has exactly one UNKNOWN-timing snapshot per game (no open/closing pair) - see docs/PHASE5_MARKET_REPORT.md. CLV math is implemented and unit-tested (tests/market/test_clv.py) against synthetic multi-snapshot data and is ready for live storage (Step 22).",
    }

    logger.info("Step 19: representative case review (best margin candidate: elo_v2)")
    error_review = select_review_cases(all_ats_bets_by_model["elo_v2"])
    error_review_summary = {
        category: [{"game_id": b.game_id, "season": b.season, "week": b.week, "edge_points": b.edge_points, "side": b.side, "grade": b.grade, "actual_margin": b.actual_margin, "home_spread_traditional": b.home_spread_traditional} for b in bets]
        for category, bets in error_review.items()
    }

    return {
        "run_id": run_id,
        "market_benchmark": benchmark_results,
        "ats_summary_by_model": ats_by_model,
        "spread_edge_bucket_report_by_model": spread_edge_by_model,
        "moneyline_edge_summary_by_model": moneyline_by_model,
        "moneyline_edge_bucket_report_by_model": moneyline_bucket_by_model,
        "incremental_information_by_model": incremental_by_model,
        "clv": clv_summary,
        "error_review": error_review_summary,
        "totals_conclusion": TOTALS_CONCLUSION,
    }


def main() -> int:
    results = run()
    out_dir = PROJECT_ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase5_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    print(json.dumps({"output_path": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
