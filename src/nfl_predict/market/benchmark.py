"""Phase 5 Step 5: the sportsbook market benchmarked as a predictor, on the SAME 570-game
2024-2025 holdout Phase 4 already scored the frozen independent models against. Reads the
Phase 4 ledger (via `scoring.load_scored_frame`, which re-verifies the ledger's hash) and the
market_snapshot layer; never mutates either, and never feeds market data back into any
independent-model code path.

The market's own predictions are derived exactly the way Step 4's `fair_line` module derives
the model's: `market_implied_margin = -home_spread_traditional` (the point at which the
market's own line implies a 50% cover probability) and a no-vig moneyline probability from
`odds_math.no_vig_two_way`. This makes the market genuinely comparable to every independent
candidate using the identical `nfl_predict.models.metrics` functions Phase 4 already used -
the market is treated as "just another model" for this comparison, per the brief's explicit
instruction not to assume our system should beat it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nfl_predict.logging_conf import get_logger
from nfl_predict.models import metrics as M
from nfl_predict.models.split import SEALED_HOLDOUT_SEASONS

from nfl_predict.backtesting.scoring import load_scored_frame
from nfl_predict.market.odds_math import no_vig_two_way
from nfl_predict.market.snapshot_store import read_market_snapshot

logger = get_logger(__name__)

MARKET_MARGIN_MODEL_ID = "market_spread"
MARKET_WIN_MODEL_ID = "market_moneyline"
MARKET_TOTAL_MODEL_ID = "market_total"


def game_level_frame(run_id: str) -> pl.DataFrame:
    """One row per holdout game: outcomes + market fields. Any single (model_id, target)
    slice of the scored ledger already has exactly one row per game, so `naive_v1`/
    `home_margin` is used purely as a row-selector to get outcomes without duplicating them
    once per candidate model."""
    scored = load_scored_frame(run_id)
    outcomes = scored.filter((pl.col("model_id") == "naive_v1") & (pl.col("target") == "home_margin")).select(
        ["game_id", "season", "week", "season_type", "home_margin", "total_points", "home_win", "is_tie"]
    )
    market = read_market_snapshot(SEALED_HOLDOUT_SEASONS)
    joined = outcomes.join(market, on="game_id", how="inner")
    if joined.height != outcomes.height:
        raise RuntimeError(f"Market join dropped rows: {outcomes.height} holdout games but only {joined.height} matched a market_snapshot row")
    return joined


def _market_no_vig_home_prob(frame: pl.DataFrame) -> np.ndarray:
    probs = []
    for home_ml, away_ml in zip(frame["home_moneyline"].to_list(), frame["away_moneyline"].to_list()):
        result = no_vig_two_way(home_ml, away_ml)
        probs.append(result.no_vig_prob_a)
    return np.array(probs)


def run_market_benchmark(run_id: str) -> dict:
    frame = game_level_frame(run_id)
    actual_margin = frame["home_margin"].to_numpy()
    actual_total = frame["total_points"].to_numpy()
    actual_win = frame["home_win"].to_numpy().astype(float)
    not_tied = ~frame["is_tie"].to_numpy()

    market_implied_margin = -frame["home_spread_traditional"].to_numpy()
    market_total_line = frame["total_line"].to_numpy()
    market_no_vig_home_prob = _market_no_vig_home_prob(frame)
    market_hold = np.array([
        no_vig_two_way(h, a).bookmaker_hold
        for h, a in zip(frame["home_moneyline"].to_list(), frame["away_moneyline"].to_list())
    ])

    margin_metrics = M.regression_metrics(actual_margin, market_implied_margin)
    total_metrics = M.regression_metrics(actual_total, market_total_line)
    win_metrics = M.win_probability_metrics(actual_win[not_tied], market_no_vig_home_prob[not_tied])
    win_metrics["calibration_bins"] = M.calibration_bins(actual_win[not_tied], market_no_vig_home_prob[not_tied])
    win_metrics["n_ties_excluded"] = int((~not_tied).sum())

    return {
        MARKET_MARGIN_MODEL_ID: {"home_margin": {"overall_2024_2025": margin_metrics}},
        MARKET_WIN_MODEL_ID: {"home_win": {"overall_2024_2025": win_metrics}},
        MARKET_TOTAL_MODEL_ID: {"total_points": {"overall_2024_2025": total_metrics}},
        "average_bookmaker_hold": float(np.mean(market_hold)),
        "n_games": frame.height,
    }
