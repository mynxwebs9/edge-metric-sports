"""Phase 5 Step 9: moneyline edge analysis.

`model_edge_probability = model_win_probability_for_side - market_no_vig_probability_for_side`,
where "side" is whichever team the model rates strictly higher than the market's own no-vig
probability - mirroring the spread-edge logic (bet whichever side the model thinks the
market is undervaluing). Edge is always >= 0 by construction once a side is selected; a
game where the model's probability exactly equals the market's no-vig probability produces
no bet, same as a zero-point spread disagreement.

**Ties**: an NFL game that ends in a tie is graded a PUSH on the moneyline (the standard,
real-world sportsbook settlement rule for a tied regulation/overtime game - not fabricated,
but also not documented anywhere in the nflverse data itself, so it's stated here
explicitly as an applied rule, not an empirical finding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_predict.market.odds_math import moneyline_profit_units, no_vig_two_way

EDGE_PROB_BUCKET_BOUNDARIES = (0.0, 0.01, 0.02, 0.03, 0.05, float("inf"))
EDGE_PROB_BUCKET_LABELS = ("<1%", "1-2%", "2-3%", "3-5%", "5%+")


@dataclass(frozen=True)
class MoneylineBetRecord:
    game_id: str
    season: int
    week: int
    season_type: str
    model_id: str
    side: str  # "home" or "away"
    model_prob_for_side: float
    market_no_vig_prob_for_side: float
    edge_probability: float  # always > 0 by construction
    edge_bucket: str
    price: int
    is_tie: bool
    won: bool | None  # None on a push (tie)
    profit_units: float


def _bucket(edge: float) -> str:
    for lo, hi, label in zip(EDGE_PROB_BUCKET_BOUNDARIES[:-1], EDGE_PROB_BUCKET_BOUNDARIES[1:], EDGE_PROB_BUCKET_LABELS):
        if lo <= edge < hi:
            return label
    return EDGE_PROB_BUCKET_LABELS[-1]


def build_moneyline_bets(model_id: str, model_home_win_prob: np.ndarray, frame: pl.DataFrame) -> list[MoneylineBetRecord]:
    """`frame`: one row per game, with game_id/season/week/season_type/home_moneyline/
    away_moneyline/home_win/is_tie, aligned 1:1 with `model_home_win_prob`."""
    game_ids = frame["game_id"].to_list()
    seasons = frame["season"].to_list()
    weeks = frame["week"].to_list()
    season_types = frame["season_type"].to_list()
    home_mls = frame["home_moneyline"].to_list()
    away_mls = frame["away_moneyline"].to_list()
    home_wins = frame["home_win"].to_list()
    is_ties = frame["is_tie"].to_list()

    records = []
    for i in range(frame.height):
        no_vig = no_vig_two_way(home_mls[i], away_mls[i])
        model_home_prob = float(model_home_win_prob[i])
        model_away_prob = 1.0 - model_home_prob

        home_edge = model_home_prob - no_vig.no_vig_prob_a
        away_edge = model_away_prob - no_vig.no_vig_prob_b
        # A tolerance rather than an exact-zero comparison: home_edge and away_edge are
        # always exact negatives of each other by construction (both derived from the same
        # complementary pair of probabilities), so float roundoff can otherwise produce a
        # spurious ~1e-16 "edge" for a game where the model and market genuinely agree.
        if home_edge <= 1e-9 and away_edge <= 1e-9:
            continue

        if home_edge > away_edge:
            side, model_prob, market_prob, edge, price = "home", model_home_prob, no_vig.no_vig_prob_a, home_edge, home_mls[i]
        else:
            side, model_prob, market_prob, edge, price = "away", model_away_prob, no_vig.no_vig_prob_b, away_edge, away_mls[i]

        if is_ties[i]:
            won, profit = None, 0.0
        else:
            actual_home_win = bool(home_wins[i])
            won = (side == "home" and actual_home_win) or (side == "away" and not actual_home_win)
            profit = moneyline_profit_units(price, won)

        records.append(MoneylineBetRecord(
            game_id=game_ids[i], season=seasons[i], week=weeks[i], season_type=season_types[i], model_id=model_id,
            side=side, model_prob_for_side=model_prob, market_no_vig_prob_for_side=market_prob,
            edge_probability=edge, edge_bucket=_bucket(edge), price=int(price), is_tie=bool(is_ties[i]),
            won=won, profit_units=profit,
        ))
    return records


def summarize_moneyline_bets(bets: list[MoneylineBetRecord]) -> dict:
    from nfl_predict.market.ats import bootstrap_roi_ci, wilson_confidence_interval

    if not bets:
        return {"n_bets": 0}
    wins = sum(1 for b in bets if b.won is True)
    losses = sum(1 for b in bets if b.won is False)
    pushes = sum(1 for b in bets if b.won is None)
    win_rate = wins / (wins + losses) if (wins + losses) else None
    ci = wilson_confidence_interval(wins, losses) if (wins + losses) else (None, None)
    mean_expected_prob = float(np.mean([b.model_prob_for_side for b in bets]))
    profits = np.array([b.profit_units for b in bets])
    roi = bootstrap_roi_ci(profits)

    return {
        "n_bets": len(bets), "wins": wins, "losses": losses, "pushes": pushes,
        "actual_win_rate": win_rate, "win_rate_ci_95": ci,
        "mean_model_expected_win_probability": mean_expected_prob,
        "calibration_gap": (win_rate - mean_expected_prob) if win_rate is not None else None,
        "total_units": float(profits.sum()), "roi_per_bet": roi["mean_roi"], "roi_ci_95": (roi["ci_low"], roi["ci_high"]),
    }


def moneyline_edge_bucket_report(bets: list[MoneylineBetRecord]) -> dict:
    report = {}
    for label in EDGE_PROB_BUCKET_LABELS:
        bucket_bets = [b for b in bets if b.edge_bucket == label]
        report[label] = {
            "combined_2024_2025": summarize_moneyline_bets(bucket_bets),
            "season_2024": summarize_moneyline_bets([b for b in bucket_bets if b.season == 2024]),
            "season_2025": summarize_moneyline_bets([b for b in bucket_bets if b.season == 2025]),
        }
    return report
