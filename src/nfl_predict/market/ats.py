"""Phase 5 Step 7-8: against-the-spread evaluation of every frozen Phase 4 margin candidate,
for the first time in this project using real historical spread outcomes.

**Side selection**: a candidate bets whichever side its own fair line disagrees with the
market toward - `side = "home"` when `edge_points > 0` (the model favors home more than the
market does), `side = "away"` when `edge_points < 0`, and no bet at all when the model's fair
line agrees exactly with the market (extremely rare with continuous predictions).

**Price**: `market_snapshot.home_spread_price`/`away_spread_price` is the REAL recorded
price for every single one of the 570 holdout games (verified: zero nulls, see
`docs/PHASE5_MARKET_REPORT.md`) - this module therefore never needs to fall back to an
assumed price for the primary analysis. A parallel assumed-`-110` analysis is still computed
and returned SEPARATELY (`assumed_price_summary`), clearly labeled, per the brief's explicit
instruction that any assumed-price return must never be blended with observed-price returns
even where real prices exist for comparison purposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfl_predict.market.disagreement import bucket_edge_points, edge_direction, edge_points
from nfl_predict.market.odds_math import ats_profit_units, break_even_probability, grade_ats

ASSUMED_STANDARD_PRICE = -110


@dataclass(frozen=True)
class ATSBetRecord:
    game_id: str
    season: int
    week: int
    season_type: str
    model_id: str
    side: str  # "home" or "away"
    edge_points: float
    edge_bucket: str
    home_spread_traditional: float
    actual_margin: float
    grade: str  # "home_covers" / "away_covers" / "push"
    real_price: int
    real_profit_units: float
    assumed_price: int
    assumed_profit_units: float


def build_ats_bets(model_id: str, predicted_home_margin: np.ndarray, frame: pl.DataFrame) -> list[ATSBetRecord]:
    """`frame` must have game_id/season/week/season_type/home_spread_traditional/
    home_spread_price/away_spread_price/home_margin, one row per game, aligned 1:1 with
    `predicted_home_margin`."""
    market_implied_margin = -frame["home_spread_traditional"].to_numpy()
    edges = edge_points(predicted_home_margin, market_implied_margin)
    directions = edge_direction(edges)
    buckets = bucket_edge_points(edges)
    home_spreads = frame["home_spread_traditional"].to_numpy()
    actual_margins = frame["home_margin"].to_numpy()
    home_prices = frame["home_spread_price"].to_list()
    away_prices = frame["away_spread_price"].to_list()
    game_ids = frame["game_id"].to_list()
    seasons = frame["season"].to_list()
    weeks = frame["week"].to_list()
    season_types = frame["season_type"].to_list()

    records = []
    for i in range(frame.height):
        if directions[i] == "no_disagreement":
            continue
        side = "home" if directions[i] == "model_favors_home_more" else "away"
        grade = grade_ats(actual_margins[i], home_spreads[i])
        real_price = home_prices[i] if side == "home" else away_prices[i]
        records.append(ATSBetRecord(
            game_id=game_ids[i], season=seasons[i], week=weeks[i], season_type=season_types[i],
            model_id=model_id, side=side, edge_points=float(edges[i]), edge_bucket=buckets[i],
            home_spread_traditional=float(home_spreads[i]), actual_margin=float(actual_margins[i]), grade=grade,
            real_price=int(real_price), real_profit_units=ats_profit_units(side, grade, int(real_price)),
            assumed_price=ASSUMED_STANDARD_PRICE, assumed_profit_units=ats_profit_units(side, grade, ASSUMED_STANDARD_PRICE),
        ))
    return records


def wilson_confidence_interval(wins: int, losses: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion - more reliable than the normal
    approximation at the modest sample sizes (tens to a few hundred) Phase 5's edge buckets
    will realistically have."""
    from scipy.stats import norm

    n = wins + losses
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(0.5 + confidence / 2)
    p_hat = wins / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def bootstrap_roi_ci(profit_units: np.ndarray, n_bootstrap: int = 2000, seed: int = 42, ci_level: float = 0.95) -> dict:
    """Bootstrap CI on mean profit-per-bet (equivalently, ROI on a flat 1-unit stake),
    resampling GAMES (bets) with replacement - mirrors `nfl_predict.backtesting.bootstrap`'s
    approach for the same reason: respect that this is a fixed set of real outcomes, not an
    infinite population, and quantify sampling uncertainty honestly."""
    profit_units = np.asarray(profit_units, dtype=float)
    n = len(profit_units)
    if n == 0:
        return {"n": 0, "mean_roi": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = profit_units[idx].mean()
    alpha = 1 - ci_level
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": n, "mean_roi": float(profit_units.mean()), "ci_low": float(lo), "ci_high": float(hi), "ci_level": ci_level, "n_bootstrap": n_bootstrap}


def summarize_ats_bets(bets: list[ATSBetRecord]) -> dict:
    """Returns a dict with `real_price_summary` and `assumed_price_summary` kept entirely
    separate (never blended) - each has n/W/L/P, cover rate + CI, ROI + bootstrap CI, average
    price, and break-even probability at the average price actually used."""
    if not bets:
        return {"n_bets": 0}

    wins = sum(1 for b in bets if b.grade != "push" and ((b.side == "home") == (b.grade == "home_covers")))
    losses = sum(1 for b in bets if b.grade != "push" and ((b.side == "home") != (b.grade == "home_covers")))
    pushes = sum(1 for b in bets if b.grade == "push")
    cover_rate = wins / (wins + losses) if (wins + losses) else None
    ci = wilson_confidence_interval(wins, losses) if (wins + losses) else (None, None)

    real_prices = [b.real_price for b in bets]
    real_profits = np.array([b.real_profit_units for b in bets])
    assumed_profits = np.array([b.assumed_profit_units for b in bets])

    avg_real_price = float(np.mean(real_prices))
    real_roi_ci = bootstrap_roi_ci(real_profits)
    assumed_roi_ci = bootstrap_roi_ci(assumed_profits)

    return {
        "n_bets": len(bets), "wins": wins, "losses": losses, "pushes": pushes,
        "cover_rate": cover_rate, "cover_rate_ci_95": ci,
        "real_price_summary": {
            "average_price": avg_real_price,
            "break_even_probability_at_average_price": break_even_probability(round(avg_real_price)),
            "total_units": float(real_profits.sum()), "roi_per_bet": real_roi_ci["mean_roi"],
            "roi_ci_95": (real_roi_ci["ci_low"], real_roi_ci["ci_high"]),
            "price_source": "real recorded market_snapshot home_spread_price/away_spread_price for every bet",
        },
        "assumed_price_summary": {
            "average_price": ASSUMED_STANDARD_PRICE,
            "break_even_probability_at_average_price": break_even_probability(ASSUMED_STANDARD_PRICE),
            "total_units": float(assumed_profits.sum()), "roi_per_bet": assumed_roi_ci["mean_roi"],
            "roi_ci_95": (assumed_roi_ci["ci_low"], assumed_roi_ci["ci_high"]),
            "price_source": f"ASSUMED constant {ASSUMED_STANDARD_PRICE} for every bet - labeled explicitly, never blended with real_price_summary",
        },
    }
