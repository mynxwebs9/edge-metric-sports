"""Phase 4 Step 9: football-oriented error analysis on PREDEFINED categories - a diagnosis,
not a new fitted model, and never used to alter a stored prediction (STEP 3/4's immutability
still applies here; this module only reads the ledger + outcomes + already-computed Phase 2
features).

Categories (decided from the football domain, not data-mined after seeing results): large
favorites, close games, early- vs late-season, postseason, QB-continuity-change cases,
personnel-continuity availability. The largest-miss inspection pulls REAL per-game context
(overtime, weather/roof/temp/wind, turnovers) from the normalized play-by-play store -
nothing here is fabricated, and none of it feeds back into Phase 4's predictions or metrics.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nfl_predict.logging_conf import get_logger
from nfl_predict.models import metrics as M
from nfl_predict.models.split import SEALED_HOLDOUT_SEASONS

from nfl_predict.backtesting.holdout_guard import assert_freeze_complete, load_holdout_game_dataset
from nfl_predict.backtesting.scoring import load_scored_frame
from nfl_predict.backtesting.walk_forward import ALL_SEASONS

logger = get_logger(__name__)

PRIMARY_MARGIN_MODEL = "ridge_margin_E_v1"
PRIMARY_WIN_MODEL = "logistic_win_E_v1"


def _category_masks(feat_frame: pl.DataFrame, predicted_margin: np.ndarray) -> dict[str, np.ndarray]:
    abs_pred = np.abs(predicted_margin)
    large_favorite_cut = float(np.quantile(abs_pred, 0.75))
    close_game_cut = float(np.quantile(abs_pred, 0.25))

    home_qb_starts = feat_frame["home_qb_consecutive_starts"].to_numpy().astype(float)
    away_qb_starts = feat_frame["away_qb_consecutive_starts"].to_numpy().astype(float)
    qb_change_likely = (np.nan_to_num(home_qb_starts, nan=99) <= 1) | (np.nan_to_num(away_qb_starts, nan=99) <= 1)

    personnel_col = feat_frame["home_off_snap_continuity_pct"].to_numpy().astype(float)
    personnel_available = ~np.isnan(personnel_col)

    season_type = feat_frame["season_type"].to_numpy()
    week = feat_frame["week"].to_numpy()

    return {
        "large_favorites_top_quartile_predicted_margin": abs_pred >= large_favorite_cut,
        "close_games_bottom_quartile_predicted_margin": abs_pred <= close_game_cut,
        "postseason": season_type == "POST",
        "early_season_reg_weeks_1_4": (season_type == "REG") & (week <= 4),
        "late_season_reg_weeks_11_plus": (season_type == "REG") & (week >= 11),
        "qb_continuity_change_likely": qb_change_likely,
        "personnel_continuity_available": personnel_available,
        "personnel_continuity_unavailable": ~personnel_available,
    }


def run_error_analysis(run_id: str) -> dict:
    assert_freeze_complete()
    scored = load_scored_frame(run_id)
    dataset = load_holdout_game_dataset(ALL_SEASONS)
    feat_frame = dataset.frame.filter(pl.col("season").is_in(SEALED_HOLDOUT_SEASONS))

    margin_frame = scored.filter((pl.col("model_id") == PRIMARY_MARGIN_MODEL) & (pl.col("target") == "home_margin"))
    margin_frame = margin_frame.join(
        feat_frame.select(["game_id", "home_qb_consecutive_starts", "away_qb_consecutive_starts", "home_off_snap_continuity_pct"]),
        on="game_id", how="left",
    )
    predicted_margin = margin_frame["predicted_value"].to_numpy()
    actual_margin = margin_frame["home_margin"].to_numpy()
    masks = _category_masks(margin_frame, predicted_margin)

    by_category_margin = {}
    for name, mask in masks.items():
        by_category_margin[name] = M.regression_metrics(actual_margin[mask], predicted_margin[mask])

    win_frame = scored.filter((pl.col("model_id") == PRIMARY_WIN_MODEL) & (pl.col("target") == "home_win"))
    win_frame = win_frame.join(
        feat_frame.select(["game_id", "home_qb_consecutive_starts", "away_qb_consecutive_starts", "home_off_snap_continuity_pct"]),
        on="game_id", how="left",
    )
    predicted_win = win_frame["predicted_value"].to_numpy()
    actual_win = win_frame["home_win"].to_numpy().astype(float)
    not_tied = ~win_frame["is_tie"].to_numpy()
    # Win-prob categories reuse the same predefined definitions (postseason, early/late
    # season, QB-continuity-change, personnel-continuity) computed via a dummy zero margin
    # array (unused by those category definitions); favorite/close-game buckets are instead
    # computed from predicted_win's implied margin proxy - the absolute distance from 0.5 -
    # since a raw predicted margin isn't available on the win-prob-only frame.
    win_masks = _category_masks(win_frame, np.zeros(win_frame.height))
    fav_close_proxy = np.abs(predicted_win - 0.5)
    win_masks["large_favorites_top_quartile_predicted_margin"] = fav_close_proxy >= np.quantile(fav_close_proxy, 0.75)
    win_masks["close_games_bottom_quartile_predicted_margin"] = fav_close_proxy <= np.quantile(fav_close_proxy, 0.25)

    by_category_win = {}
    for name, mask in win_masks.items():
        combined_mask = mask & not_tied
        by_category_win[name] = M.win_probability_metrics(actual_win[combined_mask], predicted_win[combined_mask])

    largest_misses = _largest_misses(margin_frame, actual_margin, predicted_margin, n=15)

    return {
        "margin_model": PRIMARY_MARGIN_MODEL,
        "win_model": PRIMARY_WIN_MODEL,
        "by_category_margin": by_category_margin,
        "by_category_win": by_category_win,
        "largest_margin_misses": largest_misses,
    }


def _largest_misses(margin_frame: pl.DataFrame, actual: np.ndarray, predicted: np.ndarray, n: int) -> list[dict]:
    abs_err = np.abs(predicted - actual)
    order = np.argsort(-abs_err)[:n]
    game_ids = margin_frame["game_id"].to_list()
    seasons = margin_frame["season"].to_list()
    weeks = margin_frame["week"].to_list()
    top_game_ids = [game_ids[i] for i in order]

    pbp_context = _pbp_context_for_games(top_game_ids, sorted(set(seasons[i] for i in order)))

    rows = []
    for i in order:
        gid = game_ids[i]
        row = {
            "game_id": gid, "season": seasons[i], "week": weeks[i],
            "actual_margin": float(actual[i]), "predicted_margin": float(predicted[i]),
            "abs_error": float(abs_err[i]),
        }
        row.update(pbp_context.get(gid, {}))
        rows.append(row)
    return rows


def _pbp_context_for_games(game_ids: list[str], seasons: list[int]) -> dict[str, dict]:
    """Real per-game context pulled from the normalized play-by-play store (overtime via
    max `qtr` >= 5, weather/roof/temp/wind as recorded, turnovers = interceptions +
    fumbles_lost) - purely for human inspection in the Step 9 report, never fed back into
    any prediction or metric."""
    from nfl_predict.config import get_settings

    settings = get_settings()
    frames = []
    for season in seasons:
        path = settings.data_dir / "normalized" / "play_by_play" / f"season={season}" / "data.parquet"
        if path.is_file():
            frames.append(pl.read_parquet(path, memory_map=False, columns=[
                "game_id", "qtr", "interception", "fumble_lost", "weather", "roof", "temp", "wind",
            ]))
    if not frames:
        return {}
    pbp = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("game_id").is_in(game_ids))

    agg = pbp.group_by("game_id").agg([
        pl.col("qtr").max().alias("max_qtr"),
        pl.col("interception").sum().alias("n_interceptions"),
        pl.col("fumble_lost").sum().alias("n_fumbles_lost"),
        pl.col("weather").first().alias("weather"),
        pl.col("roof").first().alias("roof"),
        pl.col("temp").first().alias("temp"),
        pl.col("wind").first().alias("wind"),
    ])

    context = {}
    for row in agg.iter_rows(named=True):
        context[row["game_id"]] = {
            "went_to_overtime": bool(row["max_qtr"] is not None and row["max_qtr"] >= 5),
            "total_turnovers": int((row["n_interceptions"] or 0) + (row["n_fumbles_lost"] or 0)),
            "weather": row["weather"], "roof": row["roof"], "temp": row["temp"], "wind": row["wind"],
        }
    return context
