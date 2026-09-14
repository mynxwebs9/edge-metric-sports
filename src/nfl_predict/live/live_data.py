"""Phase 8A Step 4/7: the LIVE model-ready data frame.

`nfl_predict.models.feature_matrix._load_game_dataset_impl` (and everything built on it,
including `nfl_predict.backtesting.holdout_guard.load_holdout_game_dataset`) INNER JOINs
features with targets - which silently DROPS any game with no targets row, i.e. every
upcoming/unplayed game. That is exactly correct for Phase 3/4's historical backtesting
(there is nothing to predict without an outcome to score against) but wrong for live
operation, where the whole point is to predict games that have NOT happened yet.

This module is the live-only equivalent: a LEFT JOIN of the Phase 2 game-level feature
table (which already covers every game, played or not - see Phase 2) against targets
computed fresh from `games` (`build_targets`, which only returns `game_status='final'` rows
by construction). Upcoming games get real features and null outcome columns; nothing here
ever fabricates an outcome. The same market-field denylist guard Phase 3/4 use is applied
here too, unconditionally.
"""

from __future__ import annotations

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.features.registry import assert_no_denylisted_columns
from nfl_predict.features.store import read_feature_table
from nfl_predict.models.feature_matrix import FEATURE_VERSION, ModelDataset
from nfl_predict.models.targets import TARGET_COLUMNS, build_targets


def build_live_game_frame(seasons: list[int]) -> ModelDataset:
    """LEFT JOIN: every game in `seasons` gets its Phase 2 features; only games with
    `game_status='final'` get real outcome columns (home_margin/total_points/home_win/
    is_tie) - everything else is null, never fabricated. Reuses `ModelDataset` so this
    frame is a drop-in argument to any of `nfl_predict.backtesting.walk_forward`'s
    `_fit_predict_*` helpers."""
    conn = get_connection()
    init_schema(conn)
    try:
        targets = build_targets(conn, seasons)
    finally:
        conn.close()

    features = read_feature_table("game", FEATURE_VERSION, seasons)
    assert_no_denylisted_columns(features.columns)

    # features is the LEFT side (every game); targets attaches real outcomes where they
    # exist. Suffix collisions (season/week/season_type appear in both) resolve the same
    # way `_load_game_dataset_impl` already does - features' own columns win, targets'
    # duplicates get a "_targets" suffix and are dropped implicitly by never being selected.
    joined = features.join(targets, on="game_id", how="left", suffix="_targets")

    missing_target_cols = [c for c in TARGET_COLUMNS if c not in joined.columns]
    if missing_target_cols:
        # A season with genuinely zero completed games (e.g. brand new season, week 1
        # before any game has finished) produces an empty `targets` frame with a
        # placeholder Utf8 schema (see `read_targets`/`build_targets`) - polars still
        # names every target column, so this should not normally trigger; defensive only.
        for col in missing_target_cols:
            joined = joined.with_columns(pl.lit(None).alias(col))

    # `is_tie` is null (unknown) for every not-yet-played game. Phase 4's walk-forward
    # helpers (`_fit_predict_logistic`/`_fit_predict_lightgbm`, reused verbatim for live
    # prediction) filter win-probability rows with `~pl.col("is_tie")` - and polars' filter
    # treats a null test result as "exclude the row," so an unfilled null `is_tie` would
    # silently drop every upcoming game from the win-probability prediction set (a real bug
    # caught while building this module - the predict set would come back empty instead of
    # predicted). An unplayed game is factually not a (yet-existing) tie, so filling null
    # with False here states a true fact, not a guess, and only applies to games with no
    # recorded outcome at all (`home_margin` is null too).
    joined = joined.with_columns(pl.col("is_tie").fill_null(False))

    return ModelDataset(seasons=seasons, frame=joined)


def completed_game_mask(frame: pl.DataFrame) -> "pl.Series":
    """True for rows with a real (non-null) outcome - i.e. usable as TRAINING data."""
    return frame["home_margin"].is_not_null()
