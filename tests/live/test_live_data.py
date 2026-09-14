"""Phase 8A Step 28 proofs #2-3: market fields cannot enter independent model features,
post-kickoff data cannot enter a pregame live snapshot (structural: upcoming games' outcome
columns are always null, never fabricated)."""

from __future__ import annotations

import pytest

from nfl_predict.live.live_data import build_live_game_frame, completed_game_mask


def test_upcoming_games_have_null_outcome_columns_never_fabricated():
    ds = build_live_game_frame([2026])
    frame = ds.frame
    upcoming = frame.filter(~completed_game_mask(frame))
    assert upcoming.height > 0
    assert upcoming["home_margin"].null_count() == upcoming.height
    assert upcoming["total_points"].null_count() == upcoming.height
    assert upcoming["home_win"].null_count() == upcoming.height


def test_completed_games_have_real_outcome_columns():
    ds = build_live_game_frame([2026])
    frame = ds.frame
    completed = frame.filter(completed_game_mask(frame))
    assert completed.height >= 1
    assert completed["home_margin"].null_count() == 0


def test_is_tie_is_never_null_after_the_fix_for_the_masking_bug():
    ds = build_live_game_frame([2026])
    assert ds.frame["is_tie"].null_count() == 0


def test_market_fields_are_never_present_in_the_live_frame():
    ds = build_live_game_frame([2026])
    columns_lower = {c.lower() for c in ds.frame.columns}
    for forbidden in ("spread_line", "total_line", "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds"):
        assert forbidden not in columns_lower


def test_build_live_game_frame_covers_every_game_in_the_season():
    ds = build_live_game_frame([2026])
    assert ds.frame.height == 272  # matches the real ingested 2026 schedule
