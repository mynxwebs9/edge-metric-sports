"""Phase 7 Step 24 proofs #12-17, #29: predefined windows, current-streak/last-N/season
calculations, headline cherry-picking is blocked, pushes handled correctly."""

from __future__ import annotations

import pytest

from nfl_predict.decision.streaks import (
    HEADLINE_CATEGORY,
    MIN_SETTLED_FOR_HEADLINE,
    PREDEFINED_WINDOWS,
    UnknownWindowError,
    compute_window,
)


def _pick(settlement, price=-110, settled_at="2026-09-15T00:00:00+00:00", kickoff_at="2026-09-13T17:00:00+00:00", category="BEST_BETS") -> dict:
    return {"category": category, "status": "SETTLED", "settlement": settlement, "price": price, "settled_at": settled_at, "kickoff_at": kickoff_at}


def test_unknown_window_is_rejected():
    with pytest.raises(UnknownWindowError):
        compute_window([], "BEST_BETS", "since_august_29")


def test_arbitrary_date_range_has_no_code_path():
    """There is no function/parameter anywhere in this module that accepts a start/end
    date - only the fixed PREDEFINED_WINDOWS names are ever valid."""
    import inspect

    from nfl_predict.decision import streaks

    sig = inspect.signature(streaks.compute_window)
    assert "start_date" not in sig.parameters
    assert "end_date" not in sig.parameters
    assert set(PREDEFINED_WINDOWS) == {"last_5", "last_10", "last_20", "last_30", "season_to_date", "current_streak"}


def test_current_streak_counts_consecutive_wins_from_the_end():
    picks = [_pick("LOSS", settled_at=f"2026-09-{d:02d}T00:00:00+00:00") for d in range(1, 4)] + [_pick("WIN", settled_at=f"2026-09-{d:02d}T00:00:00+00:00") for d in range(4, 7)]
    result = compute_window(picks, "BEST_BETS", "current_streak")
    assert result.n == 3
    assert result.wins == 3
    assert result.losses == 0


def test_current_streak_skips_over_pushes_without_breaking():
    # WIN, WIN, PUSH, WIN (chronological) -> current streak should be the trailing 3 WINs,
    # with the PUSH counted but not breaking the streak.
    picks = [
        _pick("WIN", settled_at="2026-09-01T00:00:00+00:00"),
        _pick("WIN", settled_at="2026-09-02T00:00:00+00:00"),
        _pick("PUSH", settled_at="2026-09-03T00:00:00+00:00"),
        _pick("WIN", settled_at="2026-09-04T00:00:00+00:00"),
    ]
    result = compute_window(picks, "BEST_BETS", "current_streak")
    assert result.wins == 3
    assert result.losses == 0


def test_current_streak_breaks_on_a_loss():
    picks = [
        _pick("WIN", settled_at="2026-09-01T00:00:00+00:00"),
        _pick("LOSS", settled_at="2026-09-02T00:00:00+00:00"),
    ]
    result = compute_window(picks, "BEST_BETS", "current_streak")
    assert result.losses == 1
    assert result.wins == 0


def test_last_10_takes_the_most_recent_10_chronologically():
    picks = [_pick("WIN" if i % 2 == 0 else "LOSS", settled_at=f"2026-09-{i+1:02d}T00:00:00+00:00") for i in range(15)]
    result = compute_window(picks, "BEST_BETS", "last_10")
    assert result.n == 10
    # last 10 of a 15-length alternating W/L/W/L... sequence starting at index 5 (0-based)
    expected_wins = sum(1 for i in range(5, 15) if i % 2 == 0)
    assert result.wins == expected_wins


def test_last_20_with_fewer_than_20_settled_picks_returns_all_of_them():
    picks = [_pick("WIN", settled_at=f"2026-09-{i+1:02d}T00:00:00+00:00") for i in range(7)]
    result = compute_window(picks, "BEST_BETS", "last_20")
    assert result.n == 7


def test_last_20_takes_exactly_the_most_recent_20_of_a_larger_history():
    # 25 settled picks: first 5 are LOSS (older), last 20 are WIN (more recent).
    picks = [_pick("LOSS", settled_at=f"2026-08-{i+1:02d}T00:00:00+00:00") for i in range(5)]
    picks += [_pick("WIN", settled_at=f"2026-09-{i+1:02d}T00:00:00+00:00") for i in range(20)]
    result = compute_window(picks, "BEST_BETS", "last_20")
    assert result.n == 20
    assert result.wins == 20
    assert result.losses == 0


def test_season_to_date_filters_by_kickoff_year():
    picks = [
        _pick("WIN", kickoff_at="2025-09-10T17:00:00+00:00", settled_at="2025-09-10T20:00:00+00:00"),
        _pick("WIN", kickoff_at="2026-09-13T17:00:00+00:00", settled_at="2026-09-13T20:00:00+00:00"),
        _pick("LOSS", kickoff_at="2026-09-20T17:00:00+00:00", settled_at="2026-09-20T20:00:00+00:00"),
    ]
    result = compute_window(picks, "BEST_BETS", "season_to_date", season=2026)
    assert result.n == 2
    assert result.wins == 1
    assert result.losses == 1


def test_headline_requires_the_minimum_settled_count():
    small = [_pick("WIN", settled_at=f"2026-09-{i+1:02d}T00:00:00+00:00") for i in range(MIN_SETTLED_FOR_HEADLINE - 1)]
    result = compute_window(small, "BEST_BETS", "last_5")
    assert result.headline_eligible is False
    assert result.headline is None


def test_headline_is_only_ever_generated_for_the_best_bets_category():
    picks = [_pick("WIN", settled_at=f"2026-09-{i+1:02d}T00:00:00+00:00", category="ALL_MODEL_PREDICTIONS") for i in range(10)]
    result = compute_window(picks, "ALL_MODEL_PREDICTIONS", "last_10")
    assert result.headline_eligible is False
    assert result.headline is None
    assert HEADLINE_CATEGORY == "BEST_BETS"


def test_headline_text_includes_the_category_and_units():
    picks = [_pick("WIN", settled_at=f"2026-09-{i+1:02d}T00:00:00+00:00") for i in range(10)]
    result = compute_window(picks, "BEST_BETS", "last_10")
    assert result.headline_eligible is True
    assert "Best Bets" in result.headline
    assert "units" in result.headline
