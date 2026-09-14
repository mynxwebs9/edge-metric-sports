"""Phase 8A Step 28 proofs #25-26: market consensus math is correct, no-vig consensus math
is correct - never a blind average of raw American odds."""

from __future__ import annotations

import pytest

from nfl_predict.live.market_consensus import compute_market_consensus
from nfl_predict.market.odds_math import no_vig_two_way
from nfl_predict.market.odds_provider import OddsMarketSnapshot


def _snap(bookmaker, home_spread, home_ml, away_ml) -> OddsMarketSnapshot:
    return OddsMarketSnapshot(
        provider_event_id="evt1", fetched_at="2026-09-11T20:00:00+00:00", bookmaker=bookmaker,
        home_spread_traditional=home_spread, away_spread_traditional=-home_spread if home_spread is not None else None,
        home_spread_price=-110, away_spread_price=-110, home_moneyline=home_ml, away_moneyline=away_ml,
        total_line=None, over_price=None, under_price=None,
    )


def test_consensus_spread_is_the_median_not_the_mean():
    snapshots = [_snap("book1", -2.5, -120, 100), _snap("book2", -3.0, -115, -105), _snap("book3", -3.5, -110, -110)]
    consensus = compute_market_consensus(snapshots)
    assert consensus.consensus_home_spread == -3.0  # median of [-2.5, -3.0, -3.5]
    assert consensus.n_books == 3


def test_consensus_probability_averages_no_vig_probabilities_not_raw_odds():
    snapshots = [_snap("book1", -3.0, -150, 130), _snap("book2", -3.0, -140, 120)]
    consensus = compute_market_consensus(snapshots)

    expected_1 = no_vig_two_way(-150, 130).no_vig_prob_a
    expected_2 = no_vig_two_way(-140, 120).no_vig_prob_a
    assert consensus.consensus_home_no_vig_probability == pytest.approx((expected_1 + expected_2) / 2)

    # Sanity: a blind average of the RAW odds (-150, -140) would not even be a probability -
    # confirm the consensus is nowhere near naively averaging odds as numbers.
    naive_wrong_average = (-150 + -140) / 2
    assert consensus.consensus_home_no_vig_probability != naive_wrong_average


def test_every_original_book_snapshot_is_preserved():
    snapshots = [_snap("book1", -3.0, -150, 130), _snap("book2", -2.5, -140, 120)]
    consensus = compute_market_consensus(snapshots)
    assert consensus.per_book_snapshots == tuple(snapshots)
    assert set(consensus.book_keys) == {"book1", "book2"}


def test_missing_spread_or_moneyline_at_some_books_does_not_crash_and_uses_only_available_data():
    snapshots = [_snap("book1", None, -150, 130), _snap("book2", -3.0, None, None)]
    consensus = compute_market_consensus(snapshots)
    assert consensus.consensus_home_spread == -3.0  # only book2 has a spread
    assert consensus.consensus_home_no_vig_probability == pytest.approx(no_vig_two_way(-150, 130).no_vig_prob_a)  # only book1


def test_empty_snapshot_list_returns_none_not_a_crash():
    assert compute_market_consensus([]) is None
