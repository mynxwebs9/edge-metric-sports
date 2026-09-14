"""Phase 5 Step 23 proof #13 (CLV correctness where multiple snapshots exist) and the
explicit "do not fabricate CLV with one snapshot" requirement."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predict.market.clv import (
    InsufficientSnapshotsError,
    clv_available_for_historical_data,
    compute_moneyline_clv,
    compute_spread_clv,
    proportion_beating_closing_price,
)


def test_clv_not_available_for_the_historical_single_snapshot_layer():
    assert clv_available_for_historical_data() is False


def test_compute_spread_clv_raises_on_a_single_snapshot():
    single = pl.DataFrame({"snapshot_timestamp": ["2024-09-05T10:00:00"], "home_spread_traditional": [-2.5]})
    with pytest.raises(InsufficientSnapshotsError):
        compute_spread_clv(single, bet_time_row_index=0, side="home")


def test_compute_spread_clv_on_synthetic_multi_snapshot_data_matches_the_brief_example():
    snapshots = pl.DataFrame({
        "snapshot_timestamp": ["2024-09-03T10:00:00", "2024-09-08T18:00:00"],
        "home_spread_traditional": [-2.5, -3.5],  # opened -2.5, closed -3.5 (favorite grew)
    })
    clv = compute_spread_clv(snapshots, bet_time_row_index=0, side="home")
    assert clv == pytest.approx(1.0)  # favorable, matching the worked example in odds_math


def test_compute_moneyline_clv_on_synthetic_multi_snapshot_data():
    snapshots = pl.DataFrame({
        "snapshot_timestamp": ["2024-09-03T10:00:00", "2024-09-08T18:00:00"],
        "home_moneyline": [150, 120], "away_moneyline": [-170, -140],
    })
    clv = compute_moneyline_clv(snapshots, bet_time_row_index=0, side="home")
    assert clv > 0  # home's price shortened by closing -> favorable for an earlier bettor


def test_proportion_beating_closing_price_on_empty_input_is_none_not_zero():
    assert proportion_beating_closing_price([]) is None


def test_proportion_beating_closing_price_counts_only_strictly_positive_clv():
    assert proportion_beating_closing_price([1.0, -0.5, 0.0, 2.0]) == pytest.approx(0.5)
