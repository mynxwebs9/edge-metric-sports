"""Phase 5 Step 23 proof #5 (spread disagreement sign convention) + bucket correctness."""

from __future__ import annotations

import numpy as np

from nfl_predict.market.disagreement import bucket_edge_points, edge_direction, edge_points


def test_edge_points_worked_example_from_the_brief():
    # Model fair line BUF -5.5 (home margin +5.5), market BUF -3.0 (home margin +3.0)
    # -> "model favors BUF (home) by an additional 2.5 points" -> edge_points = +2.5.
    edge = edge_points(predicted_home_margin=np.array([5.5]), market_implied_home_margin=np.array([3.0]))
    assert edge[0] == 2.5


def test_edge_points_negative_when_model_favors_away_more():
    edge = edge_points(predicted_home_margin=np.array([1.0]), market_implied_home_margin=np.array([4.0]))
    assert edge[0] == -3.0


def test_edge_direction_matches_sign():
    edge = np.array([2.5, -3.0, 0.0])
    assert edge_direction(edge) == ["model_favors_home_more", "model_favors_away_more", "no_disagreement"]


def test_bucket_edge_points_uses_absolute_value_and_fixed_boundaries():
    edge = np.array([0.5, -1.5, 2.5, -3.5, 10.0, 0.0])
    buckets = bucket_edge_points(edge)
    assert buckets == ["0.00-0.99", "1.00-1.99", "2.00-2.99", "3.00-3.99", "4.00+", "0.00-0.99"]


def test_bucket_boundaries_are_half_open_no_gaps_no_overlaps():
    # Every boundary value itself must land in exactly the bucket that starts at it.
    boundary_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    buckets = bucket_edge_points(boundary_edges)
    assert buckets == ["0.00-0.99", "1.00-1.99", "2.00-2.99", "3.00-3.99", "4.00+"]
