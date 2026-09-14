"""Phase 5 Step 6: model-vs-market disagreement.

`edge_points = predicted_home_margin - market_implied_home_margin`, both already in
margin-space (points of home-team favoring), so the sign is unambiguous regardless of which
team is actually favored: **positive means the model favors the home team MORE than the
market does** (a bigger home favorite, or a smaller home underdog, than the market's own
line implies); negative means the model favors the AWAY team more than the market does.

Worked example from the Phase 5 brief, translated into this convention: model fair line
BUF -5.5 (BUF is home, so `predicted_home_margin = 5.5`), market BUF -3.0
(`market_implied_home_margin = 3.0`) -> `edge_points = 5.5 - 3.0 = +2.5` - "the model favors
BUF [home] by an additional 2.5 points," exactly as stated.

Buckets are the Phase 5 brief's own predetermined boundaries (0-0.99, 1-1.99, 2-2.99,
3-3.99, 4+), fixed BEFORE any betting-result evaluation - `bucket_edge_points` must never be
changed after seeing Step 7-10's results, per the brief's explicit no-threshold-overfitting
instruction (Step 16).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

EDGE_BUCKET_BOUNDARIES = (0.0, 1.0, 2.0, 3.0, 4.0, float("inf"))
EDGE_BUCKET_LABELS = ("0.00-0.99", "1.00-1.99", "2.00-2.99", "3.00-3.99", "4.00+")

Direction = Literal["model_favors_home_more", "model_favors_away_more", "no_disagreement"]


def edge_points(predicted_home_margin: np.ndarray, market_implied_home_margin: np.ndarray) -> np.ndarray:
    return np.asarray(predicted_home_margin, dtype=float) - np.asarray(market_implied_home_margin, dtype=float)


def edge_direction(edge: np.ndarray) -> list[Direction]:
    result: list[Direction] = []
    for e in np.asarray(edge, dtype=float):
        if e > 0:
            result.append("model_favors_home_more")
        elif e < 0:
            result.append("model_favors_away_more")
        else:
            result.append("no_disagreement")
    return result


def bucket_edge_points(edge: np.ndarray) -> list[str]:
    """Buckets by ABSOLUTE disagreement magnitude, using the fixed boundaries above -
    direction is preserved separately via `edge_direction`, never folded into the bucket
    label itself (the brief: "Also preserve direction")."""
    abs_edge = np.abs(np.asarray(edge, dtype=float))
    labels = []
    for e in abs_edge:
        for lo, hi, label in zip(EDGE_BUCKET_BOUNDARIES[:-1], EDGE_BUCKET_BOUNDARIES[1:], EDGE_BUCKET_LABELS):
            if lo <= e < hi:
                labels.append(label)
                break
    return labels
