"""Phase 5 Step 4: fair moneyline probability and fair cover probability, derived ONLY from
already-frozen Phase 3/4 model outputs and uncertainty estimates - no new fitting, no
tuning against 2024-2025 outcomes or market data, per the Phase 5 brief's explicit
instruction.

**Point estimate**: a candidate's own frozen, already-generated walk-forward predicted home
margin (read from the Phase 4 prediction ledger - this module never recomputes it).

**Uncertainty**: Phase 3 established a margin residual standard deviation of
`13.059063938633658` (Ridge, F_full features, fit on 2010-2020, checked on 2021-2022 - see
`docs/PHASE3_MODEL_REPORT.md`'s "Predictive uncertainty" section), and Phase 4 Step 8
validated that this estimate generalizes well to the 2024-2025 holdout
(`docs/PHASE4_BACKTEST_REPORT.md`: holdout residual_std `13.327129188072425`, ratio 1.02).
**No Elo-specific (or any other candidate-specific) residual_std was ever established in
Phase 3 or 4.** This module uses that one validated general-margin-error estimate as an
approximation for every candidate's margin uncertainty and says so explicitly here rather
than fabricating per-model precision that was never validated. A zero-mean Normal
approximation is used (Phase 4 found the holdout margin residual mean is small, ~+0.15 for
Elo - not the same finding as the total-points bias Phase 4 investigated and deliberately
left uncorrected).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from nfl_predict.market.odds_math import implied_probability_to_american

# See module docstring for exactly what these are and where they come from.
PHASE3_FROZEN_MARGIN_RESIDUAL_STD = 13.059063938633658
PHASE4_HOLDOUT_VALIDATED_MARGIN_RESIDUAL_STD = 13.327129188072425  # for reporting only, not used as the sigma
DEFAULT_MARGIN_RESIDUAL_STD = PHASE3_FROZEN_MARGIN_RESIDUAL_STD


def fair_home_win_probability(predicted_home_margin: np.ndarray, margin_residual_std: float = DEFAULT_MARGIN_RESIDUAL_STD) -> np.ndarray:
    """P(actual_margin > 0) under a Normal(predicted_home_margin, margin_residual_std)
    approximation - the model's implied fair win probability for the home team."""
    predicted_home_margin = np.asarray(predicted_home_margin, dtype=float)
    return norm.cdf(predicted_home_margin / margin_residual_std)


def fair_home_cover_probability(predicted_home_margin: np.ndarray, home_spread_traditional: np.ndarray, margin_residual_std: float = DEFAULT_MARGIN_RESIDUAL_STD) -> np.ndarray:
    """P(actual_margin > -home_spread_traditional) - the home side covers a spread quoted in
    TRADITIONAL notation, per `docs/MODEL_SPEC.md`'s exact cover-probability convention
    (negative = home favored). Derivation: for X ~ Normal(mu, sigma),
    P(X > -S) = Phi((mu + S) / sigma)."""
    mu = np.asarray(predicted_home_margin, dtype=float)
    s = np.asarray(home_spread_traditional, dtype=float)
    return norm.cdf((mu + s) / margin_residual_std)


def fair_home_spread_traditional(predicted_home_margin: np.ndarray) -> np.ndarray:
    """The spread (traditional sign) at which the model's own point estimate implies a 50%
    cover probability - i.e. the model's fair line. If the model expects home to win by 7,
    its fair home spread is -7 (home favored by 7)."""
    return -np.asarray(predicted_home_margin, dtype=float)


def fair_moneyline_price(prob: np.ndarray) -> list[int]:
    """Expresses a fair win probability as an American moneyline price, elementwise - for
    reporting/comparison against the market's own quoted price, never as a real bet price
    (the model has no vig to charge)."""
    return [implied_probability_to_american(float(p)) for p in np.asarray(prob, dtype=float)]
