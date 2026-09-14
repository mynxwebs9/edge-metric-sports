"""Phase 5 Step 3/7/8/9/11: pure math on American odds, no-vig probability, ATS/moneyline
grading, and profit/loss - no I/O, no reading of `market_snapshot` rows. Every function here
takes plain numbers and returns plain numbers so it can be unit-tested exhaustively and
reused identically by ATS, moneyline-edge, and CLV analysis.

## Sign conventions (read this before touching any caller)

- **nflverse's `spread_line`** (raw, as stored in the raw schedules snapshot and in
  `market_snapshot`): **positive means the home team is favored** by that many points,
  negative means the home team is the underdog. This is empirically verified against
  `home_moneyline`/`away_moneyline` direction in the raw data AND confirmed against
  nflverse's own published field dictionary (`nflreadr::dictionary_schedules`): "A positive
  number means the home team was favored by that many points, a negative number means the
  away team was favored by that many points."
- **Traditional bookmaker notation** (and `docs/MODEL_SPEC.md`'s cover-probability
  convention): the OPPOSITE sign - a favorite's quoted spread is negative (e.g. "BUF -3.5").
  `home_spread_traditional = -spread_line`. Every function in this module that takes a
  `home_spread_traditional` argument expects THIS sign, never the raw nflverse one -
  `nflverse_spread_to_traditional_home_spread` is the one place that conversion happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ATSResult = Literal["home_covers", "away_covers", "push"]
Side = Literal["home", "away"]


def american_to_implied_probability(odds: int) -> float:
    """Raw (vig-included) implied win probability of an American odds price. E.g. -110 ->
    0.5238..., +150 -> 0.4."""
    if odds == 0:
        raise ValueError("American odds cannot be 0 - that price is undefined")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def implied_probability_to_american(prob: float) -> int:
    """Inverse of `american_to_implied_probability` - mainly useful for tests and for
    expressing a model's fair probability as a moneyline price for comparison."""
    if not (0.0 < prob < 1.0):
        raise ValueError(f"probability must be strictly between 0 and 1, got {prob}")
    if prob >= 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    return round(100.0 * (1.0 - prob) / prob)


@dataclass(frozen=True)
class NoVigResult:
    raw_implied_prob_a: float
    raw_implied_prob_b: float
    bookmaker_hold: float          # sum of raw implied probs minus 1.0 - the vig, as a fraction
    no_vig_prob_a: float
    no_vig_prob_b: float


def no_vig_two_way(odds_a: int, odds_b: int) -> NoVigResult:
    """Converts a two-sided market's American odds into de-vigged ("fair") probabilities by
    proportionally normalizing the raw implied probabilities so they sum to 1.0 - the
    standard "multiplicative" no-vig method. `odds_a`/`odds_b` are the two sides of the SAME
    market (e.g. home/away moneyline, or favorite/underdog side of a spread)."""
    raw_a = american_to_implied_probability(odds_a)
    raw_b = american_to_implied_probability(odds_b)
    total = raw_a + raw_b
    if total <= 0:
        raise ValueError("Implied probabilities must sum to a positive number")
    return NoVigResult(
        raw_implied_prob_a=raw_a, raw_implied_prob_b=raw_b, bookmaker_hold=total - 1.0,
        no_vig_prob_a=raw_a / total, no_vig_prob_b=raw_b / total,
    )


def break_even_probability(odds: int) -> float:
    """The win rate required to break even betting at this price, repeatedly, at a flat
    stake. Mathematically identical to `american_to_implied_probability` - kept as its own
    named function because Phase 5 uses it in a different conceptual role (how good does MY
    bet's win rate need to be) than the market-reading role of the other function. For -110,
    this is ~52.38%."""
    return american_to_implied_probability(odds)


def moneyline_profit_units(odds: int, won: bool) -> float:
    """Profit in units for a flat 1-unit stake. A loss is always exactly -1 unit; a win
    pays out according to the American price. Used for both real moneyline bets and ATS
    bets (a spread bet is priced in American odds the same way a moneyline bet is)."""
    if not won:
        return -1.0
    if odds > 0:
        return odds / 100.0
    return 100.0 / -odds


def nflverse_spread_to_traditional_home_spread(spread_line: float) -> float:
    """The ONE place the nflverse-sign -> traditional-sign spread conversion happens. See
    the module docstring."""
    return -float(spread_line)


def _side_spread(home_spread_traditional: float, side: Side) -> float:
    return home_spread_traditional if side == "home" else -home_spread_traditional


def grade_ats(actual_margin: float, home_spread_traditional: float) -> ATSResult:
    """`actual_margin` is home_score - away_score. `home_spread_traditional` is the home
    team's quoted spread in TRADITIONAL notation (negative = home favored) - see
    `docs/MODEL_SPEC.md`: home covers iff `actual_margin > -home_spread_traditional`,
    equivalently `actual_margin + home_spread_traditional > 0`."""
    total = actual_margin + home_spread_traditional
    if total > 0:
        return "home_covers"
    if total < 0:
        return "away_covers"
    return "push"


def ats_won(side: Side, grade: ATSResult) -> bool | None:
    """None means the bet pushed (neither a win nor a loss)."""
    if grade == "push":
        return None
    return (side == "home" and grade == "home_covers") or (side == "away" and grade == "away_covers")


def ats_profit_units(side: Side, grade: ATSResult, price: int) -> float:
    won = ats_won(side, grade)
    if won is None:
        return 0.0
    return moneyline_profit_units(price, won)


def spread_clv_points(bet_time_home_spread_traditional: float, closing_home_spread_traditional: float, side: Side) -> float:
    """Closing-line value in points, from the bettor's perspective, for `side`. Positive
    means favorable (the bettor's number was better than what the market later settled on):
    bet the favorite at -2.5, closes at -3.5 -> CLV = (-2.5) - (-3.5) = +1.0 (favorable,
    matching docs/PHASE5_MARKET_REPORT.md's worked example) - the favorite's line moved
    against new bettors after this one got in."""
    bet_line = _side_spread(bet_time_home_spread_traditional, side)
    close_line = _side_spread(closing_home_spread_traditional, side)
    return bet_line - close_line


def moneyline_clv_probability(bet_time_no_vig_prob_for_side: float, closing_no_vig_prob_for_side: float) -> float:
    """Closing-line value in no-vig probability terms, for whichever side was bet. Positive
    means favorable: bet a dog whose no-vig probability was 40% and it closes at 45% (the
    market came to see it as MORE likely to win after this bet was placed) -> CLV =
    0.45 - 0.40 = +0.05 (favorable - the bettor got a cheaper price than the side turned out
    to deserve by closing)."""
    return closing_no_vig_prob_for_side - bet_time_no_vig_prob_for_side
