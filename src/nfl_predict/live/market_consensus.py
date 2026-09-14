"""Phase 8A Step 10: a documented market-consensus representation across multiple
sportsbooks.

Never blindly averages American odds (averaging -110 and +120 as raw numbers is
mathematically meaningless - odds are not on a linear scale). For moneylines, each
book's price is converted to a no-vig probability FIRST (`nfl_predict.market.odds_math`,
already validated in Phase 5), and only the resulting PROBABILITIES are averaged. For
spreads, the median line is reported as the consensus line, alongside every book's own
line/price kept individually - never discarded.

**Phase 8A provenance correction:** every book that contributes is now named explicitly
(`book_keys`, unchanged) alongside `excluded_book_keys`/`exclusion_reason` - today nothing is
ever excluded (every snapshot passed in contributes), so both are always empty/None; the
fields exist so a future exclusion rule (e.g. a stale or outlier book) has somewhere honest
to record itself rather than silently vanishing from the average. `consensus_algorithm_version`
and `snapshot_reference` (a deterministic hash of the provider_event_id plus every
contributing book's own `(bookmaker, fetched_at)` key) let a `MarketPoint` built from this
consensus point back at the EXACT persisted rows it was computed from, without needing to
recompute or guess which of possibly many stored snapshots were used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from statistics import median

from nfl_predict.market.odds_math import no_vig_two_way
from nfl_predict.market.odds_provider import OddsMarketSnapshot

CONSENSUS_ALGORITHM_VERSION = "median_spread_mean_no_vig_v1"


@dataclass(frozen=True)
class MarketConsensus:
    provider_event_id: str
    n_books: int
    book_keys: tuple[str, ...]
    consensus_home_spread: float | None  # median across books, or None if no book has one
    consensus_home_no_vig_probability: float | None  # mean of each book's own no-vig home probability
    per_book_snapshots: tuple[OddsMarketSnapshot, ...]  # every original book record, never discarded
    consensus_algorithm_version: str
    excluded_book_keys: tuple[str, ...]  # always empty today - no exclusion rule exists yet; see module docstring
    exclusion_reason: str | None
    underlying_snapshot_keys: tuple[str, ...]  # "{bookmaker}|{fetched_at}" - the exact (provider_event_id, bookmaker, fetched_at) rows in market/live_snapshots this consensus was computed from
    snapshot_reference: str  # a short, deterministic hash of provider_event_id + underlying_snapshot_keys - a stable pointer to this exact consensus computation


def compute_market_consensus(snapshots: list[OddsMarketSnapshot]) -> MarketConsensus | None:
    if not snapshots:
        return None
    provider_event_id = snapshots[0].provider_event_id

    spreads = [s.home_spread_traditional for s in snapshots if s.home_spread_traditional is not None]
    consensus_spread = float(median(spreads)) if spreads else None

    no_vig_probs = []
    for s in snapshots:
        if s.home_moneyline is not None and s.away_moneyline is not None:
            no_vig_probs.append(no_vig_two_way(s.home_moneyline, s.away_moneyline).no_vig_prob_a)
    consensus_prob = float(sum(no_vig_probs) / len(no_vig_probs)) if no_vig_probs else None

    underlying_snapshot_keys = tuple(f"{s.bookmaker}|{s.fetched_at}" for s in snapshots)
    reference_material = provider_event_id + "|" + "|".join(sorted(underlying_snapshot_keys))
    snapshot_reference = hashlib.sha256(reference_material.encode("utf-8")).hexdigest()[:16]

    return MarketConsensus(
        provider_event_id=provider_event_id, n_books=len(snapshots), book_keys=tuple(s.bookmaker for s in snapshots),
        consensus_home_spread=consensus_spread, consensus_home_no_vig_probability=consensus_prob,
        per_book_snapshots=tuple(snapshots),
        consensus_algorithm_version=CONSENSUS_ALGORITHM_VERSION,
        excluded_book_keys=(), exclusion_reason=None,
        underlying_snapshot_keys=underlying_snapshot_keys, snapshot_reference=snapshot_reference,
    )
