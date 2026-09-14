"""Phase 8A correction, Steps 1-2: turns a real `OddsProvider` into real per-game
`MarketPoint`s - never `available=True` merely because the provider/key could be
constructed (that is `odds_automation.get_production_odds_provider()`'s job, a separate,
coarser check). Every fetched sportsbook snapshot is persisted through the existing
append-only store (`market.live_snapshot_store`) before it is used, and the consensus
representation follows `market_consensus.py`'s documented no-vig/median method - never a
raw average of American odds.

**Fetch once, reuse many times** (Phase 8A correction, credit-exhaustion postmortem - see
`nfl_predict.market.odds_provider`'s module docstring for the full incident). This module
calls `provider.get_markets_for_sport()` **exactly once** per call to
`fetch_and_snapshot_live_odds()` - a single batched request covering every event of the
sport - and groups the result locally by `provider_event_id`. It never calls
`get_markets(event_id)` in a per-event loop, which is the pattern that once burned an entire
500-credit allowance in one run. `run.py` calls this function exactly once per pipeline run
and reuses its returned `OddsIngestionResult` for every downstream use (research packets,
spread decisions, moneyline decisions, CLI display, market consensus) - see
`tests/live/test_odds_ingestion.py` for the tests proving reuse and the absence of
per-event/per-game HTTP calls.

An event with no matching `game_id` (unmapped team names) or no usable spread/moneyline
data is simply skipped, not fabricated - that game's `MarketPoint` stays `available=False`
downstream (the caller's default), and the skip is visible in `OddsIngestionResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nfl_predict.decision.schemas import MarketPoint
from nfl_predict.live.market_consensus import compute_market_consensus
from nfl_predict.live.odds_mapping import match_odds_event_to_game
from nfl_predict.logging_conf import get_logger
from nfl_predict.market.event_game_mapping import EventGameMapping, append_event_game_mapping_idempotent
from nfl_predict.market.live_snapshot_store import append_live_snapshot_idempotent
from nfl_predict.market.odds_provider import OddsProvider

logger = get_logger(__name__)


@dataclass(frozen=True)
class GameMarketResult:
    game_id: str
    provider_event_id: str
    market: MarketPoint
    n_books: int
    book_keys: tuple[str, ...]
    snapshot_paths: tuple[str, ...]


@dataclass(frozen=True)
class OddsIngestionResult:
    games: dict[str, GameMarketResult]
    events_seen: int
    events_matched: int
    events_unmatched: tuple[str, ...]  # "home @ away" strings for events with no game_id match
    events_without_usable_markets: tuple[str, ...]  # provider_event_ids with a match but no spread/moneyline data
    usage: dict = field(default_factory=dict)  # provider.usage.as_dict() - real x-requests-* headers, never fabricated


def fetch_and_snapshot_live_odds(provider: OddsProvider, season: int, game_ids: list[str] | None = None) -> OddsIngestionResult:
    """`game_ids`, when given, scopes which mapped events are actually processed/persisted
    to this run's target games - it does NOT change the number of HTTP requests made (the
    single batched `get_markets_for_sport()` call already costs the same regardless of how
    many events it covers), it just avoids doing DB-mapping/snapshot-write work for games
    this run doesn't care about."""
    events = provider.get_events()
    all_markets = provider.get_markets_for_sport()  # THE ONE credit-costing call for this entire function
    markets_by_event: dict[str, list] = {}
    for snapshot in all_markets:
        markets_by_event.setdefault(snapshot.provider_event_id, []).append(snapshot)

    games: dict[str, GameMarketResult] = {}
    unmatched: list[str] = []
    no_market: list[str] = []

    for event in events:
        match = match_odds_event_to_game(event.home_team, event.away_team, season)
        if match.status != "matched":
            # "unmatched" (no game_id resolves at all) and "ambiguous" (more than one game
            # matches) are both explicit, distinct outcomes from match_odds_event_to_game -
            # neither is ever silently treated as a real match.
            unmatched.append(f"{event.away_team} @ {event.home_team}")
            continue
        game_id = match.game_id
        if game_ids is not None and game_id not in game_ids:
            continue

        snapshots = markets_by_event.get(event.provider_event_id, [])
        if not snapshots:
            no_market.append(event.provider_event_id)
            continue

        snapshot_paths = []
        for snapshot in snapshots:
            path, _ = append_live_snapshot_idempotent(snapshot)
            snapshot_paths.append(str(path))

        consensus = compute_market_consensus(snapshots)
        has_spread = consensus is not None and consensus.consensus_home_spread is not None
        has_moneyline = consensus is not None and consensus.consensus_home_no_vig_probability is not None
        if not has_spread and not has_moneyline:
            no_market.append(event.provider_event_id)
            continue

        representative = next((s for s in snapshots if s.home_moneyline is not None and s.away_moneyline is not None), snapshots[0])
        source = (
            f"TheOddsAPI consensus of {consensus.n_books} book(s) ({', '.join(consensus.book_keys)}); "
            f"snapshots stored at market/live_snapshots/event={event.provider_event_id}"
        )
        market = MarketPoint(
            available=True,
            source=source,
            home_spread_traditional=consensus.consensus_home_spread,
            home_spread_price=representative.home_spread_price,
            away_spread_price=representative.away_spread_price,
            home_moneyline=representative.home_moneyline,
            away_moneyline=representative.away_moneyline,
            no_vig_home_win_probability=consensus.consensus_home_no_vig_probability,
            snapshot_timestamp=snapshots[0].fetched_at,
            provider_event_id=event.provider_event_id,
            consensus_algorithm_version=consensus.consensus_algorithm_version,
            consensus_book_keys=consensus.book_keys,
            consensus_excluded_book_keys=consensus.excluded_book_keys,
            consensus_exclusion_reason=consensus.exclusion_reason,
            underlying_snapshot_keys=consensus.underlying_snapshot_keys,
            market_snapshot_reference=consensus.snapshot_reference,
        )
        games[game_id] = GameMarketResult(
            game_id=game_id, provider_event_id=event.provider_event_id, market=market,
            n_books=consensus.n_books, book_keys=consensus.book_keys, snapshot_paths=tuple(snapshot_paths),
        )

        # Phase 8A provenance correction: record, additively, which canonical game this
        # provider_event_id resolved to - the ONLY way a later reader with just a game_id can
        # find its way back to the real, persisted `market/live_snapshots/event=...` rows
        # without re-deriving the team-name match (or, worse, guessing) after the fact.
        append_event_game_mapping_idempotent(EventGameMapping(
            provider_event_id=event.provider_event_id, canonical_game_id=game_id,
            season=season, week=match.week,
            provider_home_team=event.home_team, provider_away_team=event.away_team,
            home_team_id=match.home_team_id, away_team_id=match.away_team_id,
            kickoff_timestamp=match.kickoff_timestamp or event.commence_time,
            mapping_timestamp=snapshots[0].fetched_at,
            mapping_method="exact_team_name_match_v1",
        ))

    usage = provider.usage.as_dict() if hasattr(provider, "usage") else {}
    return OddsIngestionResult(
        games=games, events_seen=len(events), events_matched=len(games),
        events_unmatched=tuple(unmatched), events_without_usable_markets=tuple(no_market),
        usage=usage,
    )
