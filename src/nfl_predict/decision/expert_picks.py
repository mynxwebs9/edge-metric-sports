"""Expert picks: a human's own spread/moneyline picks, published into the SAME immutable
pick ledger as the model's picks (`PickCategory.EXPERT_PICKS`), so they are settled by the
same `settle_all_pending_picks()` pass and scored by the same price-aware record math - a
separate record, never blended into Best Bets or All Model Predictions.

What makes the public record credible, enforced here rather than promised:

- `published_at` is always the real time this code runs - the caller cannot supply or
  backdate it.
- A pick is refused once its game has kicked off (or isn't `scheduled`), so a pick can never
  be entered after the outcome is known.
- One pick per game per market, and the ledger is append-only: a published pick can't be
  edited, replaced, or deleted (a mistaken duplicate can only be voided with the ledger's
  own restricted void reasons).

    python -m nfl_predict.decision.expert_picks --game 2026_03_DEN_KC --market spread \\
        --team DEN --line 3 --price -110 --note "Why I like it" [--dry-run]

`--line` is the picked TEAM's own number as quoted ("DEN +3" -> 3, "KC -3.5" -> -3.5); it is
stored in the ledger's convention (the HOME team's traditional-sign spread - see
`nfl_predict.decision.settlement`). Omit `--line`/`--price` to record the current consensus
market instead; the pick's `sportsbook_or_source` says which.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone

from nfl_predict.api import reconstruction as recon
from nfl_predict.decision.pick_ledger import (
    PickAlreadyExistsError,
    PickCategory,
    PublishedPick,
    publish_pick,
)
from nfl_predict.decision.settlement import SUPPORTED_MARKET_TYPES
from nfl_predict.live.schedule_provider import SCHEDULED_STATUS, get_current_season, get_schedule
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)

MAX_NOTE_LENGTH = 600


class ExpertPickError(ValueError):
    """A pick that must not be published - the message says exactly why."""


def _validate_price(price: int) -> int:
    if isinstance(price, bool) or not isinstance(price, int) or -100 < price < 100:
        raise ExpertPickError(f"price={price!r} is not valid American odds (e.g. -110 or +150).")
    return price


def _team_line_to_home_line(team_line: float, selection: str) -> float:
    return team_line if selection == "home" else -team_line


def build_expert_pick(
    game_id: str, team: str, market_type: str, line: float | None = None, price: int | None = None,
    note: str | None = None, now: str | None = None,
) -> PublishedPick:
    """Validates and builds the pick without writing anything - raises `ExpertPickError`
    for anything that must not be published."""
    if market_type not in SUPPORTED_MARKET_TYPES:
        raise ExpertPickError(f"market_type={market_type!r} not supported - only {SUPPORTED_MARKET_TYPES}.")
    if market_type == "moneyline" and line is not None:
        raise ExpertPickError("A moneyline pick has no line - omit --line.")
    if line is not None and not math.isfinite(line):
        raise ExpertPickError(f"line={line!r} is not a valid number.")
    if price is not None:
        _validate_price(price)
    note = (note or "").strip() or None
    if note is not None and len(note) > MAX_NOTE_LENGTH:
        raise ExpertPickError(f"note is {len(note)} characters - keep it under {MAX_NOTE_LENGTH}.")

    season = get_current_season()
    game = next((g for g in get_schedule(season).games if g.game_id == game_id), None)
    if game is None:
        raise ExpertPickError(f"game_id={game_id!r} is not in the {season} schedule.")

    now_dt = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
    if game.game_status != SCHEDULED_STATUS:
        raise ExpertPickError(f"{game_id} is {game.game_status!r}, not scheduled - picks can't be added after a game has been played.")
    if game.kickoff_timestamp is None:
        raise ExpertPickError(f"{game_id} has no known kickoff time - refusing to publish a pick that can't be shown to be pre-game.")
    if datetime.fromisoformat(game.kickoff_timestamp) <= now_dt:
        raise ExpertPickError(f"{game_id} kicked off at {game.kickoff_timestamp} - picks can't be added after kickoff.")

    abbr = team.strip().upper()
    if abbr == game.home_team_abbr.upper():
        selection = "home"
    elif abbr == game.away_team_abbr.upper():
        selection = "away"
    else:
        raise ExpertPickError(f"team={team!r} is not in {game_id} - use {game.away_team_abbr} or {game.home_team_abbr}.")

    market = recon.latest_market_point(game.season, game.week, game_id)
    market_available = market is not None and market.available

    used_market_default = price is None or (market_type == "spread" and line is None)
    if market_type == "spread":
        if line is None:
            if not market_available or market.home_spread_traditional is None:
                raise ExpertPickError("No current market spread to record - pass --line explicitly.")
            home_line = market.home_spread_traditional
        else:
            home_line = _team_line_to_home_line(line, selection)
        if price is None:
            default_price = None
            if market_available:
                default_price = market.home_spread_price if selection == "home" else market.away_spread_price
            if default_price is None:
                raise ExpertPickError("No current market spread price to record - pass --price explicitly.")
            price = default_price
        stored_line: float | None = home_line
    else:
        if price is None:
            default_price = None
            if market_available:
                default_price = market.home_moneyline if selection == "home" else market.away_moneyline
            if default_price is None:
                raise ExpertPickError("No current market moneyline to record - pass --price explicitly.")
            price = default_price
        stored_line = None

    if used_market_default:
        source = f"Consensus of {len(market.consensus_book_keys)} sportsbooks ({market.consensus_algorithm_version}) at publication - no line/price supplied"
    else:
        source = "Expert-supplied line and price"

    return PublishedPick(
        pick_id=f"{game_id}_expert_{market_type}",  # stable: one expert pick per game per market, ever
        game_id=game_id, published_at=now_dt.isoformat(), kickoff_at=game.kickoff_timestamp,
        decision_id="n/a - expert pick (not a decision-engine output)", rule_version="n/a",
        category=PickCategory.EXPERT_PICKS.value, market_type=market_type, selection=selection,
        line=stored_line, price=_validate_price(price), sportsbook_or_source=source,
        market_snapshot_id=market.market_snapshot_reference if market_available else None,
        model_prediction_snapshot={}, research_snapshot_id=None, validation_status="PROSPECTIVE", note=note,
    )


def publish_expert_pick(
    game_id: str, team: str, market_type: str, line: float | None = None, price: int | None = None,
    note: str | None = None, now: str | None = None,
) -> PublishedPick:
    pick = build_expert_pick(game_id, team, market_type, line=line, price=price, note=note, now=now)
    try:
        publish_pick(pick)
    except PickAlreadyExistsError as e:
        raise ExpertPickError(
            f"An expert {market_type} pick for {game_id} is already published - picks are immutable and can't be edited or replaced."
        ) from e
    logger.info("Published expert pick %s (%s %s)", pick.pick_id, pick.selection, pick.market_type)
    return pick


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nfl_predict.decision.expert_picks", description=__doc__.split("\n\n")[0])
    parser.add_argument("--game", required=True, dest="game_id")
    parser.add_argument("--market", required=True, choices=list(SUPPORTED_MARKET_TYPES))
    parser.add_argument("--team", required=True, help="Team abbreviation, e.g. DEN")
    parser.add_argument("--line", type=float, default=None, help="The picked team's own line, e.g. 3 for DEN +3 or -3.5 for KC -3.5")
    parser.add_argument("--price", type=int, default=None, help="American odds, e.g. -110")
    parser.add_argument("--note", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the pick without publishing it")
    args = parser.parse_args(argv)

    try:
        if args.dry_run:
            pick = build_expert_pick(args.game_id, args.team, args.market, args.line, args.price, args.note)
        else:
            pick = publish_expert_pick(args.game_id, args.team, args.market, args.line, args.price, args.note)
    except ExpertPickError as e:
        print(f"NOT PUBLISHED: {e}", file=sys.stderr)
        return 1

    summary = {k: getattr(pick, k) for k in ("pick_id", "game_id", "market_type", "selection", "line", "price", "sportsbook_or_source", "published_at", "kickoff_at", "note")}
    print(("DRY RUN - would publish:\n" if args.dry_run else "PUBLISHED:\n") + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
