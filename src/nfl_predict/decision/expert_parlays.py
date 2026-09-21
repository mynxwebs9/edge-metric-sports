"""Expert parlays: a human's own multi-leg parlay, published into the same immutable pick
ledger (`PickCategory.EXPERT_PARLAYS`) as its own record - never blended into the expert's
single picks or the model's records.

A parlay wins only if EVERY leg wins. Legs may be a moneyline, a spread, or a player prop
(a countable box-score stat over/under a line); each leg's game must still be pre-kickoff
(the exact rule `expert_picks.require_pregame_game` enforces for single picks), the legs are
frozen into the ledger at publish time, and the parlay's own American price - what the
sportsbook actually offered, which for a same-game parlay is NOT simply the legs multiplied
together - is what the record's units use. Grading lives in `parlay_settlement.py`.

    python -m nfl_predict.decision.expert_parlays --price 180 --note "Why" [--dry-run] --legs '[
      {"type": "moneyline", "game_id": "2026_02_NYG_LA", "team": "LA", "price": -305},
      {"type": "player_prop", "game_id": "2026_02_NYG_LA", "player": "Matthew Stafford",
       "stat": "passing_yards", "at_least": 210, "price": -233},
      {"type": "spread", "game_id": "2026_03_LA_DEN", "team": "LA", "line": -1.5, "price": -110}]'

`at_least: 210` ("210+") is stored as `over 209.5`. A spread `line` is the picked TEAM's own
number, exactly as for `expert_picks`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone

from nfl_predict.data.player_stats import (
    PROP_STATS,
    PlayerNotFoundError,
    PlayerStatsUnavailableError,
    load_player_week_stats,
    resolve_player,
)
from nfl_predict.decision.expert_picks import (
    MAX_NOTE_LENGTH,
    ExpertPickError,
    _validate_price,
    require_pregame_game,
    selection_for_team,
    team_line_to_home_line,
)
from nfl_predict.decision.pick_ledger import PickAlreadyExistsError, PickCategory, PublishedPick, publish_pick
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)

LEG_TYPES = ("moneyline", "spread", "player_prop")
MIN_LEGS, MAX_LEGS = 2, 12


def _format_signed(value: float) -> str:
    return f"{value:+g}"


def _build_leg(spec: dict, now_dt: datetime, stats_cache: dict) -> tuple[dict, str]:
    """Returns (frozen leg dict, this leg's kickoff timestamp)."""
    leg_type = spec.get("type")
    if leg_type not in LEG_TYPES:
        raise ExpertPickError(f"leg type={leg_type!r} not supported - only {LEG_TYPES}.")
    if not spec.get("game_id"):
        raise ExpertPickError(f"every leg needs a game_id: {spec}")
    price = spec.get("price")
    if price is not None:
        _validate_price(price)

    game = require_pregame_game(spec["game_id"], now_dt)
    leg = {
        "type": leg_type, "game_id": game.game_id, "matchup": f"{game.away_team_abbr} @ {game.home_team_abbr}",
        "price": price,
    }

    if leg_type == "moneyline":
        leg["selection"] = selection_for_team(game, spec.get("team", ""))
        leg["team"] = (game.home_team_abbr if leg["selection"] == "home" else game.away_team_abbr).upper()
        leg["line"] = None
        leg["description"] = f"{leg['team']} moneyline"

    elif leg_type == "spread":
        if spec.get("line") is None or not math.isfinite(spec["line"]):
            raise ExpertPickError(f"a spread leg needs the picked team's own numeric line: {spec}")
        leg["selection"] = selection_for_team(game, spec.get("team", ""))
        leg["team"] = (game.home_team_abbr if leg["selection"] == "home" else game.away_team_abbr).upper()
        leg["line"] = team_line_to_home_line(float(spec["line"]), leg["selection"])  # ledger convention: HOME spread
        leg["description"] = f"{leg['team']} {_format_signed(float(spec['line']))}"

    else:  # player_prop
        stat = spec.get("stat")
        if stat not in PROP_STATS:
            raise ExpertPickError(f"stat={stat!r} not supported - only {sorted(PROP_STATS)}.")
        if game.season_type != "REG":
            raise ExpertPickError("player-prop legs are only supported for regular-season games.")
        if (spec.get("at_least") is None) == (spec.get("line") is None):
            raise ExpertPickError(f"a player-prop leg needs exactly one of 'at_least' (\"210+\") or 'line' (+ 'direction'): {spec}")
        if spec.get("at_least") is not None:
            direction, line = "over", float(spec["at_least"]) - 0.5
        else:
            direction, line = spec.get("direction"), float(spec["line"])
            if direction not in ("over", "under"):
                raise ExpertPickError(f"direction={direction!r} must be 'over' or 'under'.")
        if not math.isfinite(line) or line < 0:
            raise ExpertPickError(f"prop line={line!r} is not a valid number.")

        if game.season not in stats_cache:
            try:
                stats_cache[game.season] = load_player_week_stats(game.season)
            except PlayerStatsUnavailableError as e:
                raise ExpertPickError(str(e)) from e
        try:
            player = resolve_player(stats_cache[game.season], spec.get("player", ""), (game.home_team_abbr, game.away_team_abbr))
        except PlayerNotFoundError as e:
            raise ExpertPickError(str(e)) from e

        label = PROP_STATS[stat]
        leg.update({
            "player_id": player["player_id"], "player": player["player_display_name"], "team": player["team"],
            "stat": stat, "direction": direction, "line": line,
        })
        if direction == "over" and line % 1 == 0.5:
            leg["description"] = f"{player['player_display_name']} {math.ceil(line)}+ {label}"
        else:
            leg["description"] = f"{player['player_display_name']} {direction} {line:g} {label}"

    return leg, game.kickoff_timestamp


def _leg_identity(leg: dict) -> tuple:
    if leg["type"] == "player_prop":
        return (leg["type"], leg["game_id"], leg["player_id"], leg["stat"], leg["direction"], leg["line"])
    return (leg["type"], leg["game_id"], leg["selection"], leg["line"])


def build_expert_parlay(legs: list[dict], price: int, note: str | None = None, now: str | None = None) -> PublishedPick:
    """Validates and builds the parlay without writing anything."""
    if not MIN_LEGS <= len(legs) <= MAX_LEGS:
        raise ExpertPickError(f"a parlay needs {MIN_LEGS}-{MAX_LEGS} legs, got {len(legs)}.")
    _validate_price(price)
    note = (note or "").strip() or None
    if note is not None and len(note) > MAX_NOTE_LENGTH:
        raise ExpertPickError(f"note is {len(note)} characters - keep it under {MAX_NOTE_LENGTH}.")

    now_dt = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
    stats_cache: dict = {}
    built = [_build_leg(spec, now_dt, stats_cache) for spec in legs]
    frozen_legs = [leg for leg, _ in built]

    identities = [_leg_identity(leg) for leg in frozen_legs]
    if len(set(identities)) != len(identities):
        raise ExpertPickError("the same leg appears more than once in this parlay.")
    ml_games = [leg["game_id"] for leg in frozen_legs if leg["type"] == "moneyline"]
    if len(set(ml_games)) != len(ml_games):
        raise ExpertPickError("a parlay can't include a moneyline on both sides of the same game.")

    first_leg, first_kickoff = min(built, key=lambda b: b[1])
    digest = hashlib.sha1(json.dumps(sorted(identities, key=repr), default=str).encode("utf-8")).hexdigest()[:10]
    return PublishedPick(
        pick_id=f"expert_parlay_{digest}",  # stable per leg-set: re-publishing the same parlay is refused, not duplicated
        game_id=first_leg["game_id"], published_at=now_dt.isoformat(), kickoff_at=first_kickoff,
        decision_id="n/a - expert parlay (not a decision-engine output)", rule_version="n/a",
        category=PickCategory.EXPERT_PARLAYS.value, market_type="parlay", selection="parlay",
        line=None, price=price, sportsbook_or_source="Expert-supplied parlay price",
        market_snapshot_id=None, model_prediction_snapshot={}, research_snapshot_id=None,
        validation_status="PROSPECTIVE", note=note, legs=frozen_legs,
    )


def publish_expert_parlay(legs: list[dict], price: int, note: str | None = None, now: str | None = None) -> PublishedPick:
    pick = build_expert_parlay(legs, price, note=note, now=now)
    try:
        publish_pick(pick)
    except PickAlreadyExistsError as e:
        raise ExpertPickError("This exact parlay is already published - picks are immutable and can't be edited or replaced.") from e
    logger.info("Published expert parlay %s (%d legs at %+d)", pick.pick_id, len(pick.legs), pick.price)
    return pick


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nfl_predict.decision.expert_parlays", description=__doc__.split("\n\n")[0])
    parser.add_argument("--legs", required=True, help="JSON array of legs, or @path/to/legs.json")
    parser.add_argument("--price", required=True, type=int, help="The parlay's own American odds, e.g. 180")
    parser.add_argument("--note", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the parlay without publishing it")
    args = parser.parse_args(argv)

    raw = open(args.legs[1:], encoding="utf-8").read() if args.legs.startswith("@") else args.legs
    try:
        specs = json.loads(raw)
        pick = (build_expert_parlay if args.dry_run else publish_expert_parlay)(specs, args.price, args.note)
    except (json.JSONDecodeError, ExpertPickError) as e:
        print(f"NOT PUBLISHED: {e}", file=sys.stderr)
        return 1

    summary = {"pick_id": pick.pick_id, "price": pick.price, "published_at": pick.published_at, "kickoff_at": pick.kickoff_at,
               "note": pick.note, "legs": [{k: leg[k] for k in ("description", "matchup", "price")} for leg in pick.legs]}
    print(("DRY RUN - would publish:\n" if args.dry_run else "PUBLISHED:\n") + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
