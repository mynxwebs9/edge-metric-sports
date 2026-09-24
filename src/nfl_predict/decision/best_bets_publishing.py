"""Publishes the week's official Best Bets from the decision engine's own QUALIFIED_BET
decisions - the repeatable command for what used to be a one-off script.

    python -m nfl_predict.decision.best_bets_publishing --week 3 [--dry-run]

**The side of a Best Bet is the VALUE side, not the model's predicted winner.** The decision
engine qualifies a market because the model and the market disagree; the SIGN of that same
disagreement (`decision.engine._compute_disagreement`, reused here, never re-derived) says
who the model likes more than the market does. When the model rates the favorite lower than
the market does, the value is on the underdog - betting the model's "predicted winner" there
would bet the side the model itself thinks is overpriced. (`ALL_MODEL_PREDICTIONS` is the
opposite by design: the model's own straight-up pick. See `pick_publishing.py`.)

- One Best Bet per game, ever. If both spread and moneyline qualified, the one whose
  disagreement is the larger multiple of its own configured threshold wins (moneyline on an
  exact tie); the other is never published.
- The pick's line, price and market snapshot are the exact market the qualifying decision
  saw (`recon.market_point_for_decision_record`), so it reproduces from that decision -
  publish right after the decisions run, while that market is still current.
- Never after kickoff: the same pre-game rule expert picks follow.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from nfl_predict.api import reconstruction as recon
from nfl_predict.decision.engine import _compute_disagreement
from nfl_predict.decision.expert_picks import ExpertPickError, require_pregame_game
from nfl_predict.decision.pick_ledger import (
    PickAlreadyExistsError,
    PickCategory,
    PublishedPick,
    publish_pick,
    read_current_picks,
)
from nfl_predict.decision.rules_config import load_rule_set
from nfl_predict.live.schedule_provider import get_current_season, get_schedule
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)

THRESHOLD_RULE_IDS = {"spread": "min_spread_disagreement_points", "moneyline": "min_moneyline_disagreement_probability"}


def _candidate_for_market(market_type: str, decision: dict | None, model: dict | None, rule_set) -> dict | None:
    """The fully-resolved Best Bet for one qualified market, or None if it can't be built."""
    if decision is None or decision["decision"] != "QUALIFIED_BET" or model is None:
        return None
    market = recon.market_point_for_decision_record(decision)
    if market is None or not market.available:
        return None

    packet = SimpleNamespace(
        market=market,
        elo=SimpleNamespace(available=True, predicted_margin=model.get("elo_predicted_margin"), home_win_probability=model.get("elo_home_win_probability")),
    )
    disagreement = _compute_disagreement(packet, market_type)
    if disagreement is None or disagreement == 0:
        return None

    selection = "home" if disagreement > 0 else "away"
    if market_type == "spread":
        price = market.home_spread_price if selection == "home" else market.away_spread_price
        line = market.home_spread_traditional  # ledger convention: the HOME team's spread
    else:
        price = market.home_moneyline if selection == "home" else market.away_moneyline
        line = None
    if price is None:
        return None

    threshold = rule_set.get(THRESHOLD_RULE_IDS[market_type]).threshold
    return {
        "market_type": market_type, "selection": selection, "price": price, "line": line, "market": market,
        "decision": decision, "disagreement": disagreement, "ratio": abs(disagreement) / threshold,
    }


def choose_best_bet(candidates: list[dict]) -> dict | None:
    """The larger multiple of its own threshold wins; moneyline wins an exact tie."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c["ratio"], c["market_type"] == "moneyline"))


def build_best_bet(game_id: str, season: int, week: int, kickoff_at: str | None, now_iso: str) -> tuple[PublishedPick, dict] | None:
    model = recon.latest_model_prediction(game_id)
    decisions = recon.latest_decisions_for_game(season, week, game_id)
    rule_set = load_rule_set()
    candidates = [c for mt in ("spread", "moneyline") if (c := _candidate_for_market(mt, decisions[mt], model, rule_set))]
    chosen = choose_best_bet(candidates)
    if chosen is None:
        return None

    decision, market = chosen["decision"], chosen["market"]
    pick = PublishedPick(
        pick_id=f"{game_id}_best_bet",  # one Best Bet per game, ever - re-running is a safe no-op
        game_id=game_id, published_at=now_iso, kickoff_at=kickoff_at,
        decision_id=decision["decision_id"], rule_version=decision["decision_rule_version"],
        category=PickCategory.BEST_BETS.value, market_type=chosen["market_type"], selection=chosen["selection"],
        line=chosen["line"], price=chosen["price"],
        sportsbook_or_source=f"Consensus of {len(market.consensus_book_keys)} sportsbooks ({market.consensus_algorithm_version}): " + ", ".join(market.consensus_book_keys),
        market_snapshot_id=market.market_snapshot_reference, model_prediction_snapshot=model,
        research_snapshot_id=decision.get("research_id"), validation_status="PROSPECTIVE",
    )
    return pick, {"disagreement": chosen["disagreement"], "ratio": round(chosen["ratio"], 2), "qualified_markets": [c["market_type"] for c in candidates]}


def publish_best_bets_for_week(week: int, dry_run: bool = False, now: str | None = None) -> dict:
    """Returns real lists, never fabricated: `published` (what was, or with `dry_run` would be,
    published), `already_published`, `too_late` (kickoff passed), and `not_qualified` (a count)."""
    season = get_current_season()
    now_iso = now or datetime.now(timezone.utc).isoformat()
    now_dt = datetime.fromisoformat(now_iso)
    already = {p["game_id"] for p in read_current_picks() if p["category"] == PickCategory.BEST_BETS.value and p["status"] != "VOID"}

    published, already_published, too_late, not_qualified = [], [], [], 0
    for game in get_schedule(season, week).games:
        if game.game_id in already:
            already_published.append(game.game_id)
            continue
        built = build_best_bet(game.game_id, season, game.week, game.kickoff_timestamp, now_iso)
        if built is None:
            not_qualified += 1
            continue
        pick, detail = built
        try:
            require_pregame_game(game.game_id, now_dt)
        except ExpertPickError as e:
            too_late.append({"game_id": game.game_id, "reason": str(e)})
            continue
        if not dry_run:
            try:
                publish_pick(pick)
            except PickAlreadyExistsError:
                already_published.append(game.game_id)
                continue
        published.append({"game_id": game.game_id, "market_type": pick.market_type, "selection": pick.selection, "line": pick.line, "price": pick.price, **detail})

    logger.info("Best Bets week %s%s: %d published, %d already, %d too late, %d not qualified",
                week, " (dry run)" if dry_run else "", len(published), len(already_published), len(too_late), not_qualified)
    return {"published": published, "already_published": already_published, "too_late": too_late, "not_qualified": not_qualified}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nfl_predict.decision.best_bets_publishing", description=__doc__.split("\n\n")[0])
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be published without publishing anything")
    args = parser.parse_args(argv)
    print(json.dumps(publish_best_bets_for_week(args.week, dry_run=args.dry_run), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
