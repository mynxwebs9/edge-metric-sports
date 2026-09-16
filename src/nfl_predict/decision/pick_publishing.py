"""Phase 10 follow-up: publishes `PickCategory.ALL_MODEL_PREDICTIONS` picks - a real,
settleable, priced moneyline pick for EVERY game with real model and market data, entirely
independent of whether that game's decision engine output ever reaches `QUALIFIED_BET`.

This is deliberately a SEPARATE record from `PickCategory.BEST_BETS` (never conflated on the
website or in the ledger): Best Bets are the deterministic decision engine's vetted,
never-forced output (model + market + research all had to agree) - Zero Best Bets in a week
is a valid, expected result. ALL_MODEL_PREDICTIONS is the opposite by design: the model's own
raw, unfiltered pick for every game, published unconditionally whenever real model/market
data exists, so a visitor can see "what would happen if you just followed the model every
week" as an honest baseline to compare the vetted Best Bets record against. Neither is a
substitute for the other; both are real, both settle against real final scores.

`PickCategory.ALL_MODEL_PREDICTIONS` already existed in `pick_ledger.py`'s enum but had never
actually been published to by any code path before this module (confirmed: `docs/DECISION_ENGINE.md`
and `api/main.py`'s `/api/nfl/performance` route already expect it - `all_model_predictions_record`
has always reported zeros only because the category was empty, not because of a bug).
"""

from __future__ import annotations

from datetime import datetime, timezone

from nfl_predict.api import reconstruction as recon
from nfl_predict.decision.pick_ledger import (
    PickAlreadyExistsError,
    PickCategory,
    PublishedPick,
    publish_pick,
)
from nfl_predict.live.schedule_provider import get_schedule
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)


def build_all_model_predictions_pick(game_id: str, season: int, week: int) -> PublishedPick | None:
    """Returns a real, priced moneyline `PublishedPick` for this game, or `None` if the real
    model prediction or market data this pick would need doesn't exist yet - never a
    fabricated pick. The selection is whichever side the model's own `predicted_winner`
    favors (the same field already shown on the website's model-prediction block); the price
    is the real market's moneyline for that exact side, from the current market snapshot."""
    model = recon.latest_model_prediction(game_id)
    if model is None or model.get("predicted_winner") not in ("home", "away"):
        return None

    market = recon.latest_market_point(season, week, game_id)
    if market is None or not market.available or market.home_moneyline is None or market.away_moneyline is None:
        return None

    selection = model["predicted_winner"]
    price = market.home_moneyline if selection == "home" else market.away_moneyline

    decisions = recon.latest_decisions_for_game(season, week, game_id)
    moneyline_decision = decisions.get("moneyline")
    decision_id = moneyline_decision["decision_id"] if moneyline_decision else f"{game_id}_moneyline_no_decision_record"
    rule_version = moneyline_decision["decision_rule_version"] if moneyline_decision else "n/a"
    research_id = moneyline_decision.get("research_id") if moneyline_decision else None

    now_iso = datetime.now(timezone.utc).isoformat()
    return PublishedPick(
        pick_id=f"{game_id}_all_model_predictions",  # stable, not timestamped - one per game, ever; re-running is a safe no-op via PickAlreadyExistsError
        game_id=game_id,
        published_at=now_iso,
        kickoff_at=model.get("kickoff_timestamp"),
        decision_id=decision_id,
        rule_version=rule_version,
        category=PickCategory.ALL_MODEL_PREDICTIONS.value,
        market_type="moneyline",
        selection=selection,
        line=None,
        price=price,
        sportsbook_or_source=f"Consensus of {len(market.consensus_book_keys)} sportsbooks ({market.consensus_algorithm_version}): " + ", ".join(market.consensus_book_keys),
        market_snapshot_id=market.market_snapshot_reference,
        model_prediction_snapshot=model,
        research_snapshot_id=research_id,
        validation_status="PROSPECTIVE",
    )


def publish_all_model_predictions_for_week(season: int, week: int) -> dict:
    """Publishes one ALL_MODEL_PREDICTIONS pick per game in this week's real schedule that
    has real model + market data and isn't already published (idempotent - safe to re-run).
    Returns real counts, never fabricated: `published`, `already_published`, `no_data_yet`."""
    schedule = get_schedule(season, week)
    published, already_published, no_data_yet = [], [], []

    for game in schedule.games:
        pick = build_all_model_predictions_pick(game.game_id, season, week)
        if pick is None:
            no_data_yet.append(game.game_id)
            continue
        try:
            publish_pick(pick)
            published.append(game.game_id)
        except PickAlreadyExistsError:
            already_published.append(game.game_id)

    logger.info(
        "ALL_MODEL_PREDICTIONS publish for season=%s week=%s: %d published, %d already published, %d no data yet",
        season, week, len(published), len(already_published), len(no_data_yet),
    )
    return {
        "published": published,
        "already_published": already_published,
        "no_data_yet": no_data_yet,
    }
