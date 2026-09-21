"""Grades an expert parlay from real results: moneyline/spread legs from the real final
score (the same `decision.settlement` grading single picks use), player-prop legs from the
recorded nflverse `player_stats` snapshot.

Rules, chosen so the public record can never be flattered by a guess:

- A parlay LOSES the moment any leg loses (later legs need not be played to grade it).
- It WINS only when every leg is graded WIN.
- A leg that is not gradable yet (game not final, or the game's box score isn't in the
  ingested snapshot yet) keeps the parlay PENDING - a stale snapshot must never grade a leg.
- Anything a human has to decide is never guessed: a leg that PUSHES (the sportsbook
  re-prices the parlay without it, which this code can't know) or a player with no stat row
  in a game whose box score IS ingested (did he not play? books usually void that leg) puts
  the parlay in `review` - left unsettled and surfaced by `settle_all_pending_picks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import polars as pl

from nfl_predict.data.player_stats import (
    PROP_STATS,
    PlayerStatsUnavailableError,
    load_player_week_stats,
    player_stat_for_week,
    team_has_stats_for_week,
)
from nfl_predict.decision.settlement import settle_pick as grade_leg_from_score

WIN, LOSS, PUSH, PENDING, REVIEW, NOT_GRADED = "WIN", "LOSS", "PUSH", "PENDING", "REVIEW", "NOT_GRADED"


@dataclass
class ParlayOutcome:
    status: str  # "settled" | "pending" | "review"
    result: str | None = None  # "WIN" | "LOSS" when status == "settled"
    leg_results: list[dict] = field(default_factory=list)
    result_source: str = ""
    reason: str | None = None  # why it needs review


def _grade_leg(leg: dict, conn, stats_loader: Callable[[int], pl.DataFrame], stats_cache: dict) -> tuple[str, str]:
    game = conn.execute(
        "SELECT season, week, home_score, away_score, game_status, home_team_abbr, away_team_abbr "
        "FROM games WHERE game_id = ?", (leg["game_id"],),
    ).fetchone()
    if game is None or game["game_status"] != "final" or game["home_score"] is None or game["away_score"] is None:
        return PENDING, "game not final yet"

    score = f"{game['away_team_abbr']} {game['away_score']} - {game['home_team_abbr']} {game['home_score']}"
    home_score, away_score = game["home_score"], game["away_score"]

    if leg["type"] in ("moneyline", "spread"):
        result = grade_leg_from_score(
            market_type=leg["type"], selection=leg["selection"], line=leg["line"],
            actual_home_margin=home_score - away_score,
            actual_home_win=None if home_score == away_score else home_score > away_score,
        )
        return result.value, score

    season = game["season"]
    if season not in stats_cache:
        try:
            stats_cache[season] = stats_loader(season)
        except PlayerStatsUnavailableError:
            stats_cache[season] = None
    stats = stats_cache[season]
    if stats is None or not team_has_stats_for_week(stats, leg["team"], game["week"]):
        return PENDING, "box score not in the ingested player_stats snapshot yet - re-run the player_stats ingest"

    value = player_stat_for_week(stats, leg["player_id"], game["week"], leg["stat"])
    if value is None:
        return REVIEW, f"{leg['player']} has no stat row for this game - did he not play?"
    label = PROP_STATS[leg["stat"]]
    if value == leg["line"]:
        return PUSH, f"{value} {label}"
    won = value > leg["line"] if leg["direction"] == "over" else value < leg["line"]
    return (WIN if won else LOSS), f"{value} {label}"


def grade_parlay(legs: list[dict], conn, stats_loader: Callable[[int], pl.DataFrame] | None = None) -> ParlayOutcome:
    loader = stats_loader or load_player_week_stats  # looked up at call time, not frozen as a default
    stats_cache: dict = {}
    graded = [(leg, *_grade_leg(leg, conn, loader, stats_cache)) for leg in legs]
    results = {result for _, result, _ in graded}

    if LOSS in results:
        status, result = "settled", LOSS
    elif PENDING in results:
        status, result = "pending", None
    elif results & {REVIEW, PUSH}:
        status, result = "review", None
    else:
        status, result = "settled", WIN

    leg_results = [
        {"description": leg["description"], "result": (NOT_GRADED if status == "settled" and res == PENDING else res), "detail": detail}
        for leg, res, detail in graded
    ]
    reason = None
    if status == "review":
        reason = "; ".join(f"{lr['description']}: {lr['result']} ({lr['detail']})" for lr in leg_results if lr["result"] in (REVIEW, PUSH))
    source = "Parlay graded from real results: " + "; ".join(
        f"{i}. {lr['description']} - {lr['result']} ({lr['detail']})" for i, lr in enumerate(leg_results, 1)
    )
    return ParlayOutcome(status=status, result=result, leg_results=leg_results, result_source=source, reason=reason)
