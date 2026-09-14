"""Phase 8A Step 2 (corrected): the live orchestration command.

    python -m nfl_predict.live.run [--week N] [--game GAME_ID] [--dry-run]
        [--research-only | --odds-only | --models-only | --decisions-only]
        [--research-all] [--max-research-calls N] [--show-decisions]
        [--publish]

Default behavior is safe with respect to PUBLISHING: no mode flag runs the full pipeline
(schedule -> live models -> odds -> research -> injury -> decisions); `--publish` is
required to even ATTEMPT publishing an official Best Bet, and even then only a genuine
QUALIFIED_BET decision, computed from real data, in a non-dry-run, gets published.
`--dry-run` (the default) writes no pregame-run record, no decision-log entry, and publishes
nothing.

**`--dry-run` does NOT mean "no external calls."** Whenever `NFL_ODDS_API_KEY` is
configured, every run (dry or not) fetches real odds from The Odds API and persists them to
the append-only snapshot store. Whenever `NFL_RESEARCH_LLM_API_KEY` is configured, every run
(dry or not) may invoke Anthropic's Messages API - a REAL, BILLED call - for up to
`--max-research-calls` games (default from `NFL_RESEARCH_MAX_GAMES_PER_RUN`, 3 if unset), or
every target game if `--research-all` is passed. Each such call is itself bounded by
`NFL_RESEARCH_MAX_WEB_SEARCHES_PER_GAME` (default 3) and, pre-flight, by
`NFL_RESEARCH_MAX_ESTIMATED_COST_PER_GAME` (default $1.00 - a call whose known cost floor
already exceeds this is refused before any HTTP request, see
`nfl_predict.research.cost_tracking`). This is intentional (research/odds artifacts are
real, valuable, immutable observations worth keeping regardless of whether decisions get
published this run) but is called out explicitly here because it has a real dollar cost -
see docs/PHASE8A_LIVE_PIPELINE_REPORT.md for the estimated cost per call.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

from nfl_predict.logging_conf import get_logger

from nfl_predict.decision.decision_log import append_decision_record
from nfl_predict.decision.engine import decide
from nfl_predict.decision.model_agreement import compute_model_agreement
from nfl_predict.decision.rules_config import load_rule_set
from nfl_predict.decision.schemas import (
    DecisionRecord,
    MarketPoint,
    ModelPoint,
    ResearchPoint,
    SystemHealth,
)
from nfl_predict.decision.staleness import compute_age_seconds
from nfl_predict.live.component_status import ComponentState, RunHealthReport, run_component_safely
from nfl_predict.live.injury_automation import InjuryProviderStatus, get_production_injury_provider
from nfl_predict.live.live_models import predict_live_elo, predict_live_ridge_lightgbm_logistic
from nfl_predict.live.odds_automation import OddsProviderStatus, get_production_odds_provider
from nfl_predict.live.odds_ingestion import fetch_and_snapshot_live_odds
from nfl_predict.live.research_automation import ResearchProviderStatus, get_production_research_provider
from nfl_predict.live.research_live import compute_trigger_priorities, run_live_research_for_game, select_games_for_research
from nfl_predict.live.schedule_provider import get_current_season, get_schedule, upcoming_games
from nfl_predict.config import get_settings
from nfl_predict.research.cost_tracking import ResearchCostConfig

logger = get_logger(__name__)

ALL_SEASONS_FOR_TRAINING = list(range(2010, 2027))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_model_points(game_id: str, elo_records, other_results) -> tuple[ModelPoint, ModelPoint, ModelPoint, ModelPoint]:
    elo_margin = next((r.predicted_value for r in elo_records if r.game_id == game_id and r.target == "home_margin"), None)
    elo_prob = next((r.predicted_value for r in elo_records if r.game_id == game_id and r.target == "home_win"), None)
    elo = ModelPoint(model_id="elo_v2", available=elo_margin is not None, predicted_margin=elo_margin, home_win_probability=elo_prob)

    def _from_records(records, target):
        if records is None:
            return None
        return next((r.predicted_value for r in records if r.game_id == game_id and r.target == target), None)

    ridge_margin = _from_records(other_results.get("ridge_margin_E_v1"), "home_margin")
    ridge = ModelPoint(model_id="ridge_margin_E_v1", available=ridge_margin is not None, predicted_margin=ridge_margin)

    lgbm_margin = _from_records(other_results.get("lightgbm_F_v1"), "home_margin")
    lgbm_prob = _from_records(other_results.get("lightgbm_F_v1"), "home_win")
    lightgbm = ModelPoint(model_id="lightgbm_F_v1", available=lgbm_margin is not None, predicted_margin=lgbm_margin, home_win_probability=lgbm_prob)

    logistic_prob = _from_records(other_results.get("logistic_win_E_v1"), "home_win")
    logistic = ModelPoint(model_id="logistic_win_E_v1", available=logistic_prob is not None, home_win_probability=logistic_prob)

    return elo, ridge, lightgbm, logistic


def _filesystem_safe_run_id(timestamp_iso: str) -> str:
    """Same sanitization as `prediction_publication._filesystem_safe` - a raw ISO timestamp
    is used as a `run_id={...}` directory component by `research.storage`, and Windows
    rejects colons in path segments."""
    return timestamp_iso.replace(":", "-")


def _trigger_spread_disagreement(elo: ModelPoint, market: MarketPoint) -> float | None:
    """A simple, LOCAL disagreement figure used only to rank research priority (Step 4) -
    deliberately not a reuse of `decision.engine._compute_disagreement`, so the trigger
    module never depends on the decision engine's internals. Never gates anything - the
    trigger score is a ranking signal only (`research.trigger`'s own documented stance)."""
    if not (elo.available and elo.predicted_margin is not None and market.available and market.home_spread_traditional is not None):
        return None
    return elo.predicted_margin - (-market.home_spread_traditional)


def run_slate(
    week: int | None = None, game_id: str | None = None, dry_run: bool = True, publish: bool = False,
    modes: set[str] | None = None, research_all: bool = False, max_research_calls: int | None = None,
    show_decisions: bool = False,
) -> dict:
    modes = modes or {"schedule", "models", "odds", "research", "injury", "decisions"}
    max_research_calls = max_research_calls if max_research_calls is not None else get_settings().research_max_games_per_run
    now = _now_iso()
    health = RunHealthReport()

    season, ok = run_component_safely(health, "schedule", now, get_current_season)
    if not ok:
        return {"status": "FAILED", "health": health.as_dict()}

    schedule, ok = run_component_safely(health, "schedule", now, get_schedule, season, week)
    if not ok:
        return {"status": "FAILED", "health": health.as_dict()}

    if game_id is not None:
        # An explicit --game always wins over week filtering, wherever it falls in the season.
        target_games = [g for g in upcoming_games(schedule) if g.game_id == game_id]
    else:
        effective_week = week if week is not None else schedule.current_week
        target_games = [g for g in upcoming_games(schedule) if effective_week is None or g.week == effective_week]
    game_ids = [g.game_id for g in target_games]
    logger.info("Slate: season=%s week=%s games=%d", season, schedule.current_week, len(game_ids))

    elo_records, ridge_lgbm_logistic = [], {}
    if "models" in modes and game_ids:
        elo_records, ok = run_component_safely(health, "elo_model", now, predict_live_elo, game_ids, ALL_SEASONS_FOR_TRAINING)
        elo_records = elo_records or []
        ridge_lgbm_logistic, ok2 = run_component_safely(health, "ridge_lightgbm_logistic_models", now, predict_live_ridge_lightgbm_logistic, game_ids, ALL_SEASONS_FOR_TRAINING)
        ridge_lgbm_logistic = ridge_lgbm_logistic or {}

    # --- Live odds: real per-game MarketPoints, never available=True merely because the
    # provider/key could be constructed (odds_ingestion.py builds these from real fetched
    # markets, persisted through the existing append-only snapshot store first). ---
    odds_status = OddsProviderStatus.ODDS_PROVIDER_UNAVAILABLE
    odds_ingestion = None
    if "odds" in modes:
        odds_result, _ = run_component_safely(health, "odds_provider", now, get_production_odds_provider)
        odds_status = odds_result.status if odds_result else odds_status
        health.record("odds_provider", ComponentState.SUCCESS if odds_status == OddsProviderStatus.AVAILABLE else ComponentState.SKIPPED, odds_status.value, now)
        if odds_status == OddsProviderStatus.AVAILABLE and game_ids:
            odds_ingestion, _ = run_component_safely(health, "odds_ingestion", now, fetch_and_snapshot_live_odds, odds_result.provider, season, game_ids)

    market_by_game: dict[str, MarketPoint] = {}
    for game in target_games:
        matched = odds_ingestion.games.get(game.game_id) if odds_ingestion else None
        market_by_game[game.game_id] = matched.market if matched else MarketPoint(available=False, source=f"odds_provider status={odds_status.value}")

    research_status = ResearchProviderStatus.RESEARCH_PROVIDER_UNAVAILABLE
    research_provider = None
    if "research" in modes:
        research_result, _ = run_component_safely(health, "research_provider", now, get_production_research_provider)
        research_status = research_result.status if research_result else research_status
        research_provider = research_result.provider if research_result else None
        health.record("research_provider", ComponentState.SUCCESS if research_status == ResearchProviderStatus.AVAILABLE else ComponentState.SKIPPED, research_status.value, now)

    injury_status = InjuryProviderStatus.INJURY_PROVIDER_UNAVAILABLE
    if "injury" in modes:
        # Structured injury data is a boundary distinct from the LLM research provider above
        # (see injury_automation module docstring) - checked and reported separately, never
        # merged into research_status/market.available, and never itself a decision-rule
        # input (rule set v1 has no injury-specific required_input) - so its unavailability
        # never causes MISSING_LIVE_DATA on its own.
        injury_result, _ = run_component_safely(health, "injury_provider", now, get_production_injury_provider)
        injury_status = injury_result.status if injury_result else injury_status
        health.record("injury_provider", ComponentState.SUCCESS if injury_status == InjuryProviderStatus.AVAILABLE else ComponentState.SKIPPED, injury_status.value, now)

    model_points_by_game = {game.game_id: _build_model_points(game.game_id, elo_records, ridge_lgbm_logistic) for game in target_games}

    # --- Research: actually invoke the provider for triggered games (Step 3-4). Trigger
    # priority only decides ORDER + which games fit under `max_research_calls` - it never
    # decides eligibility (every game is ALWAYS_RESEARCHABLE per research.trigger). ---
    disagreement_by_game, dispersion_by_game = {}, {}
    for game in target_games:
        elo, ridge, lightgbm, _ = model_points_by_game[game.game_id]
        disagreement_by_game[game.game_id] = _trigger_spread_disagreement(elo, market_by_game[game.game_id])
        dispersion_by_game[game.game_id] = compute_model_agreement(elo, ridge, lightgbm).margin_dispersion
    priorities = compute_trigger_priorities(disagreement_by_game, dispersion_by_game)

    research_point_by_game: dict[str, ResearchPoint] = {g.game_id: ResearchPoint(available=False) for g in target_games}
    research_outcomes = []
    selected_game_ids: set[str] = set()
    if "research" in modes and research_status == ResearchProviderStatus.AVAILABLE and game_ids:
        selected_game_ids = set(select_games_for_research(priorities, research_all=research_all, max_calls=max_research_calls))
        run_id = _filesystem_safe_run_id(now)
        for game in target_games:
            if game.game_id not in selected_game_ids:
                continue
            outcome, _ = run_component_safely(
                health, f"research:{game.game_id}", now, run_live_research_for_game,
                game.game_id, game.season, game.week, research_provider, market_by_game[game.game_id],
                priorities.get(game.game_id), run_id, now, ResearchCostConfig(),
            )
            if outcome is not None:
                research_outcomes.append(outcome)
                research_point_by_game[game.game_id] = outcome.research_point
                if outcome.status == "cost_capped":
                    # A real, pre-flight cost-budget refusal (Phase 8A cost-controls
                    # correction) - fail safely by stopping further research THIS RUN
                    # rather than repeating the same refusal (or worse, a barely-under-cap
                    # call) for every remaining selected game.
                    logger.warning("Research cost budget exceeded on %s - stopping further research this run.", game.game_id)
                    break

    rule_set = load_rule_set()
    decisions = {}
    n_published_predictions = 0
    if "decisions" in modes:
        for game in target_games:
            elo, ridge, lightgbm, logistic = model_points_by_game[game.game_id]
            agreement = compute_model_agreement(elo, ridge, lightgbm)
            market = market_by_game[game.game_id]
            research = research_point_by_game[game.game_id]

            if not dry_run:
                from nfl_predict.live.prediction_publication import PublicationState, PublicModelPrediction, publish_model_prediction

                predicted_winner = None
                if elo.predicted_margin is not None:
                    predicted_winner = "home" if elo.predicted_margin > 0 else ("away" if elo.predicted_margin < 0 else None)
                public_pred = PublicModelPrediction(
                    prediction_id=now, game_id=game.game_id, season=season, week=game.week, generated_at=now,
                    elo_home_win_probability=elo.home_win_probability, elo_predicted_margin=elo.predicted_margin,
                    ridge_predicted_margin=ridge.predicted_margin, lightgbm_predicted_margin=lightgbm.predicted_margin,
                    model_agreement_all_agree=agreement.all_agree_on_direction, model_agreement_dispersion=agreement.margin_dispersion,
                    predicted_winner=predicted_winner,
                )
                _, published_ok = run_component_safely(health, f"publish_prediction:{game.game_id}", now, publish_model_prediction, public_pred, PublicationState.PUBLISHED)
                if published_ok:
                    n_published_predictions += 1

            market_age = compute_age_seconds(market.snapshot_timestamp, now) if market.available else None
            research_age = compute_age_seconds(research.research_timestamp, now) if research.available else None
            packet_health = SystemHealth(missing_required_data=(), stale_flags=(), market_age_seconds=market_age, research_age_seconds=research_age)
            from nfl_predict.decision.schemas import DecisionInputPacket

            packet = DecisionInputPacket(
                game_id=game.game_id, decision_timestamp=now, kickoff_timestamp=game.kickoff_timestamp,
                elo=elo, ridge=ridge, lightgbm=lightgbm, model_agreement=agreement,
                market=market, research=research, system_health=packet_health,
            )
            for market_type in ("spread", "moneyline"):
                decision, reasons = decide(packet, rule_set, market_type)
                record = DecisionRecord(
                    decision_id=f"{game.game_id}_{market_type}_{now}", game_id=game.game_id, decision_timestamp=now,
                    kickoff_timestamp=game.kickoff_timestamp, decision=decision, reason_codes=reasons,
                    decision_rule_version=rule_set.rule_set_version, market_type=market_type,
                    input_packet_hash=packet.content_hash(), validation_status="PROSPECTIVE",
                    market_provider_event_id=market.provider_event_id,
                    market_snapshot_reference=market.market_snapshot_reference,
                    market_snapshot_timestamp=market.snapshot_timestamp,
                    research_id=research.research_id,
                )
                if not dry_run:
                    append_decision_record(record, season, schedule.current_week or game.week)
                decisions[(game.game_id, market_type)] = {
                    "decision": decision.value, "reasons": [r.value for r in reasons],
                    "elo_margin": elo.predicted_margin, "market_home_spread": market.home_spread_traditional,
                    "market_no_vig_home_prob": market.no_vig_home_win_probability,
                    "research_classification": research.classification,
                }

    summary = _build_automation_report(
        season, schedule, target_games, elo_records, ridge_lgbm_logistic, odds_status, research_status, injury_status,
        decisions, health, dry_run, publish, n_published_predictions, odds_ingestion, research_outcomes, selected_game_ids,
    )
    if show_decisions:
        summary["decisions"] = [
            {"game_id": g, "market_type": m, **v} for (g, m), v in sorted(decisions.items())
        ]
    return summary


def _build_automation_report(
    season, schedule, target_games, elo_records, other_results, odds_status, research_status, injury_status,
    decisions, health, dry_run, publish, n_published_predictions=0, odds_ingestion=None, research_outcomes=None,
    selected_game_ids=None,
) -> dict:
    research_outcomes = research_outcomes or []
    selected_game_ids = selected_game_ids or set()
    n_games = len(target_games)
    n_elo = len({r.game_id for r in elo_records}) if elo_records else 0
    n_ridge = len({r.game_id for r in (other_results.get("ridge_margin_E_v1") or [])})
    n_lgbm = len({r.game_id for r in (other_results.get("lightgbm_F_v1") or [])})

    decision_counts: Counter = Counter(v["decision"] for v in decisions.values())
    reason_code_counts: Counter = Counter()
    for v in decisions.values():
        reason_code_counts.update(v["reasons"])
    # Gate 2 (missing_market_or_research_blocks_qualified_bet) is the ONLY gate a decision
    # whose reasons are exactly ("MISSING_LIVE_DATA",) could have failed at - anything else
    # in its reasons means it reached staleness/disagreement/coverage/dispersion/research
    # gates (3-9) for real, per docs/PHASE8A_LIVE_PIPELINE_REPORT.md's gate table.
    any_decision_reached_gates_3_to_9 = any(v["reasons"] != ["MISSING_LIVE_DATA"] for v in decisions.values())

    book_keys: set[str] = set()
    if odds_ingestion is not None:
        for g in odds_ingestion.games.values():
            book_keys.update(g.book_keys)

    research_ok = sum(1 for o in research_outcomes if o.status == "ok")
    research_failed = sum(1 for o in research_outcomes if o.status == "failed")
    research_cost_capped = sum(1 for o in research_outcomes if o.status == "cost_capped")
    cost_totals = {
        "total_llm_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
        "total_cache_creation_input_tokens": 0, "total_cache_read_input_tokens": 0,
        "total_web_search_requests": 0, "total_estimated_cost_usd": 0.0,
    }
    classifications: Counter = Counter()
    for o in research_outcomes:
        if o.cost:
            for k in cost_totals:
                cost_totals[k] += o.cost.get(k, 0) or 0
        if o.research_point.available and o.research_point.classification:
            classifications[o.research_point.classification] += 1

    # Phase 8A cost-controls correction, item 5's accounting fix: a real DEN@KC run made 3
    # real, billed web searches but was reported as 0 calls everywhere, because a downstream
    # (post-LLM-call) crash in run_research_for_game() propagated uncaught past
    # run_component_safely(), which swallows it and returns None - so the game's
    # LiveResearchOutcome was never even created. `run_research_for_game()` now always
    # returns a result dict instead of raising for every known failure mode (parse errors,
    # evaluator/persistence errors), so `research_outcomes` reliably contains one entry per
    # SELECTED game. These fields are named to make each stage's count independently
    # inspectable rather than collapsing them into one ambiguous "calls" number:
    #   - games_selected: the real selection this run made (independent of what happened next)
    #   - llm_calls_attempted: cost_totals["total_llm_calls"] - only increments once a real
    #     Anthropic call actually returned (so it is 1 even when parsing/evaluation/persistence
    #     failed afterward - "attempted and failed", never silently reported as zero)
    #   - llm_calls_succeeded / llm_calls_failed: whether the FULL pipeline (call, parse,
    #     evaluate, persist, ledger) completed for that game
    #   - artifacts_persisted: every status (ok/failed/cost_capped) writes a research-run
    #     artifact before returning - if this ever falls short of len(research_outcomes), an
    #     outcome was lost to something NOT caught by the layers above and needs investigation
    research_llm_calls_attempted = cost_totals["total_llm_calls"]
    research_llm_calls_failed = sum(1 for o in research_outcomes if o.status == "failed" and o.cost and o.cost.get("total_llm_calls", 0) > 0)
    research_artifacts_persisted = sum(1 for o in research_outcomes if o.run_dir)

    return {
        "season": season, "week": schedule.current_week, "games_detected": n_games,
        "feature_rows_valid": f"{n_games}/{n_games}",
        "elo_predictions": f"{n_elo}/{n_games}", "ridge_predictions": f"{n_ridge}/{n_games}", "lightgbm_predictions": f"{n_lgbm}/{n_games}",
        "odds_status": odds_status.value, "research_status": research_status.value, "injury_status": injury_status.value,
        "odds_events_seen": odds_ingestion.events_seen if odds_ingestion else 0,
        "odds_games_matched": odds_ingestion.events_matched if odds_ingestion else 0,
        "odds_events_unmatched": list(odds_ingestion.events_unmatched) if odds_ingestion else [],
        "sportsbooks_found": sorted(book_keys),
        "odds_api_usage": odds_ingestion.usage if odds_ingestion else {},
        "research_games_selected": len(selected_game_ids),
        "research_calls_made": len(research_outcomes),  # every selected game that returned an outcome is exactly one real run_research_for_game() invocation
        "research_calls_ok": research_ok, "research_calls_failed": research_failed,
        "research_cost_capped": research_cost_capped,
        "research_llm_calls_attempted": research_llm_calls_attempted,
        "research_llm_calls_succeeded": research_ok,
        "research_llm_calls_failed": research_llm_calls_failed,
        "research_artifacts_persisted": research_artifacts_persisted,
        "research_classifications": dict(classifications),
        "research_cost": cost_totals,
        # Phase 8A cost-controls correction, item 5 - the exact field names requested.
        "research": {
            "games_researched": research_ok,
            "web_searches": cost_totals["total_web_search_requests"],
            "input_tokens": cost_totals["total_input_tokens"],
            "output_tokens": cost_totals["total_output_tokens"],
            "cache_read_tokens": cost_totals["total_cache_read_input_tokens"],
            "estimated_cost_usd": cost_totals["total_estimated_cost_usd"],
        },
        "decision_counts": dict(decision_counts), "reason_code_counts": dict(reason_code_counts),
        "any_decision_reached_gates_3_to_9": any_decision_reached_gates_3_to_9,
        "dry_run": dry_run, "publish_requested": publish,
        "published_model_predictions": n_published_predictions,
        "published_official_picks": 0,  # this orchestrator pass never auto-publishes an official Best Bet - see module docstring
        "health": health.as_dict(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    default_max_research_calls = get_settings().research_max_games_per_run
    parser = argparse.ArgumentParser(
        description="Live NFL prediction/market/decision pipeline (Phase 8A). "
        "WARNING: whenever NFL_ODDS_API_KEY is set this makes real HTTP calls to The Odds "
        "API (dry-run or not); whenever NFL_RESEARCH_LLM_API_KEY is set this makes real, "
        "BILLED calls to Anthropic's Messages API (with web search) for up to "
        f"--max-research-calls games (default {default_max_research_calls}, from "
        "NFL_RESEARCH_MAX_GAMES_PER_RUN), or every target game with --research-all. Every "
        "real research call is also bounded per-game by NFL_RESEARCH_MAX_WEB_SEARCHES_PER_GAME "
        "and NFL_RESEARCH_MAX_ESTIMATED_COST_PER_GAME. See docs/PHASE8A_LIVE_PIPELINE_REPORT.md "
        "for estimated per-call cost.",
    )
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--game", type=str, default=None, dest="game_id")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run")
    parser.add_argument("--publish", action="store_true", default=False)
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--odds-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--decisions-only", action="store_true")
    parser.add_argument("--research-all", action="store_true", help="Research every target game instead of only the top --max-research-calls by trigger priority. REAL, BILLED Anthropic calls.")
    parser.add_argument("--max-research-calls", type=int, default=default_max_research_calls, help=f"Cap on real research (Anthropic) calls per run when --research-all is not set (default {default_max_research_calls}, from NFL_RESEARCH_MAX_GAMES_PER_RUN).")
    parser.add_argument("--show-decisions", action="store_true", help="Print a per-game/market_type line (decision, reason codes, model/market/research values) in addition to the JSON summary.")
    return parser


def _modes_from_args(args: argparse.Namespace) -> set[str]:
    exclusive = {"research-only": {"research"}, "odds-only": {"odds"}, "models-only": {"models"}, "decisions-only": {"decisions"}}
    for flag, modes in exclusive.items():
        if getattr(args, flag.replace("-", "_")):
            return modes
    return {"schedule", "models", "odds", "research", "injury", "decisions"}


def main() -> int:
    args = build_arg_parser().parse_args()
    modes = _modes_from_args(args)
    result = run_slate(
        week=args.week, game_id=args.game_id, dry_run=args.dry_run, publish=args.publish, modes=modes,
        research_all=args.research_all, max_research_calls=args.max_research_calls, show_decisions=args.show_decisions,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
