"""Phase 8A correction, Steps 3-4: actually invokes the LLM research provider for
triggered games, reusing Phase 6's `run_research_for_game` pipeline verbatim (build packet
-> invoke provider -> parse -> evaluate -> persist -> ledger entry) - never a second,
parallel implementation, and never a raw unvalidated LLM string handed to the decision
engine. `ResearchPoint` is only ever built from the freshly-persisted, evaluated artifact
(`nfl_predict.decision.input_packet.build_research_point_from_stored_run`), the same
function Phase 7's historical/pilot path already uses.

Selection of which games actually get researched follows Phase 6's trigger design
(`nfl_predict.research.trigger`): the trigger score is a RANKING signal only (per that
module's own docstring, "never gates research"), used here purely to decide research ORDER
under a real per-run cost cap (`max_calls`) - it never decides a game is ineligible for
research on its own. `research_all=True` bypasses the cap entirely (Step 4's explicit
`--research-all` validation mode).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from nfl_predict.decision.input_packet import build_research_point_from_stored_run
from nfl_predict.decision.schemas import MarketPoint, ResearchPoint
from nfl_predict.logging_conf import get_logger
from nfl_predict.research.cost_tracking import ResearchCostConfig
from nfl_predict.research.input_packet import MarketContext, ResearchInputPacket, build_live_input_packet
from nfl_predict.research.llm_provider import LLMResearchProvider
from nfl_predict.research.run_research import run_research_for_game
from nfl_predict.research.storage import read_research_run
from nfl_predict.research.trigger import TriggerInputs, TriggerResult, compute_research_priority

logger = get_logger(__name__)

MATCHUP_RESEARCH_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "matchup_research_v2.md"
# ^ the DYNAMIC (per-game) half only - the static half (matchup_research_system_v1.md) is
# loaded and cached separately by AnthropicMessagesProvider itself (Phase 8A cost-controls
# correction). See prompts/matchup_research_v2.md's header for why this is v2, not v1.


@dataclass(frozen=True)
class LiveResearchOutcome:
    game_id: str
    invoked: bool  # True iff run_research_for_game was actually called (a real, billed API call when the live provider is used)
    status: str  # "not_selected" | "ok" | "failed" | "cost_capped"
    research_point: ResearchPoint
    priority: TriggerResult | None
    cost: dict | None
    run_dir: str | None
    failure_status: str | None = None  # Phase 8A cost-controls correction: the underlying FailureStatus value (e.g. "COST_BUDGET_EXCEEDED"), for explicit machine-readable reporting


def build_live_research_input_packet_with_market(game_id: str, market: MarketPoint | None) -> ResearchInputPacket:
    """`build_live_input_packet` (Phase 6) always marks its market section unavailable -
    correct when no live odds exist, wrong now that Phase 8A's `odds_ingestion` may have
    real numbers. When `market` carries real data, this replaces just the market section
    (and the disagreement figure derived from it) with the real values - every other field
    (Elo continuation, Ridge/LightGBM availability) is untouched."""
    packet = build_live_input_packet(game_id)
    if market is None or not market.available:
        return packet

    market_ctx = MarketContext(
        available=True,
        home_spread_traditional=market.home_spread_traditional, home_spread_price=market.home_spread_price,
        away_spread_price=market.away_spread_price, home_moneyline=market.home_moneyline,
        away_moneyline=market.away_moneyline, no_vig_home_win_probability=market.no_vig_home_win_probability,
        snapshot_timestamp=market.snapshot_timestamp, snapshot_type="live_the_odds_api",
    )
    disagreement = None
    if packet.elo.available and packet.elo.predicted_margin is not None and market.home_spread_traditional is not None:
        disagreement = packet.elo.predicted_margin - (-market.home_spread_traditional)
    return dataclasses.replace(packet, market=market_ctx, model_market_disagreement_points=disagreement)


def select_games_for_research(
    priorities: dict[str, TriggerResult], research_all: bool, max_calls: int,
) -> list[str]:
    """Every game is ALWAYS_RESEARCHABLE in principle (`trigger.py`'s own documented
    stance) - this just decides ORDER and, absent `--research-all`, how many of them this
    run can actually afford to research (Step 9's cost cap)."""
    ordered = sorted(priorities.keys(), key=lambda gid: priorities[gid].priority_score, reverse=True)
    if research_all:
        return ordered
    return ordered[:max_calls]


def run_live_research_for_game(
    game_id: str, season: int, week: int, provider: LLMResearchProvider, market: MarketPoint | None,
    priority: TriggerResult | None, run_id: str, now: str, cost_config: ResearchCostConfig | None = None,
) -> LiveResearchOutcome:
    packet = build_live_research_input_packet_with_market(game_id, market)
    prompt_template_text = MATCHUP_RESEARCH_PROMPT_PATH.read_text(encoding="utf-8")

    result = run_research_for_game(
        packet=packet, provider=provider, research_prompt_template_text=prompt_template_text,
        run_id=run_id, now=now, cost_config=cost_config,
    )

    if result["status"] != "ok":
        failure_status = result.get("failure_status")
        # COST_BUDGET_EXCEEDED is an explicit, distinct status (Phase 8A cost-controls
        # correction) - a real, pre-flight refusal, never conflated with a genuine
        # LLM/network/parse failure in reporting.
        outcome_status = "cost_capped" if failure_status == "COST_BUDGET_EXCEEDED" else "failed"
        logger.warning("Live research %s for %s: %s", outcome_status, game_id, failure_status)
        return LiveResearchOutcome(
            game_id=game_id, invoked=True, status=outcome_status, research_point=ResearchPoint(available=False),
            priority=priority, cost=result.get("cost"), run_dir=result.get("run_dir"), failure_status=failure_status,
        )

    stored = read_research_run(season, week, game_id, run_id)
    research_point = build_research_point_from_stored_run(stored, now)
    return LiveResearchOutcome(
        game_id=game_id, invoked=True, status="ok", research_point=research_point,
        priority=priority, cost=result.get("cost"), run_dir=result.get("run_dir"),
    )


def compute_trigger_priorities(
    disagreement_points_by_game: dict[str, float | None], cross_model_dispersion_by_game: dict[str, float | None],
) -> dict[str, TriggerResult]:
    return {
        game_id: compute_research_priority(TriggerInputs(
            model_market_disagreement_points=disagreement_points_by_game.get(game_id),
            cross_model_disagreement_points=cross_model_dispersion_by_game.get(game_id),
        ))
        for game_id in disagreement_points_by_game
    }
