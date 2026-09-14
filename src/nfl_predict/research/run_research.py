"""Phase 6 orchestrator: builds the input packet, checks the historical-leakage guard,
invokes the LLM research step, parses/validates its output, evaluates it, and persists
everything (immutable research-run storage + a prospective-ledger entry) - all before any
game outcome is available.

**Design note on the evaluator**: Step 15 describes an "evaluation step" with specific,
mechanical jobs (verify internal consistency, downgrade weak sources, deduplicate evidence,
assess materiality, produce the final classification). `nfl_predict.research.evaluator`
implements this as deterministic code operating directly on the parsed `ResearchFindings` -
not a second LLM call - because those jobs are exactly the kind of reliable, auditable,
rule-based work `docs/DECISION_ENGINE.md` already insists on for anything downstream of an
LLM. `prompts/research_evaluator.md` is still written and versioned (Step 13's requirement),
kept for a possible future LLM-assisted evaluation enhancement, but the pipeline below does
not call an LLM a second time to get it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from nfl_predict.logging_conf import get_logger

from nfl_predict.research.cost_tracking import CostTracker, ResearchCostBudgetExceededError, ResearchCostConfig
from nfl_predict.research.evaluator import evaluate_research
from nfl_predict.research.historical_guard import ArchivedSourceAuthorization, assert_research_may_proceed
from nfl_predict.research.input_packet import ResearchInputPacket
from nfl_predict.research.llm_provider import LLMResearchProvider
from nfl_predict.research.parsing import ResearchOutputParseError, parse_research_output
from nfl_predict.research.prospective_ledger import ProspectiveLedgerEntry, append_ledger_entry
from nfl_predict.research.schemas import FailedResearchRun, FailureStatus
from nfl_predict.research.storage import write_research_run

logger = get_logger(__name__)

# Phase 8A cost-controls correction: the prompt is now split into a static, cacheable
# system half (matchup_research_system_v1.md, loaded by AnthropicMessagesProvider itself)
# and this dynamic, per-game half (matchup_research_v2.md) - same research methodology and
# instructions, just relocated, which is why this is v2 rather than an in-place edit of v1
# (v1 has already been used to generate real stored research runs).
MATCHUP_RESEARCH_PROMPT_VERSION = "matchup_research_v2"
RESEARCH_EVALUATOR_PROMPT_VERSION = "research_evaluator_v1"

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def build_prompt_from_template(template_text: str, packet: ResearchInputPacket) -> str:
    """Fills the prompt template's `{{placeholders}}` from the packet - never invents a
    field not present in it. Raises if the template references a placeholder this function
    doesn't know how to fill, rather than silently leaving `{{unfilled}}` in the prompt."""
    substitutions = {
        "game_id": packet.game_id, "home_team_id": packet.home_team_id, "away_team_id": packet.away_team_id,
        "season": packet.season, "week": packet.week, "kickoff_timestamp": packet.kickoff_timestamp,
        "elo_predicted_margin": packet.elo.predicted_margin, "elo_home_win_probability": packet.elo.home_win_probability,
        "ridge_predicted_margin": packet.ridge.predicted_margin if packet.ridge.available else "UNAVAILABLE",
        "lightgbm_predicted_margin": packet.lightgbm.predicted_margin if packet.lightgbm.available else "UNAVAILABLE",
        "market_home_spread_traditional": packet.market.home_spread_traditional if packet.market.available else "UNAVAILABLE",
        "market_home_moneyline": packet.market.home_moneyline if packet.market.available else "UNAVAILABLE",
        "market_total_line": packet.market.total_line if packet.market.available else "UNAVAILABLE",
        "model_market_disagreement_points": packet.model_market_disagreement_points if packet.model_market_disagreement_points is not None else "UNAVAILABLE",
    }
    unknown = set(_PLACEHOLDER_RE.findall(template_text)) - set(substitutions)
    if unknown:
        raise ValueError(f"Prompt template references unknown placeholder(s): {sorted(unknown)}")
    text = template_text
    for key, value in substitutions.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def run_research_for_game(
    packet: ResearchInputPacket,
    provider: LLMResearchProvider,
    research_prompt_template_text: str,
    run_id: str,
    now: str | None = None,
    archived_source_authorization: ArchivedSourceAuthorization | None = None,
    cost_config: ResearchCostConfig | None = None,
) -> dict:
    now_iso = now or datetime.now(timezone.utc).isoformat()
    research_id = f"{packet.game_id}_{run_id}"

    assert_research_may_proceed(
        kickoff_timestamp=packet.kickoff_timestamp, research_timestamp=now_iso, now=now_iso,
        archived_source_authorization=archived_source_authorization,
    )

    tracker = CostTracker(config=cost_config) if cost_config else CostTracker()
    prompt = build_prompt_from_template(research_prompt_template_text, packet)
    input_hash = packet.content_hash()

    try:
        call_result = provider.run_research(prompt, prompt_version=MATCHUP_RESEARCH_PROMPT_VERSION)
    except ResearchCostBudgetExceededError as e:
        # A real, computed pre-flight refusal (Phase 8A cost-controls correction) - no HTTP
        # call was made, so this is an "explicit high-uncertainty status," never a silent
        # skip and never conflated with a genuine LLM/network failure.
        failure = FailedResearchRun(
            research_id=research_id, game_id=packet.game_id, research_timestamp=now_iso,
            prompt_version=MATCHUP_RESEARCH_PROMPT_VERSION, model_provider="unknown", model_name="unknown",
            input_packet_hash=input_hash, failure_status=FailureStatus.COST_BUDGET_EXCEEDED, failure_detail=str(e),
        )
        run_dir = write_research_run(packet.season, packet.week, packet.game_id, run_id, packet, failure, "failed_run")
        logger.warning("Research run refused (COST_BUDGET_EXCEEDED) for %s: %s", packet.game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir), "cost": tracker.summary()}
    except Exception as e:
        failure = FailedResearchRun(
            research_id=research_id, game_id=packet.game_id, research_timestamp=now_iso,
            prompt_version=MATCHUP_RESEARCH_PROMPT_VERSION, model_provider="unknown", model_name="unknown",
            input_packet_hash=input_hash, failure_status=FailureStatus.LLM_FAILURE, failure_detail=str(e),
        )
        run_dir = write_research_run(packet.season, packet.week, packet.game_id, run_id, packet, failure, "failed_run")
        logger.warning("Research run failed (LLM_FAILURE) for %s: %s", packet.game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir), "cost": tracker.summary()}

    tracker.record_llm_call(
        call_result.input_tokens, call_result.output_tokens, call_result.estimated_cost_usd,
        cache_creation_input_tokens=call_result.cache_creation_input_tokens,
        cache_read_input_tokens=call_result.cache_read_input_tokens,
        web_search_requests=call_result.web_search_requests,
    )

    try:
        findings = parse_research_output(
            call_result.raw_output_text, research_id=research_id, game_id=packet.game_id,
            research_timestamp=now_iso, prompt_version=MATCHUP_RESEARCH_PROMPT_VERSION,
            model_provider=call_result.provider_name, model_name=call_result.model_name, input_packet_hash=input_hash,
        )
    except ResearchOutputParseError as e:
        failure = FailedResearchRun(
            research_id=research_id, game_id=packet.game_id, research_timestamp=now_iso,
            prompt_version=MATCHUP_RESEARCH_PROMPT_VERSION, model_provider=call_result.provider_name,
            model_name=call_result.model_name, input_packet_hash=input_hash,
            failure_status=FailureStatus.INVALID_JSON, failure_detail=str(e),
        )
        run_dir = write_research_run(packet.season, packet.week, packet.game_id, run_id, packet, failure, "failed_run")
        logger.warning("Research run failed (INVALID_JSON) for %s: %s", packet.game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir), "cost": tracker.summary()}

    # A real LLM call was made and its output parsed successfully at this point - the money
    # is spent and `tracker` already holds the real usage regardless of what happens next.
    # Everything from here on (evaluation, persistence, ledger) is wrapped so that ANY
    # unexpected failure still returns a result dict carrying that real cost data, rather
    # than raising uncaught and silently reporting this game as zero calls made (the exact
    # "0 calls despite 3 real web searches" accounting bug this correction fixes).
    try:
        evaluation = evaluate_research(
            findings, evaluation_id=f"{research_id}_eval", model_provider=call_result.provider_name,
            model_name=call_result.model_name, prompt_version=RESEARCH_EVALUATOR_PROMPT_VERSION, now=now_iso,
        )

        run_dir = write_research_run(packet.season, packet.week, packet.game_id, run_id, packet, findings, "findings", evaluation)

        source_urls = tuple(dict.fromkeys(s.source_url for c in findings.material_facts + findings.uncertain_reports for s in c.sources))
        ledger_entry = ProspectiveLedgerEntry(
            game_id=packet.game_id, season=packet.season, week=packet.week, research_id=research_id, run_id=run_id,
            kickoff_timestamp=packet.kickoff_timestamp, research_timestamp=now_iso,
            elo_predicted_margin=packet.elo.predicted_margin, elo_home_win_probability=packet.elo.home_win_probability,
            ridge_predicted_margin=packet.ridge.predicted_margin, lightgbm_predicted_margin=packet.lightgbm.predicted_margin,
            market_home_spread_traditional=packet.market.home_spread_traditional, market_home_moneyline=packet.market.home_moneyline,
            market_snapshot_timestamp=packet.market.snapshot_timestamp, research_classification=evaluation.final_classification.value,
            materiality_level=int(evaluation.overall_materiality), source_urls=source_urls,
            unresolved_risks=tuple(findings.missing_information), frozen_at=now_iso,
        )
        append_ledger_entry(ledger_entry)
    except Exception as e:
        failure = FailedResearchRun(
            research_id=research_id, game_id=packet.game_id, research_timestamp=now_iso,
            prompt_version=MATCHUP_RESEARCH_PROMPT_VERSION, model_provider=call_result.provider_name,
            model_name=call_result.model_name, input_packet_hash=input_hash,
            failure_status=FailureStatus.DOWNSTREAM_PROCESSING_FAILURE, failure_detail=f"{type(e).__name__}: {e}",
        )
        run_dir = write_research_run(packet.season, packet.week, packet.game_id, run_id, packet, failure, "failed_run")
        logger.error("Research run failed (DOWNSTREAM_PROCESSING_FAILURE) for %s after a successful, parsed LLM call: %s", packet.game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir), "cost": tracker.summary()}

    logger.info("Research run complete for %s: classification=%s materiality=%s", packet.game_id, evaluation.final_classification.value, evaluation.overall_materiality.name)
    return {
        "status": "ok", "run_dir": str(run_dir), "classification": evaluation.final_classification.value,
        "materiality": evaluation.overall_materiality.name, "cost": tracker.summary(),
    }
