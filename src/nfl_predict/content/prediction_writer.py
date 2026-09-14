"""Phase 8B follow-up: generates the public-facing "why" preview for one game.

Runs OFFLINE as a pipeline step - see docs/WEBSITE_SPEC.md#the-website-never-calls-an-llm-at-request-time.
`write_prediction_preview()` gathers the exact same real, persisted data
`nfl_predict.api.reconstruction` already assembles for the website's own game-detail
endpoint (never a second, competing way of reading the same state), fills the versioned
`prompts/prediction_writer_v1.md` template, calls a provider, and persists the result
immutably via `content.storage` - regardless of success or failure, so a failed generation
is always an explicit, diagnosable record, never a silently-missing article.

No number is ever computed here beyond simple TEXT FORMATTING of already-computed values
(e.g. "KC by 2.5" from a stored spread) - the same kind of display-only rendering the
website's own frontend already does, never a new derivation. `disagreement_text` reuses the
identical formula `nfl_predict.api.main.game_detail` already uses, not a second one.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nfl_predict.api import consumer_language as lang
from nfl_predict.api import reconstruction as recon
from nfl_predict.config import get_llm_pricing_config, get_settings
from nfl_predict.content.schemas import FailedPreviewRun, GamePreview, PreviewFailureStatus
from nfl_predict.content.storage import write_preview_run
from nfl_predict.logging_conf import get_logger
from nfl_predict.research.cost_tracking import (
    ResearchCostBudgetExceededError,
    assert_known_cost_floor_within_budget,
    estimate_cost_usd,
)

logger = get_logger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "prediction_writer_v2.md"
PREDICTION_WRITER_PROMPT_VERSION = "prediction_writer_v2"
_UNSET = object()

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class ContentWriterNotConfiguredError(Exception):
    """Raised when a live content-writer provider is used without its required credentials
    configured - mirrors `research.llm_provider.LLMProviderNotConfiguredError`."""


class ContentWriterResponseError(Exception):
    """Raised when a live response cannot be turned into article text (no text content,
    a refusal, an unexpected stop reason)."""


@dataclass(frozen=True)
class ContentWriteResult:
    provider_name: str
    model_name: str
    text: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None


class ContentWriterProvider(ABC):
    @abstractmethod
    def write(self, prompt: str) -> ContentWriteResult: ...


class AnthropicContentWriterProvider(ContentWriterProvider):
    """A genuine Anthropic Messages API call - no tools, no web search (the article is
    written entirely from context already gathered), so this is the cheapest possible real
    call shape: plain prose in, plain prose out."""

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self, model_name: str = "claude-sonnet-5", api_key: str | None = None, max_tokens: int = 1024,
        pricing_config: dict | None = None, max_estimated_cost_usd: float | None = _UNSET,
    ):
        self._api_key = api_key if api_key is not None else get_settings().research_llm_api_key
        if not self._api_key:
            raise ContentWriterNotConfiguredError(
                "No LLM API key is configured - set NFL_RESEARCH_LLM_API_KEY in .env before "
                "using AnthropicContentWriterProvider. Live calls are refused, not fabricated."
            )
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._pricing_config = pricing_config if pricing_config is not None else get_llm_pricing_config()
        self._max_estimated_cost_usd = (
            get_settings().content_max_estimated_cost_per_game_usd if max_estimated_cost_usd is _UNSET else max_estimated_cost_usd
        )

    def _post(self, body: dict[str, Any]) -> dict:
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL, data=payload,
            headers={"x-api-key": self._api_key, "anthropic-version": self.API_VERSION, "content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # pragma: no cover - never exercised without a live key
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - never exercised without a live key
            detail = e.read().decode("utf-8", errors="replace")[:1000]
            raise ContentWriterResponseError(f"Anthropic Messages API returned HTTP {e.code}: {detail}") from e

    def write(self, prompt: str) -> ContentWriteResult:
        assert_known_cost_floor_within_budget(
            self._model_name, self._pricing_config, self._max_tokens, 0, self._max_estimated_cost_usd,
        )
        payload = self._post({"model": self._model_name, "max_tokens": self._max_tokens, "messages": [{"role": "user", "content": prompt}]})
        usage = payload.get("usage", {})
        text = "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text").strip()
        if not text:
            raise ContentWriterResponseError(f"No text content in response: stop_reason={payload.get('stop_reason')!r}")
        cost = estimate_cost_usd(
            self._model_name, self._pricing_config,
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
        )
        return ContentWriteResult(
            provider_name="anthropic", model_name=self._model_name, text=text,
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), estimated_cost_usd=cost,
        )


class FixtureContentWriterProvider(ContentWriterProvider):
    """Offline adapter for tests/local development - reads a canned JSON fixture, makes no
    network call. Fixture shape: `{"<key>": {"text": ..., "model_name": ..., ...}}`."""

    def __init__(self, fixture_path: str | Path, key: str = "default"):
        self._fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        self._key = key

    def write(self, prompt: str) -> ContentWriteResult:
        entry = self._fixture[self._key]
        return ContentWriteResult(
            provider_name="fixture", model_name=entry.get("model_name", "fixture-model"), text=entry["text"],
            input_tokens=entry.get("input_tokens"), output_tokens=entry.get("output_tokens"),
            estimated_cost_usd=entry.get("estimated_cost_usd"),
        )


def _format_margin_text(margin: float | None, home_abbr: str, away_abbr: str) -> str:
    if margin is None:
        return "not available"
    favored = home_abbr if margin >= 0 else away_abbr
    return f"{favored} by {abs(margin):.1f}"


def _format_prob_text(prob: float | None) -> str:
    return "not available" if prob is None else f"{prob * 100:.0f}%"


def _format_market_summary(market: dict | None, home_abbr: str) -> str:
    if not market or not market.get("available"):
        return "No market line is available yet."
    parts = []
    spread = market.get("home_spread_traditional")
    if spread is not None:
        sign = "+" if spread > 0 else ""
        parts.append(f"{home_abbr} {sign}{spread:.1f}")
    no_vig = market.get("no_vig_home_win_probability")
    if no_vig is not None:
        parts.append(f"{home_abbr} no-vig win probability {no_vig * 100:.0f}%")
    n_books = len(market.get("consensus_book_keys") or ())
    if n_books:
        parts.append(f"consensus of {n_books} sportsbook(s)")
    return "; ".join(parts) if parts else "No usable market line."


def _format_disagreement_text(elo_margin: float | None, home_spread: float | None, home_abbr: str, away_abbr: str) -> str:
    """Reuses the exact formula `nfl_predict.api.main.game_detail` already computes with -
    never a second, competing derivation."""
    if elo_margin is None or home_spread is None:
        return "not computable (missing model or market data)"
    disagreement = elo_margin - (-home_spread)
    leaning_team = home_abbr if disagreement > 0 else away_abbr
    return f"about {abs(disagreement):.1f} points (the model leans more toward {leaning_team} than the market does)"


def _format_decision_reasons_text(record: dict | None) -> str:
    if record is None:
        return "no decision recorded yet"
    codes = record.get("reason_codes") or []
    if not codes:
        return "no reasons recorded"
    return "; ".join(lang.translate_reason_code(c) for c in codes)


def _format_publication_status_text(record: dict | None, market_type: str, published_market_types: set[str]) -> str:
    """States the one fact the reason codes alone can't convey: whether THIS market_type was
    actually published as the game's official pick. A QUALIFIED_BET decision is a real,
    independent-per-market-type verdict - publishing it is always a separate, deliberate step
    (never automatic, and never both market types for one game), so the reason codes alone
    (e.g. NO_DEMONSTRATED_EDGE) can make an actually-published pick read as if nothing was
    recommended. See docs/DECISION_ENGINE.md."""
    if record is None or record.get("decision") != "QUALIFIED_BET":
        return "not applicable - this market type did not qualify"
    if market_type in published_market_types:
        return "YES - this was published as this game's official Best Bet"
    return "NO - this qualified under our criteria but was NOT published as an official pick (at most one market type is ever published per game)"


def _format_research_summary_text(research: dict | None) -> str:
    if research is None:
        return "No research has been conducted for this game yet."
    if not research.get("available"):
        return "Research was attempted for this game but did not complete successfully this cycle."
    classification = lang.translate_research_classification(research["classification"])
    parts = [f"Classification: {classification}."]
    if research.get("summary"):
        parts.append(research["summary"])
    risks = research.get("unresolved_risks") or ()
    if risks:
        parts.append("Still unresolved: " + "; ".join(risks) + ".")
    return " ".join(parts)


def gather_game_content_context(
    season: int, week: int, game_id: str, home_team_id: str, away_team_id: str, kickoff_timestamp: str | None,
) -> dict:
    """Pure data gathering - reuses `nfl_predict.api.reconstruction`'s real, already-tested
    functions, the exact same ones the website's own `/api/nfl/games/{game_id}` endpoint
    calls, so the article is always built from the same real state a visitor would see."""
    teams = recon.team_lookup()
    home = teams.get(home_team_id, {})
    away = teams.get(away_team_id, {})
    model = recon.latest_model_prediction(game_id)
    market = recon.latest_market_point(season, week, game_id)
    decisions = recon.latest_decisions_for_game(season, week, game_id)
    research = recon.latest_research_summary(season, week, game_id)
    # Real bug caught by a user reading a generated article: without this, the model only
    # sees a QUALIFIED_BET decision's reason codes (e.g. NO_DEMONSTRATED_EDGE) and has no way
    # to know whether that market_type was actually published as an official pick - it wrote
    # "the system is not recommending a bet on either side" for a game where the moneyline
    # WAS a published Best Bet, directly contradicting what the same page shows. Passing the
    # real published state closes that gap.
    published_market_types = sorted(recon.published_best_bet_market_types(game_id))

    return {
        "game_id": game_id, "season": season, "week": week,
        "home_team_id": home_team_id, "away_team_id": away_team_id,
        "home_team_name": home.get("name", home_team_id), "away_team_name": away.get("name", away_team_id),
        "home_team_abbr": home.get("abbr", home_team_id), "away_team_abbr": away.get("abbr", away_team_id),
        "kickoff_timestamp": kickoff_timestamp,
        "model": model,
        "market": asdict(market) if market is not None else None,
        "decisions": decisions,
        "research": research,
        "published_market_types": published_market_types,
    }


def build_prompt_from_template(template_text: str, context: dict) -> str:
    """Fills the prompt template's `{{placeholders}}` from `context` - never invents a field
    not derivable from it. Raises if the template references a placeholder this function
    doesn't know how to fill, rather than silently leaving `{{unfilled}}` in the prompt."""
    home_abbr, away_abbr = context["home_team_abbr"], context["away_team_abbr"]
    model = context["model"]
    market_dict = context["market"]
    decisions = context["decisions"]

    home_spread = market_dict.get("home_spread_traditional") if market_dict and market_dict.get("available") else None
    spread_record = decisions.get("spread")
    moneyline_record = decisions.get("moneyline")
    published_market_types = set(context.get("published_market_types") or ())

    substitutions = {
        "away_team_name": context["away_team_name"], "home_team_name": context["home_team_name"],
        "season": context["season"], "week": context["week"],
        "kickoff_timestamp": context["kickoff_timestamp"] or "TBD",
        "elo_margin_text": _format_margin_text((model or {}).get("elo_predicted_margin"), home_abbr, away_abbr),
        "home_team_abbr": home_abbr,
        "elo_home_win_prob_text": _format_prob_text((model or {}).get("elo_home_win_probability")),
        "ridge_margin_text": _format_margin_text((model or {}).get("ridge_predicted_margin"), home_abbr, away_abbr),
        "lightgbm_margin_text": _format_margin_text((model or {}).get("lightgbm_predicted_margin"), home_abbr, away_abbr),
        "market_summary_text": _format_market_summary(market_dict, home_abbr),
        "disagreement_text": _format_disagreement_text((model or {}).get("elo_predicted_margin"), home_spread, home_abbr, away_abbr),
        "spread_decision_label": lang.translate_decision(spread_record["decision"]) if spread_record else "not yet decided",
        "spread_decision_reasons_text": _format_decision_reasons_text(spread_record),
        "spread_published_text": _format_publication_status_text(spread_record, "spread", published_market_types),
        "moneyline_decision_label": lang.translate_decision(moneyline_record["decision"]) if moneyline_record else "not yet decided",
        "moneyline_decision_reasons_text": _format_decision_reasons_text(moneyline_record),
        "moneyline_published_text": _format_publication_status_text(moneyline_record, "moneyline", published_market_types),
        "research_summary_text": _format_research_summary_text(context["research"]),
    }
    unknown = set(_PLACEHOLDER_RE.findall(template_text)) - set(substitutions)
    if unknown:
        raise ValueError(f"Prompt template references unknown placeholder(s): {sorted(unknown)}")
    text = template_text
    for key, value in substitutions.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def write_prediction_preview(
    season: int, week: int, game_id: str, home_team_id: str, away_team_id: str, kickoff_timestamp: str | None,
    provider: ContentWriterProvider, prompt_template_text: str, run_id: str, now: str | None = None,
) -> dict:
    now_iso = now or datetime.now(timezone.utc).isoformat()
    preview_id = f"{game_id}_{run_id}"
    context = gather_game_content_context(season, week, game_id, home_team_id, away_team_id, kickoff_timestamp)
    prompt = build_prompt_from_template(prompt_template_text, context)

    try:
        result = provider.write(prompt)
    except ContentWriterNotConfiguredError as e:
        failure = FailedPreviewRun(
            preview_id=preview_id, game_id=game_id, generated_at=now_iso, prompt_version=PREDICTION_WRITER_PROMPT_VERSION,
            context_hash="", failure_status=PreviewFailureStatus.PROVIDER_UNAVAILABLE, failure_detail=str(e),
        )
        run_dir = write_preview_run(season, week, game_id, run_id, context, failure, "failed_run")
        logger.warning("Preview generation refused (PROVIDER_UNAVAILABLE) for %s: %s", game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir)}
    except ResearchCostBudgetExceededError as e:
        failure = FailedPreviewRun(
            preview_id=preview_id, game_id=game_id, generated_at=now_iso, prompt_version=PREDICTION_WRITER_PROMPT_VERSION,
            context_hash="", failure_status=PreviewFailureStatus.COST_BUDGET_EXCEEDED, failure_detail=str(e),
        )
        run_dir = write_preview_run(season, week, game_id, run_id, context, failure, "failed_run")
        logger.warning("Preview generation refused (COST_BUDGET_EXCEEDED) for %s: %s", game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir)}
    except ContentWriterResponseError as e:
        failure = FailedPreviewRun(
            preview_id=preview_id, game_id=game_id, generated_at=now_iso, prompt_version=PREDICTION_WRITER_PROMPT_VERSION,
            context_hash="", failure_status=PreviewFailureStatus.EMPTY_RESPONSE, failure_detail=str(e),
        )
        run_dir = write_preview_run(season, week, game_id, run_id, context, failure, "failed_run")
        logger.warning("Preview generation failed (EMPTY_RESPONSE) for %s: %s", game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir)}
    except Exception as e:
        failure = FailedPreviewRun(
            preview_id=preview_id, game_id=game_id, generated_at=now_iso, prompt_version=PREDICTION_WRITER_PROMPT_VERSION,
            context_hash="", failure_status=PreviewFailureStatus.LLM_FAILURE, failure_detail=str(e),
        )
        run_dir = write_preview_run(season, week, game_id, run_id, context, failure, "failed_run")
        logger.warning("Preview generation failed (LLM_FAILURE) for %s: %s", game_id, e)
        return {"status": "failed", "failure_status": failure.failure_status.value, "run_dir": str(run_dir)}

    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    preview = GamePreview(
        preview_id=preview_id, game_id=game_id, generated_at=now_iso, prompt_version=PREDICTION_WRITER_PROMPT_VERSION,
        model_provider=result.provider_name, model_name=result.model_name, context_hash=context_hash, text=result.text,
    )
    run_dir = write_preview_run(season, week, game_id, run_id, context, preview, "preview")
    logger.info("Preview generated for %s (%d chars)", game_id, len(result.text))
    return {
        "status": "ok", "run_dir": str(run_dir), "text": result.text,
        "cost": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "estimated_cost_usd": result.estimated_cost_usd},
    }
