"""Phase 6 Step 21: a provider-neutral LLM research interface.

`LLMResearchProvider` is the abstract contract - `run_research()` and `run_evaluation()` -
so nothing else in the platform hard-codes a specific vendor. Every concrete provider must
record its own `provider_name`/`model_name` on every output (Step 13/26's requirement that
model/provider is always recorded).

`AnthropicMessagesProvider` is the first real adapter: it builds a genuine Messages API
request (endpoint, headers, payload) and is gated on `NFL_RESEARCH_LLM_API_KEY` via
`nfl_predict.config.get_settings()`, exactly like Phase 5's `TheOddsAPIProvider` - if the
key is unset it raises `LLMProviderNotConfiguredError` rather than fabricating a response.

`FixtureLLMProvider` is a second, deterministic, offline adapter that reads canned JSON
fixtures - used by every test in this package so none of them need network access or a
live key - see `manual_provider.py` for the provider actually used for Phase 6's real pilot
(Step 25), which is explicit about not being an automated API call.

**Phase 8A correction - INVALID_JSON postmortem (see
docs/PHASE8A_LIVE_PIPELINE_REPORT.md for the full incident report):** the first live
validation attempt's web searches succeeded, but every research call then failed with
`Expecting value: line 1 column 1` - `json.loads()` on an empty or non-JSON string. Root
cause: a Messages API turn that uses the server-side web-search tool returns MULTIPLE
content blocks (narration `text` blocks, `server_tool_use` blocks, `web_search_tool_result`
blocks), and the old code concatenated every `text`-type block together and fed the result
straight to `json.loads()` - if the model's first text block was narration ("Let me look
into...") rather than the final JSON, the concatenated string was not valid JSON from
position 0, or a mid-turn `pause_turn` response had no final text block at all (an empty
string, giving exactly that error). Fixed by never asking the model to hand back JSON as
free text at all: `AnthropicMessagesProvider` now gives the model a second, custom tool
(`submit_research_findings`, `input_schema` = the real `ResearchFindings` shape) alongside
the hosted web-search tool, with `tool_choice: "auto"` (NOT forced - forcing the submit tool
from turn 1 would prevent the model from searching first). The model is instructed to search
as needed, then call `submit_research_findings` exactly once as its final action; when it
does, Anthropic has already validated `input` against the schema, so it arrives as a real
Python dict, not text to be parsed - no `json.loads()` on model prose, ever. `pause_turn`
(the server-tool turn-continuation signal for long research turns) is handled by resending
the conversation with the partial assistant turn appended, up to a small bounded number of
continuations.

**Phase 8A cost-controls correction:** real usage now recorded in full
(`cache_creation_input_tokens`/`cache_read_input_tokens`/`web_search_requests`, per
Anthropic's actual `usage` object), a real dollar cost computed from
`config/llm_pricing.yaml` (never hard-coded), a pre-flight known-cost-floor budget guard,
and prompt caching for the STATIC half of the research prompt: `matchup_research_system_v1.md`
(role framing, source hierarchy, fact/opinion rules, materiality rubric, hard constraints)
is sent as the `system` parameter with `cache_control`, and the `tools` array (the web-search
tool definition AND the `submit_research_findings` schema, both identical on every call) is
cached too via `cache_control` on its last entry. Only `matchup_research_v2.md`'s per-game
"Context provided to you" section - genuinely different every call - is ever sent as the
uncached `messages` user turn.
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nfl_predict.config import get_llm_pricing_config, get_settings
from nfl_predict.logging_conf import get_logger
from nfl_predict.research.cost_tracking import assert_known_cost_floor_within_budget, estimate_cost_usd
from nfl_predict.research.schemas import ClaimCategory, MaterialityLevel, ResearchClassification, SourceTier

logger = get_logger(__name__)

SUBMIT_TOOL_NAME = "submit_research_findings"
MAX_PAUSE_TURN_CONTINUATIONS = 5
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "matchup_research_system_v1.md"
_UNSET = object()  # distinguishes "not passed, use configured default" from an explicit None ("no cap")


class LLMProviderNotConfiguredError(Exception):
    """Raised when a live LLM provider is used without its required credentials configured."""


class AnthropicResponseError(Exception):
    """Raised when a live Anthropic response cannot be turned into research findings - e.g.
    the model never called `submit_research_findings`, a refusal/max_tokens stop, or too
    many `pause_turn` continuations. The message is a compact, secret-free diagnostic
    (response id, stop_reason, content block types/counts, usage) - never the raw payload,
    and never dumped into a giant log line - intended to be persisted as-is in the
    resulting `FailedResearchRun.failure_detail` (Phase 8A correction Step B1/B5)."""


@dataclass(frozen=True)
class LLMCallResult:
    provider_name: str
    model_name: str
    raw_output_text: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    web_search_requests: int | None = None


class LLMResearchProvider(ABC):
    @abstractmethod
    def run_research(self, prompt: str, prompt_version: str) -> LLMCallResult: ...

    @abstractmethod
    def run_evaluation(self, prompt: str, prompt_version: str) -> LLMCallResult: ...


def _research_output_json_schema() -> dict:
    """The `submit_research_findings` tool's `input_schema` - mirrors
    `nfl_predict.research.parsing.parse_research_output`'s expected shape and
    `prompts/matchup_research.md`'s documented output format exactly, built from the same
    enums `nfl_predict.research.schemas` defines so the schema can never silently drift from
    what the parser actually accepts.

    **Every property is listed in `required`** (nullable via `type: [..., "null"]` where a
    field is genuinely optional) and every object sets `additionalProperties: false` - the
    conventional shape Anthropic's (and OpenAI's) STRICT tool-use/structured-output mode
    requires to actually guarantee schema conformance. This was added after a real single-
    game validation call (Phase 8A correction) returned a tool call missing
    `research_classification` under plain (non-strict) tool use - proving the field being
    merely `required` in a non-strict schema is not itself enough to guarantee the model
    includes it."""
    source_schema = {
        "type": "object",
        "properties": {
            "source_url": {"type": "string"},
            "source_title": {"type": "string"},
            "publisher": {"type": "string"},
            "source_tier": {"type": "string", "enum": [t.value for t in SourceTier]},
            "retrieval_timestamp": {"type": "string"},
            "publication_timestamp": {"type": ["string", "null"]},
        },
        "required": ["source_url", "source_title", "publisher", "source_tier", "retrieval_timestamp", "publication_timestamp"],
        "additionalProperties": False,
    }
    claim_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "category": {"type": "string", "enum": [c.value for c in ClaimCategory]},
            "sources": {"type": "array", "items": source_schema},
            "materiality_level": {"type": "integer", "enum": [int(m) for m in MaterialityLevel]},
            # No `minimum`/`maximum` here - Anthropic's strict custom-tool schema validator
            # rejects those keywords on `number` types (confirmed via a real 400 response:
            # "tools.1.custom: For 'number' type, properties maximum, minimum are not
            # supported"). The [0.0, 1.0] range is still enforced downstream by
            # `nfl_predict.research.schemas.Claim.__post_init__`, which raises if violated.
            "confidence_in_fact": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["text", "category", "sources", "materiality_level", "confidence_in_fact", "reason"],
        "additionalProperties": False,
    }
    external_prediction_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "prediction_text": {"type": "string"},
            "market": {"type": ["string", "null"]},
            "line_at_publication": {"type": ["string", "null"]},
            "publication_timestamp": {"type": ["string", "null"]},
            "stated_confidence": {"type": ["string", "null"]},
        },
        "required": ["source", "prediction_text", "market", "line_at_publication", "publication_timestamp", "stated_confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        # Short, critical fields FIRST - a real call once returned a tool input truncated
        # partway through (present: the large citation-heavy arrays; missing: every short
        # trailing field, consistent with hitting max_tokens mid-generation). Property order
        # isn't a hard generation-order guarantee, but costs nothing and may help; the real
        # fix is the larger max_tokens budget above.
        "properties": {
            "research_classification": {"type": "string", "enum": [c.value for c in ResearchClassification]},
            "qb_status": {"type": "string"},
            "ol_status": {"type": "string"},
            "skill_position_status": {"type": "string"},
            "defensive_personnel_status": {"type": "string"},
            "weather_status": {"type": "string"},
            "coaching_status": {"type": "string"},
            # A real call once submitted this as a single comma-joined string instead of an
            # array (caught by the strict parser in research/parsing.py, which correctly
            # refused to coerce it rather than silently splitting on commas - a genuine
            # array item can legitimately contain a comma). This description is the
            # mitigation: make the required shape unambiguous rather than loosen validation.
            "missing_information": {
                "type": "array", "items": {"type": "string"},
                "description": "One array entry per distinct missing item - never a single string with items joined by commas or newlines, even if there is only one item.",
            },
            "material_facts": {"type": "array", "items": claim_schema},
            "uncertain_reports": {"type": "array", "items": claim_schema},
            "external_model_opinions": {"type": "array", "items": external_prediction_schema},
            "analyst_opinions": {"type": "array", "items": claim_schema},
        },
        "required": [
            "research_classification", "qb_status", "ol_status", "skill_position_status",
            "defensive_personnel_status", "weather_status", "coaching_status", "missing_information",
            "material_facts", "uncertain_reports", "external_model_opinions", "analyst_opinions",
        ],
        "additionalProperties": False,
    }


class AnthropicMessagesProvider(LLMResearchProvider):
    """Real Messages API request construction, gated on `NFL_RESEARCH_LLM_API_KEY`.

    `matchup_research.md`'s prompt asks the model to find and cite real, current sources
    (injury reports, beat-reporter stories, line-movement context) - a plain Messages API
    call has no browsing capability and cannot do that from training data alone for a game
    played after the model's knowledge cutoff. So by default (`enable_web_search=True`) the
    request includes Anthropic's server-side web-search tool (`web_search_20250305`),
    letting the model actually search the web; `max_web_search_uses` bounds how many
    searches one call may make (default 3, per the Phase 8A correction's cost cap). The
    model's FINAL structured answer is obtained via a second, custom tool
    (`submit_research_findings`) rather than free-text JSON - see this module's docstring
    for why free-text JSON parsing was the root cause of a real INVALID_JSON failure."""

    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"
    WEB_SEARCH_TOOL_TYPE = "web_search_20250305"

    def __init__(
        self, model_name: str = "claude-sonnet-5", api_key: str | None = None,
        enable_web_search: bool = True, max_web_search_uses: int | None = None, max_tokens: int = 16000,
        system_prompt_path: Path | None = DEFAULT_SYSTEM_PROMPT_PATH, pricing_config: dict | None = None,
        max_estimated_cost_usd: float | None = _UNSET,
    ):
        # A real single-game validation call (Phase 8A correction) returned a
        # submit_research_findings call missing ALL of the short trailing fields
        # (qb_status..research_classification) while the earlier, larger array fields
        # (material_facts, sources cited from 3 real web searches) were present - consistent
        # with hitting max_tokens mid-generation before reaching the end of the object.
        # 16000 gives real research (with actual source citations) more room before that.
        self._api_key = api_key if api_key is not None else get_settings().research_llm_api_key
        if not self._api_key:
            raise LLMProviderNotConfiguredError(
                "No LLM API key is configured - set NFL_RESEARCH_LLM_API_KEY in .env before "
                "using AnthropicMessagesProvider. Live calls are refused, not fabricated."
            )
        self._model_name = model_name
        self._enable_web_search = enable_web_search
        self._max_web_search_uses = max_web_search_uses if max_web_search_uses is not None else get_settings().research_max_web_searches_per_game
        self._max_tokens = max_tokens
        # Read once at construction (not per-call) - the whole point is that this content is
        # STABLE across every call this provider instance makes, which is what makes it
        # cacheable in the first place (see _build_request's `system` block).
        self._system_text = system_prompt_path.read_text(encoding="utf-8") if system_prompt_path else None
        self._pricing_config = pricing_config if pricing_config is not None else get_llm_pricing_config()
        self._max_estimated_cost_usd = (
            get_settings().research_max_estimated_cost_per_game_usd if max_estimated_cost_usd is _UNSET else max_estimated_cost_usd
        )

    def _tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if self._enable_web_search:
            tools.append({"type": self.WEB_SEARCH_TOOL_TYPE, "name": "web_search", "max_uses": self._max_web_search_uses})
        tools.append({
            "name": SUBMIT_TOOL_NAME,
            "description": (
                "Submit the final, complete research findings for this matchup. Call this "
                "exactly once, as your last action, after any web searches you needed are "
                "done. Every field is required, including research_classification - if a "
                "category has nothing to report, use an empty array or a status string like "
                "'No change reported.' rather than omitting the field."
            ),
            "input_schema": _research_output_json_schema(),
            # NOT strict: a real call confirmed Anthropic's strict-mode grammar compiler
            # rejects this schema as too large ("The compiled grammar is too large...
            # Simplify your tool schemas or reduce the number of strict tools") once nested
            # claim/source/external-prediction arrays are all in play - shrinking the schema
            # enough to fit would mean dropping real fields the parser needs. Falling back to
            # plain (non-strict) tool use, which a real call already proved reliably invokes
            # the tool with a well-formed (if not always 100%-complete) object - the
            # `required`/`additionalProperties` shape and the description above still guide
            # generation even without a hard guarantee, and a genuinely incomplete
            # submission still fails safely into a diagnosable FailedResearchRun rather than
            # a bare JSONDecodeError (see this module's docstring).
            #
            # cache_control on this, the LAST tool in the array, marks the ENTIRE tools
            # array (this schema + the web-search tool definition above it) as a stable,
            # cacheable prefix - both are byte-identical on every call this provider makes
            # (Phase 8A cost-controls correction, item 3: "research schema/tool definition").
            "cache_control": {"type": "ephemeral"},
        })
        return tools

    def _build_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model_name, "max_tokens": self._max_tokens, "messages": messages,
            "tools": self._tools(),
            # "auto" (the default) is deliberate, not omitted-by-accident: forcing
            # tool_choice to SUBMIT_TOOL_NAME would require the model to call it on its
            # very first turn, before it could search at all.
            "tool_choice": {"type": "auto"},
        }
        if self._system_text is not None:
            # A cacheable system block (Anthropic's per-request cache order is tools ->
            # system -> messages - each level's cache depends on everything before it being
            # unchanged, which holds here since this text is read once at construction and
            # never touched again). This carries the "system research instructions, source
            # hierarchy" half of Phase 8A cost-controls correction item 3 - the
            # `matchup_research_v2.md` per-game context stays in `messages`, uncached, since
            # it's genuinely different every call.
            body["system"] = [{"type": "text", "text": self._system_text, "cache_control": {"type": "ephemeral"}}]
        return {
            "url": self.BASE_URL,
            "headers": {"x-api-key": self._api_key, "anthropic-version": self.API_VERSION, "content-type": "application/json"},
            "json": body,
        }

    def run_research(self, prompt: str, prompt_version: str) -> LLMCallResult:
        return self._call(prompt)

    def run_evaluation(self, prompt: str, prompt_version: str) -> LLMCallResult:
        return self._call(prompt)

    def _post(self, messages: list[dict[str, Any]]) -> dict:
        import urllib.error

        request = self._build_request(messages)
        body = json.dumps(request["json"]).encode("utf-8")
        req = urllib.request.Request(request["url"], data=body, headers=request["headers"], method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # pragma: no cover - never exercised without a live key
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - never exercised without a live key
            detail = e.read().decode("utf-8", errors="replace")[:1000]
            raise AnthropicResponseError(f"Anthropic Messages API returned HTTP {e.code}: {detail}") from e

    def _call(self, prompt: str) -> LLMCallResult:
        """`_post()` is the only method that touches the network (`# pragma: no cover`
        inside it) - this method's pause_turn/tool-extraction logic is directly testable by
        monkeypatching `_post` to return canned payloads, exactly like
        `TheOddsAPIProvider._get()` is monkeypatched in `tests/market/test_odds_provider.py`.

        A `pause_turn` continuation resends the conversation with ONLY the single partial
        assistant turn Anthropic just returned appended, unmodified - never reconstructing,
        summarizing, or duplicating prior search results ourselves (Phase 8A cost-controls
        correction item 4's context-growth audit: this is already the minimal pattern the
        API's own continuation contract requires; nothing extra is resent)."""
        assert_known_cost_floor_within_budget(
            self._model_name, self._pricing_config, self._max_tokens, self._max_web_search_uses,
            self._max_estimated_cost_usd,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_creation_tokens = 0
        total_cache_read_tokens = 0
        total_web_searches = 0

        for _ in range(MAX_PAUSE_TURN_CONTINUATIONS + 1):
            payload = self._post(messages)
            usage = payload.get("usage", {})
            total_input_tokens += usage.get("input_tokens") or 0
            total_output_tokens += usage.get("output_tokens") or 0
            total_cache_creation_tokens += usage.get("cache_creation_input_tokens") or 0
            total_cache_read_tokens += usage.get("cache_read_input_tokens") or 0
            n_web_searches = usage.get("server_tool_use", {}).get("web_search_requests") or 0
            total_web_searches += n_web_searches
            if n_web_searches:
                logger.info("AnthropicMessagesProvider: model made %s web search call(s) this turn", n_web_searches)

            if payload.get("stop_reason") == "pause_turn":
                messages = messages + [{"role": "assistant", "content": payload.get("content", [])}]
                continue

            tool_input = _extract_submit_tool_input(payload)
            estimated_cost = estimate_cost_usd(
                self._model_name, self._pricing_config, input_tokens=total_input_tokens,
                output_tokens=total_output_tokens, cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens, web_search_requests=total_web_searches,
            )
            return LLMCallResult(
                provider_name="anthropic", model_name=self._model_name, raw_output_text=json.dumps(tool_input),
                input_tokens=total_input_tokens, output_tokens=total_output_tokens, estimated_cost_usd=estimated_cost,
                cache_creation_input_tokens=total_cache_creation_tokens, cache_read_input_tokens=total_cache_read_tokens,
                web_search_requests=total_web_searches,
            )

        raise AnthropicResponseError(
            f"Exceeded {MAX_PAUSE_TURN_CONTINUATIONS} pause_turn continuations without a final "
            "response - giving up rather than looping indefinitely."
        )


def _extract_submit_tool_input(payload: dict) -> dict:
    """Locates the `submit_research_findings` tool_use block in a completed (non-pause_turn)
    Messages API response and returns its already-schema-validated `input` dict. Raises
    `AnthropicResponseError` with a compact, secret-free diagnostic - never a partial/garbage
    parse - if the model's final turn didn't include that tool call."""
    content = payload.get("content", [])
    block_types = [b.get("type") for b in content]
    text_blocks = [b for b in content if b.get("type") == "text"]

    for block in content:
        if block.get("type") == "tool_use" and block.get("name") == SUBMIT_TOOL_NAME:
            return block.get("input", {})

    usage = payload.get("usage", {})
    text_preview = (text_blocks[-1].get("text", "") if text_blocks else "")[:200]
    raise AnthropicResponseError(
        f"Anthropic response id={payload.get('id')} never called the {SUBMIT_TOOL_NAME!r} tool - "
        f"stop_reason={payload.get('stop_reason')!r}, content block types={block_types}, "
        f"text_block_count={len(text_blocks)}, usage={usage}, "
        f"last_text_preview={text_preview!r}."
    )


class FixtureLLMProvider(LLMResearchProvider):
    """Offline, deterministic adapter for tests - reads a local JSON fixture keyed by
    prompt_version, makes no network call. Fixture shape: `{"<prompt_version>":
    {"raw_output_text": ..., "input_tokens": ..., "output_tokens": ...}}`."""

    def __init__(self, fixture_path: str | Path, model_name: str = "fixture-model"):
        self._fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        self._model_name = model_name

    def _lookup(self, prompt_version: str) -> LLMCallResult:
        entry = self._fixture.get(prompt_version)
        if entry is None:
            raise KeyError(f"No fixture entry for prompt_version={prompt_version!r}")
        return LLMCallResult(
            provider_name="fixture", model_name=self._model_name, raw_output_text=entry["raw_output_text"],
            input_tokens=entry.get("input_tokens"), output_tokens=entry.get("output_tokens"),
            estimated_cost_usd=entry.get("estimated_cost_usd"),
            cache_creation_input_tokens=entry.get("cache_creation_input_tokens"),
            cache_read_input_tokens=entry.get("cache_read_input_tokens"),
            web_search_requests=entry.get("web_search_requests"),
        )

    def run_research(self, prompt: str, prompt_version: str) -> LLMCallResult:
        return self._lookup(prompt_version)

    def run_evaluation(self, prompt: str, prompt_version: str) -> LLMCallResult:
        return self._lookup(prompt_version)
