"""Phase 6 Step 22: cost/usage tracking and a hard cap on research depth, so an automated
research run cannot create uncontrolled cost by repeatedly searching or calling the LLM.

Phase 8A cost-controls correction: `estimate_cost_usd()` is the ONLY place a dollar figure
gets computed anywhere in this codebase - it reads per-model rates from
`config/llm_pricing.yaml` (via `nfl_predict.config.get_llm_pricing_config()`), never a
hard-coded number. `CostTracker` now also records cache-creation/cache-read tokens and
hosted web-search-tool call counts (Anthropic's real, observed usage fields), and
`assert_known_cost_floor_within_budget()` provides a real, pre-flight refusal - computed
from the request's own configured `max_tokens`/`max_web_search_uses` ceilings, never a
speculative guess about how large search-result content might grow - so a game whose
worst-case KNOWN cost already exceeds budget is refused before any HTTP call, and a run
that has already spent up to its per-run game cap simply stops selecting more games (see
`nfl_predict.live.research_live`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ResearchDepthExceededError(Exception):
    """Raised when a research run would exceed its configured max search-query count or
    max total source count."""


class ResearchCostBudgetExceededError(Exception):
    """Raised BEFORE any HTTP call is made, when that call's known-floor worst-case cost
    (output tokens at `max_tokens` + `max_web_search_uses` searches, both real configured
    ceilings - excluding the unpredictable cost of search-result input tokens) would already
    exceed `max_estimated_cost_usd`. A real, computable refusal, not a guess."""


@dataclass(frozen=True)
class ResearchCostConfig:
    max_search_queries: int = 6
    max_source_count: int = 20
    max_estimated_cost_usd: float | None = None


@dataclass
class CostTracker:
    """One tracker per research run (not shared across games) - `record_*` methods raise
    immediately once a configured limit would be exceeded, rather than letting a run
    silently keep going over budget."""

    config: ResearchCostConfig = field(default_factory=ResearchCostConfig)
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_web_search_requests: int = 0
    total_estimated_cost_usd: float = 0.0
    total_search_queries: int = 0
    total_sources_consulted: int = 0

    def record_llm_call(
        self, input_tokens: int | None, output_tokens: int | None, estimated_cost_usd: float | None,
        cache_creation_input_tokens: int | None = None, cache_read_input_tokens: int | None = None,
        web_search_requests: int | None = None,
    ) -> None:
        self.total_llm_calls += 1
        self.total_input_tokens += input_tokens or 0
        self.total_output_tokens += output_tokens or 0
        self.total_cache_creation_input_tokens += cache_creation_input_tokens or 0
        self.total_cache_read_input_tokens += cache_read_input_tokens or 0
        self.total_web_search_requests += web_search_requests or 0
        self.total_estimated_cost_usd += estimated_cost_usd or 0.0

    def record_search_query(self, n_sources_returned: int) -> None:
        self.total_search_queries += 1
        self.total_sources_consulted += n_sources_returned
        if self.total_search_queries > self.config.max_search_queries:
            raise ResearchDepthExceededError(
                f"Research run exceeded max_search_queries={self.config.max_search_queries} "
                f"(attempted query #{self.total_search_queries})."
            )
        if self.total_sources_consulted > self.config.max_source_count:
            raise ResearchDepthExceededError(
                f"Research run exceeded max_source_count={self.config.max_source_count} "
                f"(consulted {self.total_sources_consulted} sources)."
            )

    def summary(self) -> dict:
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_creation_input_tokens": self.total_cache_creation_input_tokens,
            "total_cache_read_input_tokens": self.total_cache_read_input_tokens,
            "total_web_search_requests": self.total_web_search_requests,
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "total_search_queries": self.total_search_queries,
            "total_sources_consulted": self.total_sources_consulted,
        }


def estimate_cost_usd(
    model_name: str,
    pricing_config: dict,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    web_search_requests: int | None = None,
) -> float | None:
    """The one place a dollar figure gets computed. Returns `None` (never a fabricated
    number) if `model_name` has no entry in `pricing_config` - an unpriced model should
    show up as "unknown cost," not a wrong one. `cache_write_5m_per_mtok` is used for
    `cache_creation_input_tokens` since `AnthropicMessagesProvider` only ever requests the
    default 5-minute cache TTL (see its `_build_request`)."""
    model_pricing = (pricing_config or {}).get("models", {}).get(model_name)
    if model_pricing is None:
        return None
    return (
        (input_tokens or 0) / 1_000_000 * model_pricing["base_input_per_mtok"]
        + (output_tokens or 0) / 1_000_000 * model_pricing["output_per_mtok"]
        + (cache_creation_input_tokens or 0) / 1_000_000 * model_pricing["cache_write_5m_per_mtok"]
        + (cache_read_input_tokens or 0) / 1_000_000 * model_pricing["cache_read_per_mtok"]
        + (web_search_requests or 0) * model_pricing["web_search_per_call"]
    )


def known_cost_floor_usd(
    model_name: str, pricing_config: dict, max_tokens: int, max_web_search_uses: int,
) -> float | None:
    """A REAL, computable lower bound on what a call could cost - output tokens at the
    request's own `max_tokens` ceiling, plus `max_web_search_uses` searches at their flat
    per-search rate. Deliberately excludes input-token cost, since how much a web search
    result adds to input is not predictable in advance without guessing - this is a floor,
    not a full worst-case estimate, but it is REAL (derived from the request's own
    configured limits) rather than invented. Returns `None` if `model_name` is unpriced."""
    model_pricing = (pricing_config or {}).get("models", {}).get(model_name)
    if model_pricing is None:
        return None
    return (
        max_tokens / 1_000_000 * model_pricing["output_per_mtok"]
        + max_web_search_uses * model_pricing["web_search_per_call"]
    )


def assert_known_cost_floor_within_budget(
    model_name: str, pricing_config: dict, max_tokens: int, max_web_search_uses: int,
    max_estimated_cost_usd: float | None,
) -> None:
    """Raises `ResearchCostBudgetExceededError` BEFORE any HTTP call if the request's own
    known-floor cost already exceeds the configured per-game budget. A `None` budget means
    no cap is configured - never refuses. An unpriced model never refuses here either (no
    known floor to compare), but `estimate_cost_usd` will still return `None` for its actual
    cost afterward, which callers should treat as "unknown," not "free.\""""
    if max_estimated_cost_usd is None:
        return
    floor = known_cost_floor_usd(model_name, pricing_config, max_tokens, max_web_search_uses)
    if floor is not None and floor > max_estimated_cost_usd:
        raise ResearchCostBudgetExceededError(
            f"Refusing to call {model_name!r}: the known-floor cost of this request "
            f"(max_tokens={max_tokens} output + {max_web_search_uses} web searches) is "
            f"already ${floor:.4f}, exceeding the configured "
            f"NFL_RESEARCH_MAX_ESTIMATED_COST_PER_GAME=${max_estimated_cost_usd:.4f}. "
            "No HTTP request was made."
        )
