"""Phase 8A cost-controls correction: cost estimation reads ONLY config/llm_pricing.yaml
(never a hard-coded number), and the pre-flight budget guard refuses a request whose own
known-floor cost already exceeds the configured per-game cap - before any HTTP call."""

from __future__ import annotations

import pytest

from nfl_predict.research.cost_tracking import (
    CostTracker,
    ResearchCostBudgetExceededError,
    ResearchCostConfig,
    ResearchDepthExceededError,
    assert_known_cost_floor_within_budget,
    estimate_cost_usd,
    known_cost_floor_usd,
)

PRICING = {
    "models": {
        "claude-sonnet-5": {
            "base_input_per_mtok": 2.00,
            "cache_write_5m_per_mtok": 2.50,
            "cache_write_1h_per_mtok": 4.00,
            "cache_read_per_mtok": 0.20,
            "output_per_mtok": 10.00,
            "web_search_per_call": 0.01,
        },
    },
}


def test_estimate_cost_usd_matches_the_real_observed_single_game_call():
    """Regression check against Correction 2's real, successful single-game validation
    call: 69,942 input tokens, 7,586 output tokens, 3 web searches, no caching."""
    cost = estimate_cost_usd(
        "claude-sonnet-5", PRICING, input_tokens=69942, output_tokens=7586, web_search_requests=3,
    )
    expected = 69942 / 1_000_000 * 2.00 + 7586 / 1_000_000 * 10.00 + 3 * 0.01
    assert cost == pytest.approx(expected)
    assert cost == pytest.approx(0.245744, abs=1e-6)


def test_estimate_cost_usd_includes_cache_creation_and_cache_read():
    cost = estimate_cost_usd(
        "claude-sonnet-5", PRICING, input_tokens=1000, output_tokens=500,
        cache_creation_input_tokens=2000, cache_read_input_tokens=10000, web_search_requests=1,
    )
    expected = (1000 / 1e6 * 2.00) + (500 / 1e6 * 10.00) + (2000 / 1e6 * 2.50) + (10000 / 1e6 * 0.20) + (1 * 0.01)
    assert cost == pytest.approx(expected)


def test_estimate_cost_usd_returns_none_for_an_unpriced_model_never_fabricates_a_number():
    cost = estimate_cost_usd("some-future-model-not-in-the-table", PRICING, input_tokens=1000, output_tokens=500)
    assert cost is None


def test_known_cost_floor_is_computed_from_the_requests_own_configured_ceilings():
    floor = known_cost_floor_usd("claude-sonnet-5", PRICING, max_tokens=16000, max_web_search_uses=3)
    expected = 16000 / 1e6 * 10.00 + 3 * 0.01
    assert floor == pytest.approx(expected)


def test_budget_guard_refuses_before_any_call_when_the_floor_exceeds_the_cap():
    with pytest.raises(ResearchCostBudgetExceededError, match="No HTTP request was made"):
        assert_known_cost_floor_within_budget(
            "claude-sonnet-5", PRICING, max_tokens=16000, max_web_search_uses=3, max_estimated_cost_usd=0.01,
        )


def test_budget_guard_allows_a_request_within_budget():
    assert_known_cost_floor_within_budget(
        "claude-sonnet-5", PRICING, max_tokens=16000, max_web_search_uses=3, max_estimated_cost_usd=1.00,
    )  # does not raise


def test_budget_guard_with_no_configured_cap_never_refuses():
    assert_known_cost_floor_within_budget(
        "claude-sonnet-5", PRICING, max_tokens=999999999, max_web_search_uses=999, max_estimated_cost_usd=None,
    )  # does not raise even for an absurd ceiling


def test_budget_guard_never_refuses_an_unpriced_model_since_there_is_no_known_floor():
    assert_known_cost_floor_within_budget(
        "unpriced-model", PRICING, max_tokens=16000, max_web_search_uses=3, max_estimated_cost_usd=0.001,
    )  # does not raise - nothing to compare


def test_cost_tracker_records_cache_and_web_search_usage():
    tracker = CostTracker()
    tracker.record_llm_call(
        input_tokens=100, output_tokens=50, estimated_cost_usd=0.01,
        cache_creation_input_tokens=200, cache_read_input_tokens=300, web_search_requests=2,
    )
    summary = tracker.summary()
    assert summary["total_cache_creation_input_tokens"] == 200
    assert summary["total_cache_read_input_tokens"] == 300
    assert summary["total_web_search_requests"] == 2


def test_cost_tracker_accumulates_across_multiple_calls():
    tracker = CostTracker()
    tracker.record_llm_call(input_tokens=100, output_tokens=50, estimated_cost_usd=0.01, web_search_requests=1)
    tracker.record_llm_call(input_tokens=200, output_tokens=60, estimated_cost_usd=0.02, web_search_requests=2)
    summary = tracker.summary()
    assert summary["total_input_tokens"] == 300
    assert summary["total_web_search_requests"] == 3
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.03)


def test_search_depth_guard_still_works_unchanged():
    tracker = CostTracker(config=ResearchCostConfig(max_search_queries=1))
    tracker.record_search_query(n_sources_returned=1)
    with pytest.raises(ResearchDepthExceededError):
        tracker.record_search_query(n_sources_returned=1)
