"""Phase 6 Step 24 proof #12: API secrets are not stored in research artifacts (checked
here at the provider level); provider abstraction + credential gating.

Phase 8A correction: the INVALID_JSON incident (see docs/PHASE8A_LIVE_PIPELINE_REPORT.md)
came from concatenating free-text `text` blocks and running `json.loads()` on them. These
tests prove the fix - a forced-schema `submit_research_findings` tool call, `pause_turn`
continuation, and a clear diagnostic error (never a bare JSONDecodeError) when the model
never calls the tool - entirely via a mocked `_post`, no live network access."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predict.research.llm_provider import (
    SUBMIT_TOOL_NAME,
    AnthropicMessagesProvider,
    AnthropicResponseError,
    FixtureLLMProvider,
    LLMProviderNotConfiguredError,
    LLMResearchProvider,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "llm_fixture.json"

_VALID_FINDINGS = {
    "material_facts": [], "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
    "qb_status": "No change reported.", "ol_status": "No change reported.", "skill_position_status": "No change reported.",
    "defensive_personnel_status": "No change reported.", "weather_status": "Not checked.", "coaching_status": "No change reported.",
    "missing_information": [], "research_classification": "NO_MATERIAL_NEW_INFORMATION",
}


def _final_payload(tool_input=_VALID_FINDINGS, response_id="msg_1", stop_reason="tool_use", extra_content=()):
    content = list(extra_content) + [{"type": "tool_use", "id": "tool_1", "name": SUBMIT_TOOL_NAME, "input": tool_input}]
    return {"id": response_id, "stop_reason": stop_reason, "content": content, "usage": {"input_tokens": 100, "output_tokens": 50}}


def test_anthropic_provider_refuses_to_construct_without_a_key(monkeypatch):
    monkeypatch.delenv("NFL_RESEARCH_LLM_API_KEY", raising=False)
    from nfl_predict.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(LLMProviderNotConfiguredError):
        AnthropicMessagesProvider()
    get_settings.cache_clear()


def test_anthropic_provider_constructs_with_an_explicit_key():
    provider = AnthropicMessagesProvider(api_key="test-key-not-a-real-secret")
    assert isinstance(provider, LLMResearchProvider)


def test_anthropic_provider_request_never_embeds_the_key_in_the_body():
    provider = AnthropicMessagesProvider(api_key="super-secret-value")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    assert "super-secret-value" not in str(request["json"])
    assert request["headers"]["x-api-key"] == "super-secret-value"  # key belongs only in the header, never the body


def test_fixture_provider_returns_deterministic_output_without_network_access():
    provider = FixtureLLMProvider(FIXTURE_PATH)
    result = provider.run_research("any prompt text", prompt_version="matchup_research_v1")
    assert result.provider_name == "fixture"
    assert "NO_MATERIAL_NEW_INFORMATION" in result.raw_output_text
    assert result.input_tokens == 1200
    assert result.output_tokens == 150


def test_fixture_provider_raises_on_an_unknown_prompt_version():
    provider = FixtureLLMProvider(FIXTURE_PATH)
    with pytest.raises(KeyError):
        provider.run_research("any prompt text", prompt_version="nonexistent_version")


def test_anthropic_provider_request_includes_web_search_tool_by_default():
    """Phase 8A correction Step 3: a plain Messages API call cannot browse the web, and
    matchup_research.md's prompt demands real, current, cited sources - so the request must
    include Anthropic's server-side web-search tool by default."""
    provider = AnthropicMessagesProvider(api_key="test-key-not-a-real-secret")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    tool_types = [t.get("type") for t in request["json"]["tools"]]
    assert "web_search_20250305" in tool_types
    web_search_tool = next(t for t in request["json"]["tools"] if t.get("type") == "web_search_20250305")
    assert web_search_tool["max_uses"] == 3  # Phase 8A correction Step B7's conservative default


def test_anthropic_provider_tool_choice_is_auto_not_forced():
    """Forcing tool_choice to submit_research_findings from turn 1 would require the model
    to call it immediately, before it could search at all - auto lets the model search
    first and submit as its final action."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    assert request["json"]["tool_choice"] == {"type": "auto"}


def test_anthropic_provider_always_includes_the_submit_findings_tool():
    provider = AnthropicMessagesProvider(api_key="test-key")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    tool_names = [t.get("name") for t in request["json"]["tools"]]
    assert SUBMIT_TOOL_NAME in tool_names


def test_submit_tool_schema_lists_every_field_as_required_but_is_not_strict():
    """Regression test tracking two REAL, live-observed findings in sequence:
    (1) plain (non-strict) tool use once returned a tool call missing the required
    `research_classification` field, so every property is listed in `required` (nullable
    types stand in for genuinely-optional fields) with `additionalProperties: false` to
    guide generation as strongly as possible without a hard guarantee; (2) enabling
    Anthropic's `strict: true` on this schema was then rejected with a real 400
    ("The compiled grammar is too large... Simplify your tool schemas") once the nested
    claim/source/external-prediction arrays were included - so `strict` is deliberately
    NOT set, see the `_tools()` comment for the full story."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    submit_tool = next(t for t in request["json"]["tools"] if t.get("name") == SUBMIT_TOOL_NAME)
    assert "strict" not in submit_tool

    schema = submit_tool["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    assert "research_classification" in schema["required"]

    # Nested object schemas (claims, sources, external predictions) must be strict too.
    claim_schema = schema["properties"]["material_facts"]["items"]
    assert claim_schema["additionalProperties"] is False
    assert set(claim_schema["required"]) == set(claim_schema["properties"].keys())

    # Regression: a real 400 response confirmed Anthropic's strict schema validator
    # rejects `minimum`/`maximum` on number types - the range is still enforced downstream
    # by nfl_predict.research.schemas.Claim.__post_init__.
    confidence_schema = claim_schema["properties"]["confidence_in_fact"]
    assert "minimum" not in confidence_schema
    assert "maximum" not in confidence_schema


def test_anthropic_provider_web_search_can_be_disabled_and_max_uses_is_configurable():
    provider = AnthropicMessagesProvider(api_key="test-key", enable_web_search=False)
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    tool_types = [t.get("type") for t in request["json"]["tools"]]
    assert "web_search_20250305" not in tool_types
    assert SUBMIT_TOOL_NAME in [t.get("name") for t in request["json"]["tools"]]  # still present even without search

    capped_provider = AnthropicMessagesProvider(api_key="test-key", max_web_search_uses=2)
    request2 = capped_provider._build_request([{"role": "user", "content": "some prompt"}])
    web_search_tool = next(t for t in request2["json"]["tools"] if t.get("type") == "web_search_20250305")
    assert web_search_tool["max_uses"] == 2


def test_anthropic_provider_extracts_structured_input_from_the_submit_tool_call(monkeypatch):
    """Test proving the actual INVALID_JSON fix: the final result comes from the tool's
    ALREADY-PARSED `input` dict, never from concatenating/parsing free text."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    monkeypatch.setattr(provider, "_post", lambda messages: _final_payload())
    result = provider._call("some prompt")
    assert result.provider_name == "anthropic"
    import json
    assert json.loads(result.raw_output_text) == _VALID_FINDINGS


def test_anthropic_provider_ignores_narration_text_blocks_before_the_tool_call(monkeypatch):
    """The exact incident shape: a narration text block ('Let me research...') appears
    alongside the tool_use block - it must never be concatenated into the parsed result."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    payload = _final_payload(extra_content=[{"type": "text", "text": "Let me research this matchup..."}])
    monkeypatch.setattr(provider, "_post", lambda messages: payload)
    result = provider._call("some prompt")
    import json
    assert json.loads(result.raw_output_text) == _VALID_FINDINGS  # narration text never leaked into the JSON


def test_anthropic_provider_handles_pause_turn_by_continuing(monkeypatch):
    """The hosted web-search tool can return stop_reason=pause_turn mid-research (a
    documented long-running-tool-use signal) - the provider must resend the conversation
    with the partial turn appended, not treat the empty/partial response as final."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    calls = []

    def fake_post(messages):
        calls.append(len(messages))
        if len(calls) == 1:
            return {"id": "msg_1", "stop_reason": "pause_turn", "content": [{"type": "server_tool_use", "id": "t1"}], "usage": {"input_tokens": 10, "output_tokens": 5}}
        return _final_payload(response_id="msg_2")

    monkeypatch.setattr(provider, "_post", fake_post)
    result = provider._call("some prompt")
    assert len(calls) == 2  # one pause_turn, then one final call
    assert calls[1] == 2  # the second call's messages list grew (original + the paused assistant turn appended)
    import json
    assert json.loads(result.raw_output_text) == _VALID_FINDINGS


def test_anthropic_provider_raises_a_diagnostic_error_when_the_tool_is_never_called(monkeypatch):
    """This is the actual bug being fixed: if the model answers in free text instead of
    calling submit_research_findings, this must raise a clear, secret-free diagnostic -
    never attempt json.loads() on the text and crash with a bare JSONDecodeError."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    payload = {
        "id": "msg_bad", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "I looked into this and found nothing notable."}],
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }
    monkeypatch.setattr(provider, "_post", lambda messages: payload)
    with pytest.raises(AnthropicResponseError) as exc_info:
        provider._call("some prompt")
    message = str(exc_info.value)
    assert "msg_bad" in message
    assert "end_turn" in message
    assert SUBMIT_TOOL_NAME in message


def test_anthropic_provider_surfaces_the_real_http_error_body_on_failure(monkeypatch):
    """Regression test for a real observed gap: an HTTP 400 from Anthropic (e.g. a rejected
    request shape) must surface its actual response body, not just a bare
    'HTTP Error 400: Bad Request' with no diagnostic content."""
    import io
    import urllib.error

    provider = AnthropicMessagesProvider(api_key="test-key")

    def fake_urlopen(req, timeout=120):
        raise urllib.error.HTTPError(url="http://x", code=400, msg="Bad Request", hdrs={}, fp=io.BytesIO(b'{"error": {"message": "tools.1.strict: Extra inputs are not permitted"}}'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(AnthropicResponseError, match="strict"):
        provider._call("some prompt")


def test_anthropic_provider_gives_up_after_too_many_pause_turn_continuations(monkeypatch):
    provider = AnthropicMessagesProvider(api_key="test-key")
    monkeypatch.setattr(provider, "_post", lambda messages: {"id": "msg_x", "stop_reason": "pause_turn", "content": [], "usage": {}})
    with pytest.raises(AnthropicResponseError, match="pause_turn"):
        provider._call("some prompt")


def test_fixture_provider_records_model_and_provider_name():
    provider = FixtureLLMProvider(FIXTURE_PATH, model_name="test-model-x")
    result = provider.run_evaluation("prompt", prompt_version="matchup_research_v1")
    assert result.model_name == "test-model-x"
    assert result.provider_name == "fixture"


# --- Phase 8A cost-controls correction: prompt caching + real usage/cost recording ---

_PRICING = {
    "models": {
        "claude-sonnet-5": {
            "base_input_per_mtok": 2.00, "cache_write_5m_per_mtok": 2.50, "cache_write_1h_per_mtok": 4.00,
            "cache_read_per_mtok": 0.20, "output_per_mtok": 10.00, "web_search_per_call": 0.01,
        },
    },
}


def test_static_system_instructions_are_loaded_and_cache_controlled_by_default():
    provider = AnthropicMessagesProvider(api_key="test-key")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    assert "system" in request["json"]
    system_blocks = request["json"]["system"]
    assert system_blocks[-1]["cache_control"] == {"type": "ephemeral"}
    # Static instructions, never game-specific placeholders.
    assert "{{" not in system_blocks[-1]["text"]
    assert "source hierarchy" in system_blocks[-1]["text"].lower() or "Source hierarchy" in system_blocks[-1]["text"]


def test_system_prompt_can_be_disabled():
    provider = AnthropicMessagesProvider(api_key="test-key", system_prompt_path=None)
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    assert "system" not in request["json"]


def test_tools_array_is_cache_controlled_on_its_last_entry():
    """cache_control on the LAST tool caches the WHOLE tools array (web_search def +
    submit_research_findings schema) - both are byte-identical on every call."""
    provider = AnthropicMessagesProvider(api_key="test-key")
    request = provider._build_request([{"role": "user", "content": "some prompt"}])
    tools = request["json"]["tools"]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_call_captures_real_cache_and_web_search_usage_and_computes_cost(monkeypatch):
    """Regression test for the actual usage/cost recording this correction adds - proves
    cache_creation_input_tokens, cache_read_input_tokens, and web_search_requests all reach
    the returned LLMCallResult, and that cost is computed via the injected pricing config
    (never hard-coded)."""
    provider = AnthropicMessagesProvider(api_key="test-key", pricing_config=_PRICING)
    payload = _final_payload()
    payload["usage"] = {
        "input_tokens": 1000, "output_tokens": 500,
        "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 10000,
        "server_tool_use": {"web_search_requests": 3},
    }
    monkeypatch.setattr(provider, "_post", lambda messages: payload)

    result = provider._call("some prompt")

    assert result.input_tokens == 1000
    assert result.output_tokens == 500
    assert result.cache_creation_input_tokens == 2000
    assert result.cache_read_input_tokens == 10000
    assert result.web_search_requests == 3
    expected_cost = (1000 / 1e6 * 2.00) + (500 / 1e6 * 10.00) + (2000 / 1e6 * 2.50) + (10000 / 1e6 * 0.20) + (3 * 0.01)
    assert result.estimated_cost_usd == pytest.approx(expected_cost)


def test_call_uses_the_configured_pricing_table_not_a_hard_coded_rate(monkeypatch):
    """Swap in a deliberately different pricing table and confirm the computed cost changes
    accordingly - proving the rate comes from configuration, not a literal in the code."""
    custom_pricing = {"models": {"claude-sonnet-5": {**_PRICING["models"]["claude-sonnet-5"], "output_per_mtok": 999.0}}}
    provider = AnthropicMessagesProvider(api_key="test-key", pricing_config=custom_pricing, max_estimated_cost_usd=None)
    payload = _final_payload()
    payload["usage"] = {"input_tokens": 0, "output_tokens": 1_000_000}
    monkeypatch.setattr(provider, "_post", lambda messages: payload)

    result = provider._call("some prompt")
    assert result.estimated_cost_usd == pytest.approx(999.0)


def test_call_returns_none_cost_for_a_model_not_in_the_pricing_table(monkeypatch):
    provider = AnthropicMessagesProvider(api_key="test-key", model_name="some-unpriced-model", pricing_config=_PRICING)
    monkeypatch.setattr(provider, "_post", lambda messages: _final_payload())
    result = provider._call("some prompt")
    assert result.estimated_cost_usd is None


def test_pause_turn_continuation_accumulates_usage_across_turns_without_resending_extra_content(monkeypatch):
    """Item 4's context-growth audit: the continuation appends ONLY the single prior
    assistant turn, unmodified - never reconstructing or duplicating search content - and
    usage from EVERY turn (not just the last) is accumulated into the final result."""
    provider = AnthropicMessagesProvider(api_key="test-key", pricing_config=_PRICING)
    calls = []

    def fake_post(messages):
        calls.append([dict(m) for m in messages])
        if len(calls) == 1:
            return {
                "id": "msg_1", "stop_reason": "pause_turn",
                "content": [{"type": "server_tool_use", "id": "t1"}],
                "usage": {"input_tokens": 500, "output_tokens": 100, "server_tool_use": {"web_search_requests": 1}},
            }
        payload = _final_payload(response_id="msg_2")
        payload["usage"] = {"input_tokens": 300, "output_tokens": 200, "server_tool_use": {"web_search_requests": 1}}
        return payload

    monkeypatch.setattr(provider, "_post", fake_post)
    result = provider._call("some prompt")

    assert result.input_tokens == 800  # 500 + 300, accumulated across both turns
    assert result.output_tokens == 300
    assert result.web_search_requests == 2
    # The continuation's message list is exactly [original user turn, the one partial
    # assistant turn] - nothing duplicated or reconstructed.
    assert len(calls[1]) == 2
    assert calls[1][0]["role"] == "user"
    assert calls[1][1] == {"role": "assistant", "content": [{"type": "server_tool_use", "id": "t1"}]}
