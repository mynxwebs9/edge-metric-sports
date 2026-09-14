"""Phase 6 Step 25: the provider actually used for this phase's real prospective pilot.

This is explicitly NOT an automated API call. `run_research`'s output text is pre-authored
JSON, produced by a human/agent (this Claude Code session, using its own real WebSearch/
WebFetch tools against the live web) reasoning about real, current sources for a real game -
then wrapped in the same `LLMCallResult` shape every other provider returns, so it flows
through the identical orchestration/parsing/evaluation/storage code path an automated
provider would use. `provider_name` is always recorded as `"manual-claude-code-session"`,
never disguised as `"anthropic"` or any name that would suggest an automated
`AnthropicMessagesProvider` call actually fired - see
`docs/PHASE6_RESEARCH_AGENT_REPORT.md`'s pilot section for why this path was used instead of
a live-configured `AnthropicMessagesProvider`.
"""

from __future__ import annotations

from nfl_predict.research.llm_provider import LLMCallResult, LLMResearchProvider

PROVIDER_NAME = "manual-claude-code-session"


class ManualResearchProvider(LLMResearchProvider):
    def __init__(self, research_output_text: str, model_name: str = "claude-sonnet-5"):
        self._research_output_text = research_output_text
        self._model_name = model_name

    def run_research(self, prompt: str, prompt_version: str) -> LLMCallResult:
        return LLMCallResult(
            provider_name=PROVIDER_NAME, model_name=self._model_name, raw_output_text=self._research_output_text,
            input_tokens=None, output_tokens=None, estimated_cost_usd=None,
        )

    def run_evaluation(self, prompt: str, prompt_version: str) -> LLMCallResult:
        raise NotImplementedError("Evaluation is deterministic code (see nfl_predict.research.evaluator), not an LLM call - this provider never needs run_evaluation.")
