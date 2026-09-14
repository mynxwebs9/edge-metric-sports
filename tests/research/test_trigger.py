"""Phase 6 Step 2: the research-trigger mechanism ranks priority, never gates research."""

from __future__ import annotations

from nfl_predict.research.trigger import (
    ALWAYS_RESEARCHABLE,
    TriggerInputs,
    TriggerWeights,
    compute_research_priority,
)


def test_always_researchable_is_true():
    assert ALWAYS_RESEARCHABLE is True


def test_all_missing_inputs_gives_zero_score_and_lists_every_input_unavailable():
    result = compute_research_priority(TriggerInputs())
    assert result.priority_score == 0.0
    assert "model_market_disagreement" in result.inputs_unavailable
    assert "cross_model_disagreement" in result.inputs_unavailable


def test_qb_uncertainty_flag_contributes_its_configured_weight():
    weights = TriggerWeights(qb_uncertainty=2.0)
    result = compute_research_priority(TriggerInputs(qb_uncertainty_flag=True), weights=weights)
    assert result.priority_score == 2.0
    assert any("QB uncertainty" in r for r in result.reasons)


def test_large_model_market_disagreement_increases_score_and_is_reasoned():
    small = compute_research_priority(TriggerInputs(model_market_disagreement_points=0.2))
    large = compute_research_priority(TriggerInputs(model_market_disagreement_points=6.0))
    assert large.priority_score > small.priority_score
    assert any("disagreement" in r for r in large.reasons)
    assert not any("disagreement" in r for r in small.reasons)  # below the 1.0-point reasoning threshold


def test_weights_are_configurable_not_hardcoded():
    default_result = compute_research_priority(TriggerInputs(injury_uncertainty_flag=True))
    custom_result = compute_research_priority(TriggerInputs(injury_uncertainty_flag=True), weights=TriggerWeights(injury_uncertainty=10.0))
    assert custom_result.priority_score > default_result.priority_score


def test_multiple_flags_combine_additively():
    result = compute_research_priority(TriggerInputs(qb_uncertainty_flag=True, offensive_line_issue_flag=True))
    weights = TriggerWeights()
    assert result.priority_score == weights.qb_uncertainty + weights.offensive_line_issue
