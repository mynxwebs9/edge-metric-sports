"""Phase 6 Step 24 proofs #6-8, #15: source tiers, fact/opinion separation, duplicate
detection, and classification-enum enforcement, exercised through the evaluator."""

from __future__ import annotations

from nfl_predict.research.evaluator import (
    compute_overall_materiality,
    deduplicate_claims,
    determine_final_classification,
    downgrade_unsupported_claims,
    evaluate_research,
    find_unsupported_claims,
)
from nfl_predict.research.schemas import (
    Claim,
    ClaimCategory,
    MaterialityLevel,
    ResearchClassification,
    ResearchFindings,
    SourceRecord,
    SourceTier,
)


def _source(tier: SourceTier, corroborated_by=()) -> SourceRecord:
    return SourceRecord(
        source_url="https://example.com/a", source_title="Example", publisher="Example Publisher",
        source_tier=tier, retrieval_timestamp="2026-09-11T10:00:00+00:00", publication_timestamp="2026-09-10T18:00:00+00:00",
        corroborated_by=corroborated_by,
    )


def _findings(material_facts=(), uncertain_reports=(), analyst_opinions=(), classification=ResearchClassification.NO_MATERIAL_NEW_INFORMATION) -> ResearchFindings:
    return ResearchFindings(
        research_id="r1", game_id="g1", research_timestamp="2026-09-11T10:00:00+00:00",
        prompt_version="matchup_research_v1", model_provider="fixture", model_name="fixture-model",
        input_packet_hash="deadbeef", material_facts=material_facts, uncertain_reports=uncertain_reports,
        external_model_opinions=(), analyst_opinions=analyst_opinions,
        qb_status="n/a", ol_status="n/a", skill_position_status="n/a", defensive_personnel_status="n/a",
        weather_status="n/a", coaching_status="n/a", missing_information=(), research_classification=classification,
    )


def test_verified_fact_requires_at_least_one_source():
    import pytest

    with pytest.raises(ValueError, match="VERIFIED_FACT"):
        Claim(text="Unsupported claim", category=ClaimCategory.VERIFIED_FACT, sources=(), materiality_level=MaterialityLevel.MODERATE, confidence_in_fact=0.9, reason="none")


def test_downgrade_unsupported_claims_downgrades_uncorroborated_tier4_fact():
    claim = Claim(
        text="Starting LT ruled out", category=ClaimCategory.VERIFIED_FACT,
        sources=(_source(SourceTier.TIER_4_COMMUNITY),), materiality_level=MaterialityLevel.MAJOR,
        confidence_in_fact=0.9, reason="reddit post",
    )
    result, downgraded = downgrade_unsupported_claims((claim,))
    assert downgraded == ("Starting LT ruled out",)
    assert result[0].category == ClaimCategory.REPORTED_NOT_CONFIRMED


def test_downgrade_unsupported_claims_leaves_tier1_facts_alone():
    claim = Claim(
        text="Starting LT ruled out", category=ClaimCategory.VERIFIED_FACT,
        sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.MAJOR,
        confidence_in_fact=0.95, reason="official injury report",
    )
    result, downgraded = downgrade_unsupported_claims((claim,))
    assert downgraded == ()
    assert result[0].category == ClaimCategory.VERIFIED_FACT


def test_downgrade_unsupported_claims_leaves_corroborated_tier4_facts_alone():
    claim = Claim(
        text="Starting LT ruled out", category=ClaimCategory.VERIFIED_FACT,
        sources=(_source(SourceTier.TIER_4_COMMUNITY, corroborated_by=("https://official.example.com",)),),
        materiality_level=MaterialityLevel.MAJOR, confidence_in_fact=0.9, reason="reddit, corroborated",
    )
    result, downgraded = downgrade_unsupported_claims((claim,))
    assert downgraded == ()


def test_deduplicate_claims_groups_near_identical_text():
    a = Claim(text="Starting LT ruled out for Sunday", category=ClaimCategory.VERIFIED_FACT, sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.MAJOR, confidence_in_fact=0.9, reason="x")
    b = Claim(text="starting lt ruled out for sunday", category=ClaimCategory.REPORTED_NOT_CONFIRMED, sources=(_source(SourceTier.TIER_2_REPUTABLE_REPORTER),), materiality_level=MaterialityLevel.MAJOR, confidence_in_fact=0.7, reason="y")
    c = Claim(text="Weather expected to be clear", category=ClaimCategory.VERIFIED_FACT, sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.NOISE, confidence_in_fact=0.9, reason="z")

    representatives, duplicate_groups = deduplicate_claims((a, b, c))
    assert len(representatives) == 2
    assert len(duplicate_groups) == 1
    assert set(duplicate_groups[0]) == {a.text, b.text}


def test_find_unsupported_claims_flags_sourceless_opinions():
    opinion = Claim(text="Analyst X thinks this matters", category=ClaimCategory.ANALYST_OPINION, sources=(), materiality_level=MaterialityLevel.MINOR, confidence_in_fact=0.5, reason="x")
    assert find_unsupported_claims((opinion,)) == (opinion.text,)


def test_compute_overall_materiality_takes_the_max():
    low = Claim(text="a", category=ClaimCategory.VERIFIED_FACT, sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.MINOR, confidence_in_fact=0.9, reason="x")
    high = Claim(text="b", category=ClaimCategory.VERIFIED_FACT, sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.CRITICAL, confidence_in_fact=0.9, reason="y")
    assert compute_overall_materiality((low, high)) == MaterialityLevel.CRITICAL
    assert compute_overall_materiality(()) == MaterialityLevel.NOISE


def test_critical_materiality_forces_veto_consideration_regardless_of_reported_classification():
    fact = Claim(text="Starting QB status genuinely uncertain", category=ClaimCategory.VERIFIED_FACT, sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.CRITICAL, confidence_in_fact=0.9, reason="x")
    findings = _findings(material_facts=(fact,), classification=ResearchClassification.SUPPORTS_MODEL)
    assert determine_final_classification(findings, MaterialityLevel.CRITICAL) == ResearchClassification.VETO_CONSIDERATION


def test_no_facts_or_reports_collapses_to_no_material_new_information():
    findings = _findings(classification=ResearchClassification.SUPPORTS_MARKET)  # raw output disagrees with reality
    assert determine_final_classification(findings, MaterialityLevel.NOISE) == ResearchClassification.NO_MATERIAL_NEW_INFORMATION


def test_evaluate_research_returns_a_classification_from_the_allowed_enum():
    fact = Claim(text="Starting LT ruled out", category=ClaimCategory.VERIFIED_FACT, sources=(_source(SourceTier.TIER_1_OFFICIAL),), materiality_level=MaterialityLevel.MAJOR, confidence_in_fact=0.9, reason="x")
    findings = _findings(material_facts=(fact,), classification=ResearchClassification.MIXED)
    result = evaluate_research(findings, evaluation_id="e1", model_provider="fixture", model_name="fixture-model", prompt_version="research_evaluator_v1")
    assert isinstance(result.final_classification, ResearchClassification)
    assert result.final_classification in list(ResearchClassification)
