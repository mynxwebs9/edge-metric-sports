"""Phase 6 Step 24 proofs #13, #15, #17: invalid LLM output fails safely, classifications
are limited to the allowed enum, and output hashes are reproducible."""

from __future__ import annotations

import hashlib
import json

import pytest

from nfl_predict.research.parsing import ResearchOutputParseError, parse_research_output
from nfl_predict.research.schemas import ResearchClassification

VALID_OUTPUT = json.dumps({
    "material_facts": [], "uncertain_reports": [], "external_model_opinions": [], "analyst_opinions": [],
    "qb_status": "Starter confirmed healthy.", "ol_status": "No change.", "skill_position_status": "No change.",
    "defensive_personnel_status": "No change.", "weather_status": "Not checked.", "coaching_status": "No change.",
    "missing_information": [], "research_classification": "NO_MATERIAL_NEW_INFORMATION",
})


def test_valid_output_parses_successfully():
    findings = parse_research_output(VALID_OUTPUT, "r1", "g1", "2026-09-11T10:00:00Z", "v1", "fixture", "fixture-model", "hash123")
    assert findings.research_classification == ResearchClassification.NO_MATERIAL_NEW_INFORMATION
    assert findings.qb_status == "Starter confirmed healthy."


def test_non_json_text_raises_parse_error():
    with pytest.raises(ResearchOutputParseError, match="not valid JSON"):
        parse_research_output("this is not json {{{", "r1", "g1", "ts", "v1", "p", "m", "h")


def test_json_array_instead_of_object_raises_parse_error():
    with pytest.raises(ResearchOutputParseError, match="must be a JSON object"):
        parse_research_output("[1, 2, 3]", "r1", "g1", "ts", "v1", "p", "m", "h")


def test_missing_required_field_raises_parse_error():
    incomplete = json.dumps({"qb_status": "ok"})
    with pytest.raises(ResearchOutputParseError, match="missing required field"):
        parse_research_output(incomplete, "r1", "g1", "ts", "v1", "p", "m", "h")


def test_invalid_classification_value_raises_parse_error():
    bad = json.loads(VALID_OUTPUT)
    bad["research_classification"] = "TOTALLY_MADE_UP_CLASSIFICATION"
    with pytest.raises(ResearchOutputParseError, match="Invalid research_classification"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_invalid_source_tier_in_a_claim_raises_parse_error():
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = [{
        "text": "x", "category": "VERIFIED_FACT", "materiality_level": 2, "confidence_in_fact": 0.9, "reason": "y",
        "sources": [{"source_url": "u", "source_title": "t", "publisher": "p", "source_tier": "NOT_A_REAL_TIER", "retrieval_timestamp": "ts"}],
    }]
    with pytest.raises(ResearchOutputParseError):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


# --- Regression tests for a real production incident: a real DEN@KC Anthropic call put a
# bare string somewhere a nested object was expected (e.g. a raw URL in `sources` instead of
# a full {source_url, source_title, ...} object). The unguarded d["field"] subscripting in
# _parse_source/_parse_claim/_parse_external_prediction raised a bare
# `TypeError: string indices must be integers, not 'str'` - NOT caught by their own
# `except (KeyError, ValueError)`, so it propagated uncaught past run_research_for_game
# entirely. Every case below must now raise the SAME ResearchOutputParseError every other
# malformed-output case raises - never a TypeError, and never silently coerced into a fake
# well-formed record.


def test_a_bare_string_in_a_claims_sources_list_raises_parse_error_not_typeerror():
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = [{
        "text": "x", "category": "VERIFIED_FACT", "materiality_level": 2, "confidence_in_fact": 0.9, "reason": "y",
        "sources": ["https://example.com/some-article"],  # a bare URL string, not a source object - the real incident shape
    }]
    with pytest.raises(ResearchOutputParseError, match="Source record must be a JSON object"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_a_bare_string_claim_in_material_facts_raises_parse_error_not_typeerror():
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = ["Mahomes is questionable"]  # a bare string, not a claim object
    with pytest.raises(ResearchOutputParseError, match="Claim entry must be a JSON object"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_material_facts_as_a_non_list_raises_parse_error_not_typeerror():
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = "no material facts found"  # a string where an array was expected
    with pytest.raises(ResearchOutputParseError, match="'material_facts' must be a JSON array"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_external_model_opinions_with_a_non_dict_entry_raises_parse_error_not_typeerror():
    bad = json.loads(VALID_OUTPUT)
    bad["external_model_opinions"] = [{"source": "x", "prediction_text": "y"}, "some analyst's tweet"]
    with pytest.raises(ResearchOutputParseError, match="External prediction entry must be a JSON object"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_missing_information_with_a_non_string_entry_raises_parse_error():
    bad = json.loads(VALID_OUTPUT)
    bad["missing_information"] = [123, "a real gap"]
    with pytest.raises(ResearchOutputParseError, match="missing_information' entries must all be strings"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_missing_information_as_a_bare_string_is_wrapped_not_rejected():
    """Real, recurring model mistake: writing this one plain-string-list field as a bare
    string instead of a single-element array. Safe to wrap because the content - the
    model's exact text - is preserved verbatim, nothing is invented."""
    bad = json.loads(VALID_OUTPUT)
    bad["missing_information"] = "Final injury designations were not yet available."
    findings = parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")
    assert findings.missing_information == ("Final injury designations were not yet available.",)


def test_material_facts_as_a_bare_string_still_raises_parse_error_not_wrapped():
    """Unlike `missing_information`, this field holds structured Claim objects - wrapping a
    bare string would require fabricating fields (category, materiality_level, ...), so it
    must still be rejected outright, never silently coerced."""
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = "Lions' starting QB is questionable."
    with pytest.raises(ResearchOutputParseError, match="'material_facts' must be a JSON array"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_a_claims_sources_field_as_a_non_list_raises_parse_error_not_typeerror():
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = [{
        "text": "x", "category": "VERIFIED_FACT", "materiality_level": 2, "confidence_in_fact": 0.9, "reason": "y",
        "sources": "https://example.com/some-article",  # a bare string where an array was expected
    }]
    with pytest.raises(ResearchOutputParseError, match="Claim 'sources' must be a JSON array"):
        parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")


def test_a_well_formed_source_list_still_parses_correctly_after_the_type_guard_fix():
    """The fix must not break the legitimate, well-formed case."""
    bad = json.loads(VALID_OUTPUT)
    bad["material_facts"] = [{
        "text": "x", "category": "VERIFIED_FACT", "materiality_level": 2, "confidence_in_fact": 0.9, "reason": "y",
        "sources": [{"source_url": "https://example.com/a", "source_title": "t", "publisher": "p", "source_tier": "TIER_1_OFFICIAL", "retrieval_timestamp": "ts", "publication_timestamp": None}],
    }]
    findings = parse_research_output(json.dumps(bad), "r1", "g1", "ts", "v1", "p", "m", "h")
    assert len(findings.material_facts) == 1
    assert findings.material_facts[0].sources[0].source_url == "https://example.com/a"


def test_parsing_the_same_output_twice_produces_the_same_findings_hash():
    findings_a = parse_research_output(VALID_OUTPUT, "r1", "g1", "ts", "v1", "p", "m", "h")
    findings_b = parse_research_output(VALID_OUTPUT, "r1", "g1", "ts", "v1", "p", "m", "h")

    import dataclasses

    hash_a = hashlib.sha256(json.dumps(dataclasses.asdict(findings_a), sort_keys=True, default=str).encode("utf-8")).hexdigest()
    hash_b = hashlib.sha256(json.dumps(dataclasses.asdict(findings_b), sort_keys=True, default=str).encode("utf-8")).hexdigest()
    assert hash_a == hash_b
