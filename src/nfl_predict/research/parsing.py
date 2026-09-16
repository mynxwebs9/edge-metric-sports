"""Phase 6: parses raw LLM output text into `ResearchFindings`.

Never coerces a parse failure into `NO_MATERIAL_NEW_INFORMATION` - `parse_research_output`
raises `ResearchOutputParseError` on malformed JSON, a missing required field, or an
out-of-enum classification/tier/category value; the caller (`run_research.py`) catches this
and stores an explicit `FailedResearchRun` with `FailureStatus.INVALID_JSON` instead.

**Phase 8A cost-controls follow-up - a real TypeError incident:** a real DEN@KC call's
`submit_research_findings` tool input had a bare string somewhere a nested object was
expected (e.g. a raw URL string inside a `sources` array instead of a full
`{source_url, source_title, ...}` object) - the model complying with the tool's declared
`input_schema` is a strong signal, not a hard guarantee (this codebase already documented
that non-strict tool use can't guarantee 100% compliance; see
`nfl_predict.research.llm_provider`'s module docstring). `_parse_source`/`_parse_claim`/
`_parse_external_prediction` did unguarded `d["field"]` subscripting, and their
`except (KeyError, ValueError)` clauses never caught the resulting bare `TypeError:
string indices must be integers, not 'str'` - it propagated all the way out of
`parse_research_output`, past `run_research_for_game`'s specific
`except ResearchOutputParseError`, uncaught. `_require_dict`/`_require_list` below convert
any type mismatch - at every nesting level, not just the top - into the same
`ResearchOutputParseError` every other malformed-output case already produces. This is
explicit rejection, never silent coercion: a bare string is never guessed-converted into a
fake source object.
"""

from __future__ import annotations

import json

from nfl_predict.research.schemas import (
    Claim,
    ClaimCategory,
    ExternalPrediction,
    MaterialityLevel,
    ResearchClassification,
    ResearchFindings,
    SourceRecord,
    SourceTier,
)


class ResearchOutputParseError(Exception):
    """Raised when raw LLM output cannot be parsed into a valid `ResearchFindings` -
    callers must convert this into an explicit `FailedResearchRun`, never a silent
    NO_MATERIAL_NEW_INFORMATION."""


def _require_dict(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise ResearchOutputParseError(f"{context} must be a JSON object, got {type(value).__name__}: {str(value)[:200]!r}")
    return value


def _require_list(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise ResearchOutputParseError(f"{context} must be a JSON array, got {type(value).__name__}: {str(value)[:200]!r}")
    return value


def _parse_source(d: object) -> SourceRecord:
    d = _require_dict(d, "Source record")
    try:
        return SourceRecord(
            source_url=d["source_url"], source_title=d["source_title"], publisher=d["publisher"],
            source_tier=SourceTier(d["source_tier"]), retrieval_timestamp=d["retrieval_timestamp"],
            publication_timestamp=d.get("publication_timestamp"), corroborated_by=tuple(d.get("corroborated_by", [])),
        )
    except (KeyError, ValueError) as e:
        raise ResearchOutputParseError(f"Malformed source record: {e}") from e


def _parse_claim(d: object) -> Claim:
    d = _require_dict(d, "Claim entry")
    try:
        return Claim(
            text=d["text"], category=ClaimCategory(d["category"]),
            sources=tuple(_parse_source(s) for s in _require_list(d.get("sources", []), "Claim 'sources'")),
            materiality_level=MaterialityLevel(d["materiality_level"]), confidence_in_fact=float(d["confidence_in_fact"]),
            reason=d.get("reason", ""),
        )
    except (KeyError, ValueError) as e:
        raise ResearchOutputParseError(f"Malformed claim entry: {e}") from e


def _parse_external_prediction(d: object) -> ExternalPrediction:
    d = _require_dict(d, "External prediction entry")
    try:
        return ExternalPrediction(
            source=d["source"], prediction_text=d["prediction_text"], market=d.get("market"),
            line_at_publication=d.get("line_at_publication"), publication_timestamp=d.get("publication_timestamp"),
            stated_confidence=d.get("stated_confidence"),
        )
    except KeyError as e:
        raise ResearchOutputParseError(f"Malformed external prediction entry: {e}") from e


REQUIRED_TOP_LEVEL_FIELDS = (
    "qb_status", "ol_status", "skill_position_status", "defensive_personnel_status",
    "weather_status", "coaching_status", "research_classification",
)


def parse_research_output(
    raw_text: str, research_id: str, game_id: str, research_timestamp: str, prompt_version: str,
    model_provider: str, model_name: str, input_packet_hash: str,
) -> ResearchFindings:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ResearchOutputParseError(f"LLM output is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ResearchOutputParseError(f"LLM output must be a JSON object, got {type(data).__name__}")

    missing = [k for k in REQUIRED_TOP_LEVEL_FIELDS if k not in data]
    if missing:
        raise ResearchOutputParseError(f"LLM output missing required field(s): {missing}")

    try:
        classification = ResearchClassification(data["research_classification"])
    except ValueError as e:
        raise ResearchOutputParseError(f"Invalid research_classification value: {data['research_classification']!r}") from e

    material_facts = tuple(_parse_claim(c) for c in _require_list(data.get("material_facts", []), "'material_facts'"))
    uncertain_reports = tuple(_parse_claim(c) for c in _require_list(data.get("uncertain_reports", []), "'uncertain_reports'"))
    analyst_opinions = tuple(_parse_claim(c) for c in _require_list(data.get("analyst_opinions", []), "'analyst_opinions'"))
    external_model_opinions = tuple(
        _parse_external_prediction(e) for e in _require_list(data.get("external_model_opinions", []), "'external_model_opinions'")
    )
    raw_missing_information = data.get("missing_information", [])
    if isinstance(raw_missing_information, str):
        # Real, recurring failure mode: the model sometimes writes this one field as a bare
        # string instead of a single-element array, even with an explicit array description
        # in the tool schema (non-strict tool use gives no hard guarantee - see
        # llm_provider.py's module docstring). Wrapping it is safe here specifically because
        # `missing_information` is a plain list of strings - the model's exact text is kept
        # verbatim, nothing is invented. This does NOT apply to the list-of-object fields
        # below (material_facts/uncertain_reports/analyst_opinions), which still reject a
        # bare string outright, since coercing those would require fabricating structure.
        raw_missing_information = [raw_missing_information]
    missing_information_raw = _require_list(raw_missing_information, "'missing_information'")
    if not all(isinstance(x, str) for x in missing_information_raw):
        raise ResearchOutputParseError(f"'missing_information' entries must all be strings, got: {missing_information_raw!r}"[:300])
    missing_information = tuple(missing_information_raw)

    return ResearchFindings(
        research_id=research_id, game_id=game_id, research_timestamp=research_timestamp,
        prompt_version=prompt_version, model_provider=model_provider, model_name=model_name,
        input_packet_hash=input_packet_hash,
        material_facts=material_facts, uncertain_reports=uncertain_reports,
        external_model_opinions=external_model_opinions, analyst_opinions=analyst_opinions,
        qb_status=data["qb_status"], ol_status=data["ol_status"], skill_position_status=data["skill_position_status"],
        defensive_personnel_status=data["defensive_personnel_status"], weather_status=data["weather_status"],
        coaching_status=data["coaching_status"], missing_information=missing_information,
        research_classification=classification,
    )
