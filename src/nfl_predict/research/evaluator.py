"""Phase 6 Step 15: the research evaluator - a second, independent step that never trusts
the research agent's own first-pass classification blindly. Verifies internal consistency,
downgrades low-quality/uncorroborated sources, deduplicates evidence, assesses materiality,
and re-derives the final `ResearchClassification` with deterministic rules layered on top of
whatever the raw findings reported. Never generates a betting pick.
"""

from __future__ import annotations

from datetime import datetime, timezone

from nfl_predict.research.schemas import (
    Claim,
    ClaimCategory,
    EvaluationResult,
    MaterialityLevel,
    ResearchClassification,
    ResearchFindings,
    SourceTier,
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate_claims(claims: tuple[Claim, ...]) -> tuple[tuple[Claim, ...], tuple[tuple[str, ...], ...]]:
    """A conservative, explainable near-duplicate rule (normalized-text equality or
    substring containment) - not fuzzy NLP, since over-merging distinct claims would hide
    real independent corroboration (Step 12's "do not scrape a large number of syndicated
    copies and count them independently" applies here too, to claims generally, not just
    external predictions). Returns (one representative claim per group, groups of duplicate
    claim texts - only groups with more than one member)."""
    groups: list[list[Claim]] = []
    for claim in claims:
        placed = False
        for group in groups:
            if _normalize(claim.text) == _normalize(group[0].text) or _normalize(claim.text) in _normalize(group[0].text) or _normalize(group[0].text) in _normalize(claim.text):
                group.append(claim)
                placed = True
                break
        if not placed:
            groups.append([claim])
    representatives = tuple(group[0] for group in groups)
    duplicate_groups = tuple(tuple(c.text for c in group) for group in groups if len(group) > 1)
    return representatives, duplicate_groups


def downgrade_unsupported_claims(claims: tuple[Claim, ...]) -> tuple[tuple[Claim, ...], tuple[str, ...]]:
    """Step 5's explicit rule, enforced here rather than trusted from the raw LLM output: a
    VERIFIED_FACT claim sourced ONLY from Tier 4 (community) sources, with no independent
    corroboration recorded, is downgraded to REPORTED_NOT_CONFIRMED."""
    downgraded_texts = []
    result = []
    for claim in claims:
        if claim.category == ClaimCategory.VERIFIED_FACT:
            tier4_only = all(s.source_tier == SourceTier.TIER_4_COMMUNITY for s in claim.sources)
            corroborated = any(s.corroborated_by for s in claim.sources)
            if tier4_only and not corroborated:
                result.append(Claim(
                    text=claim.text, category=ClaimCategory.REPORTED_NOT_CONFIRMED, sources=claim.sources,
                    materiality_level=claim.materiality_level, confidence_in_fact=min(claim.confidence_in_fact, 0.5),
                    reason=claim.reason + " [downgraded by evaluator: Tier 4 source only, uncorroborated]",
                ))
                downgraded_texts.append(claim.text)
                continue
        result.append(claim)
    return tuple(result), tuple(downgraded_texts)


def find_unsupported_claims(claims: tuple[Claim, ...]) -> tuple[str, ...]:
    """Claims of ANY category with zero attached sources - flagged, never silently
    dropped. (`Claim.__post_init__` already refuses to construct a sourceless
    VERIFIED_FACT, so in practice this only ever catches OPINION/REPORTED-category claims
    the raw LLM output tried to assert with no citation at all.)"""
    return tuple(c.text for c in claims if not c.sources)


def compute_overall_materiality(claims: tuple[Claim, ...]) -> MaterialityLevel:
    if not claims:
        return MaterialityLevel.NOISE
    return max((c.materiality_level for c in claims), default=MaterialityLevel.NOISE)


def determine_final_classification(findings: ResearchFindings, overall_materiality: MaterialityLevel) -> ResearchClassification:
    """Deterministic re-derivation, never a blind pass-through of the LLM's self-reported
    classification. CRITICAL materiality always wins (forces VETO_CONSIDERATION,
    regardless of what the raw findings claimed); an empty findings set always collapses to
    NO_MATERIAL_NEW_INFORMATION even if the raw output claimed otherwise. Otherwise the
    reported classification stands - the evaluator's job is to catch cases where the
    materiality assessment CONTRADICTS the reported label, not to second-guess every
    plausible one."""
    if overall_materiality == MaterialityLevel.CRITICAL:
        return ResearchClassification.VETO_CONSIDERATION
    if not findings.material_facts and not findings.uncertain_reports:
        return ResearchClassification.NO_MATERIAL_NEW_INFORMATION
    return findings.research_classification


def evaluate_research(
    findings: ResearchFindings, evaluation_id: str, model_provider: str, model_name: str,
    prompt_version: str, now: str | None = None,
) -> EvaluationResult:
    all_claims = findings.material_facts + findings.uncertain_reports + findings.analyst_opinions
    downgraded_facts, downgraded_texts = downgrade_unsupported_claims(findings.material_facts)
    _, duplicate_groups = deduplicate_claims(all_claims)
    unsupported = find_unsupported_claims(all_claims)
    overall_materiality = compute_overall_materiality(downgraded_facts + findings.uncertain_reports)
    final_classification = determine_final_classification(findings, overall_materiality)

    return EvaluationResult(
        evaluation_id=evaluation_id, research_id=findings.research_id, game_id=findings.game_id,
        evaluation_timestamp=now or datetime.now(timezone.utc).isoformat(),
        prompt_version=prompt_version, model_provider=model_provider, model_name=model_name,
        final_classification=final_classification, overall_materiality=overall_materiality,
        downgraded_claim_texts=downgraded_texts, duplicate_claim_groups=duplicate_groups,
        unsupported_claim_texts=unsupported,
        notes=(
            f"{len(downgraded_texts)} claim(s) downgraded (uncorroborated Tier 4), "
            f"{len(duplicate_groups)} duplicate group(s) found, "
            f"{len(unsupported)} unsupported claim(s) flagged."
        ),
    )
