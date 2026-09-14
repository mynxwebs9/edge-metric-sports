"""Loads and validates config/features.yaml - the authoritative feature registry.

Two jobs: (1) every column the engine actually produces must have a registry entry, and
vice versa (registry and code must agree - Phase 0's contract, still true), and (2) enforce
the market-data denylist: no feature derived from sportsbook fields may be tagged
`independent`, and no sportsbook-named column may appear in the engine's output at all. See
leakage test #8 in tests/features/test_leakage.py for the automated version of this check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nfl_predict.config import get_features_config

REQUIRED_FIELDS = {
    "name", "category", "description", "source_datasets", "formula", "unit",
    "rolling_window", "min_observations", "pregame_availability", "missing_value_policy",
    "opponent_adjusted", "garbage_time_filtered", "earliest_reliable_season",
    "known_limitations", "leakage_notes", "model_tracks",
}

# Sportsbook/market-derived field names that must NEVER appear as an independent-model
# feature. Matches raw nflverse schedules columns per docs/PHASE1_DATA_REPORT.md's
# "Football data vs. market data" section. Checked both against the registry and, in
# tests, against the actual engine output columns.
MARKET_FIELD_DENYLIST = frozenset({
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "under_odds", "over_odds",
})

# Result/target field names that must never appear in the feature registry or output.
TARGET_FIELD_DENYLIST = frozenset({
    "home_score", "away_score", "winner", "ats_result", "total_result", "margin", "result",
})


@dataclass(frozen=True)
class FeatureRegistryIssue:
    code: str
    message: str


def load_feature_registry() -> list[dict[str, Any]]:
    config = get_features_config()
    return config.get("features", [])


def validate_registry_entries(entries: list[dict[str, Any]]) -> list[FeatureRegistryIssue]:
    issues: list[FeatureRegistryIssue] = []
    seen_names: set[str] = set()

    for entry in entries:
        name = entry.get("name", "<missing name>")
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            issues.append(FeatureRegistryIssue(
                "missing_fields", f"{name}: missing required fields {sorted(missing)}"
            ))
        if name in seen_names:
            issues.append(FeatureRegistryIssue("duplicate_name", f"{name}: duplicate feature name"))
        seen_names.add(name)

        if name in MARKET_FIELD_DENYLIST or name in TARGET_FIELD_DENYLIST:
            issues.append(FeatureRegistryIssue("denylisted_name", f"{name}: denylisted field name in registry"))

        model_tracks = entry.get("model_tracks", [])
        if "independent" in model_tracks:
            source_datasets = entry.get("source_datasets", [])
            if "market" in source_datasets or "sportsbook" in source_datasets:
                issues.append(FeatureRegistryIssue(
                    "market_data_in_independent", f"{name}: tagged independent but sourced from market data"
                ))

    return issues


def registry_names() -> set[str]:
    return {e["name"] for e in load_feature_registry()}


_DENYLIST = MARKET_FIELD_DENYLIST | TARGET_FIELD_DENYLIST


def assert_no_denylisted_columns(columns: list[str]) -> None:
    """Active runtime guard, not just a test: raises loudly if any market-data or
    target/result column name appears in a set of columns about to become part of the
    independent feature matrix. Called from build.py on every feature build, in addition
    to being covered by the automated leakage tests (test_h_* / test_g_* in
    tests/features/test_leakage.py) - a passive test alone doesn't stop a bad build from
    being written to disk if someone runs the CLI without running pytest first.
    """
    found = _DENYLIST & set(columns)
    if found:
        raise ValueError(
            f"Denylisted column(s) found in independent feature output: {sorted(found)}. "
            "Market-derived and target/result fields must never enter the independent "
            "feature matrix - see docs/FEATURE_DICTIONARY.md#market-data-exclusion."
        )
