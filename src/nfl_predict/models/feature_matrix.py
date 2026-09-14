"""Builds model-ready feature matrices from the Phase 2 game-level feature table, joined
with Phase 3 targets.

Every feature used here is `diff_<name>` (home minus away) or, for the two symmetric
boolean context flags, `home_<name>` — and every `<name>` is a real entry in
`config/features.yaml`. Sample-size/support features (`games_played_current_season`,
`n_games_trailing_*`, etc.) and the categorical `qb_primary_id` are deliberately excluded
from the model input set - they're metadata, not football performance signals, per
docs/FEATURE_DICTIONARY.md.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import polars as pl

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.features.registry import assert_no_denylisted_columns, registry_names
from nfl_predict.features.store import read_feature_table
from nfl_predict.models.split import assert_seasons_not_sealed
from nfl_predict.models.targets import read_targets

FEATURE_VERSION = "v1"

# Non-diffable, symmetric boolean context (home == away by construction, so a diff would
# always be 0) - included once, undiffed, since they're still real pregame context.
SYMMETRIC_CONTEXT_FEATURES = ["home_is_neutral_site", "home_is_divisional_game"]

# --- Ablation ladder (A -> F), each a strict superset of the previous tier. ---

_BASIC_CONTEXT = [
    "diff_days_rest", "diff_opponent_days_rest", "diff_rest_diff",
    "diff_short_week", "diff_post_bye",
] + SYMMETRIC_CONTEXT_FEATURES

_OFF_DEF_EFFICIENCY = [
    "diff_off_epa_pp_season", "diff_off_success_rate_season",
    "diff_off_pass_epa_dropback_season", "diff_off_pass_success_rate_season",
    "diff_off_rush_epa_season", "diff_off_rush_success_rate_season",
    "diff_def_epa_pp_allowed_season", "diff_def_success_rate_allowed_season",
    "diff_def_pass_epa_allowed_season", "diff_def_pass_success_rate_allowed_season",
    "diff_def_rush_epa_allowed_season", "diff_def_rush_success_rate_allowed_season",
]

_QB_FEATURES = [
    "diff_qb_epa_dropback_season", "diff_qb_success_rate_season",
    "diff_qb_sack_rate_season", "diff_qb_int_rate_season", "diff_qb_cpoe_season",
    "diff_qb_scramble_epa_season", "diff_qb_consecutive_starts",
]

_RECENT_WINDOWS = [
    f"diff_{base}_{w}g"
    for base in ("off_epa_pp", "off_success_rate", "off_pass_epa_dropback", "off_pass_success_rate",
                  "off_rush_epa", "off_rush_success_rate", "def_epa_pp_allowed", "def_success_rate_allowed",
                  "def_pass_epa_allowed", "def_pass_success_rate_allowed", "def_rush_epa_allowed",
                  "def_rush_success_rate_allowed")
    for w in (3, 5, 8)
]

_PERSONNEL = ["diff_off_snap_continuity_pct", "diff_def_snap_continuity_pct"]

_REMAINING_FOR_FULL = [
    "diff_off_epa_pp_comp_3g", "diff_off_epa_pp_comp_5g", "diff_off_epa_pp_comp_8g", "diff_off_epa_pp_comp_season",
    "diff_off_success_rate_comp_3g", "diff_off_success_rate_comp_5g", "diff_off_success_rate_comp_8g", "diff_off_success_rate_comp_season",
    "diff_off_explosive_pass_rate_season", "diff_off_explosive_rush_rate_season",
    "diff_def_explosive_pass_rate_allowed_season", "diff_def_explosive_rush_rate_allowed_season",
    "diff_off_sack_rate_allowed_season", "diff_off_qb_hit_rate_allowed_season",
    "diff_def_sack_rate_generated_season", "diff_def_qb_hit_rate_generated_season",
    "diff_off_early_down_epa_season", "diff_off_early_down_success_rate_season",
    "diff_def_early_down_epa_allowed_season", "diff_def_early_down_success_rate_allowed_season",
    "diff_off_third_down_conv_rate_season", "diff_def_third_down_allowed_season",
    "diff_off_redzone_td_rate_season", "diff_def_redzone_td_rate_allowed_season",
    "diff_off_yards_per_dropback_season", "diff_off_yards_per_rush_season",
    "diff_off_int_rate_season", "diff_off_fumble_lost_rate_season",
    "diff_def_int_rate_generated_season", "diff_def_fumble_forced_rate_season",
    "diff_off_cpoe_season",
    "diff_opp_off_epa_faced_season", "diff_opp_def_epa_faced_season",
    "diff_fg_pct_season", "diff_punt_yards_avg_season",
    "diff_prev_season_off_epa_pp", "diff_prev_season_off_success_rate",
    "diff_prev_season_def_epa_pp_allowed", "diff_prev_season_def_success_rate_allowed",
]

ABLATIONS: dict[str, list[str]] = {
    "A_basic_context": list(_BASIC_CONTEXT),
    "B_off_def_efficiency": list(_BASIC_CONTEXT) + _OFF_DEF_EFFICIENCY,
    "C_add_qb": list(_BASIC_CONTEXT) + _OFF_DEF_EFFICIENCY + _QB_FEATURES,
    "D_add_recent_windows": list(_BASIC_CONTEXT) + _OFF_DEF_EFFICIENCY + _QB_FEATURES + _RECENT_WINDOWS,
    "E_add_personnel": list(_BASIC_CONTEXT) + _OFF_DEF_EFFICIENCY + _QB_FEATURES + _RECENT_WINDOWS + _PERSONNEL,
    "F_full": list(_BASIC_CONTEXT) + _OFF_DEF_EFFICIENCY + _QB_FEATURES + _RECENT_WINDOWS + _PERSONNEL + _REMAINING_FOR_FULL,
}

# CORE: broad-historical-availability-only subset of F (excludes personnel [2013+] and
# previous-season [2011+]) - see docs/PHASE3_MODEL_REPORT.md's "Missingness and historical
# coverage" section.
CORE_FEATURES = [f for f in ABLATIONS["F_full"] if f not in _PERSONNEL and not f.startswith("diff_prev_season_")]


def _underlying_registry_name(feature_col: str) -> str:
    if feature_col.startswith("diff_"):
        return feature_col[len("diff_"):]
    if feature_col.startswith("home_"):
        return feature_col[len("home_"):]
    raise ValueError(f"Unrecognized feature column naming: {feature_col}")


def validate_ablation_features_are_registered() -> None:
    """Every column name used anywhere in the ablation ladder must correspond to a real
    config/features.yaml entry - guards against a typo silently using a nonexistent (null)
    column, and against ever accidentally referencing a non-registered field."""
    names = registry_names()
    all_cols = {c for cols in ABLATIONS.values() for c in cols}
    for col in all_cols:
        underlying = _underlying_registry_name(col)
        if underlying not in names:
            raise ValueError(f"Feature column {col!r} does not correspond to a registered feature ({underlying!r} not in config/features.yaml)")


validate_ablation_features_are_registered()


@dataclass(frozen=True)
class ModelDataset:
    seasons: list[int]
    frame: pl.DataFrame  # game_id, season, week, season_type, targets, and every feature column present in the source table


def _load_game_dataset_impl(seasons: list[int], conn: sqlite3.Connection | None = None) -> ModelDataset:
    """The actual loading logic, with NO seal check - never call this directly outside of
    (a) `load_game_dataset` below, which checks the seal first, or (b) Phase 4's
    freeze-gated holdout loader (`nfl_predict.backtesting.holdout_guard`), which requires
    the holdout freeze to be verified complete before it will import/call this at all. This
    split exists so Phase 3's guarantee (this module never reads 2024-2025) stays intact
    and unit-testable in isolation from Phase 4's separate, explicit unsealing mechanism."""
    owns_conn = conn is None
    conn = conn or get_connection()
    init_schema(conn)
    try:
        targets = read_targets(seasons)
    finally:
        if owns_conn:
            conn.close()

    features = read_feature_table("game", FEATURE_VERSION, seasons)
    assert_no_denylisted_columns(features.columns)

    joined = targets.join(
        features, on="game_id", how="inner", suffix="_feat"
    )
    return ModelDataset(seasons=seasons, frame=joined)


def load_game_dataset(seasons: list[int], conn: sqlite3.Connection | None = None) -> ModelDataset:
    """Loads the Phase 2 game-level feature table for `seasons`, joins Phase 3 targets, and
    runs the market/target denylist guard on the resulting column set. `seasons` is
    explicitly validated against the sealed-holdout guard on every call - this is the single
    choke point through which all Phase 3 model code reads game-level data."""
    assert_seasons_not_sealed(seasons)
    return _load_game_dataset_impl(seasons, conn)


def without_ties(dataset: ModelDataset) -> ModelDataset:
    """For win-probability modeling: a tie is neither a win nor a loss for either team, so
    it cannot be a classification label. Excluded here explicitly (13 games in
    2010-2025, all regular season) rather than coerced into 0 or 1."""
    return ModelDataset(seasons=dataset.seasons, frame=dataset.frame.filter(~pl.col("is_tie")))


# Team/game identifiers must never become numeric model inputs - Phase 2 team_id values
# ("0200", "0325", ...) are strings by design, but a careless int() cast or a future
# refactor could turn them into a spurious numeric "signal". Checked by name, defensively,
# on every feature selection - not just relied on by convention.
IDENTIFIER_COLUMN_NAMES = frozenset({
    "game_id", "team_id", "opponent_id", "home_team_id", "away_team_id",
})


def assert_no_identifier_columns(columns: list[str]) -> None:
    found = IDENTIFIER_COLUMN_NAMES & set(columns)
    if found:
        raise ValueError(
            f"Identifier column(s) requested as model feature(s): {sorted(found)}. "
            "team_id/opponent_id/game_id must never be used as numeric predictive "
            "variables - see docs/PHASE3_MODEL_REPORT.md#team-identifiers."
        )


def select_features(dataset: ModelDataset, feature_columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Returns (X, columns_used) — X is the raw (possibly-null) feature slice; imputation/
    scaling happens later inside each model's own fitted Pipeline, never here."""
    assert_no_identifier_columns(feature_columns)
    missing = [c for c in feature_columns if c not in dataset.frame.columns]
    if missing:
        raise ValueError(f"Requested feature column(s) not present in dataset: {missing}")
    selected = dataset.frame.select(feature_columns)
    non_numeric = [c for c, dt in zip(selected.columns, selected.dtypes) if dt == pl.Utf8]
    if non_numeric:
        raise ValueError(f"Non-numeric column(s) requested as model feature(s): {non_numeric}")
    return selected, feature_columns
