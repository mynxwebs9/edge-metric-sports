"""Phase 4 Steps C-F: freeze candidate models, features, metrics, and write the immutable
holdout-freeze manifest - all BEFORE 2024-2025 are ever touched.

`create_and_write_freeze_manifest()` is the last thing that runs before unsealing. Once it
has written the manifest and its sidecar hash file, `holdout_guard.assert_freeze_complete()`
becomes satisfiable and Phase 4's walk-forward code is allowed to proceed. Nothing in this
module reads target outcomes for 2024/2025 - it only records configuration/hyperparameters
decided from 2010-2023 data (Steps A and B) and static definitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nfl_predict.config import PROJECT_ROOT, get_settings
from nfl_predict.features.provenance import compute_feature_registry_config_hash, get_git_revision
from nfl_predict.models.elo import EloConfig
from nfl_predict.models.feature_matrix import ABLATIONS, CORE_FEATURES
from nfl_predict.models.linear_models import RANDOM_SEED as LINEAR_RANDOM_SEED
from nfl_predict.models.boosted import DEFAULT_PARAMS as LGBM_DEFAULT_PARAMS, RANDOM_SEED as LGBM_RANDOM_SEED
from nfl_predict.models.split import DEVELOPMENT_SEASONS, SEALED_HOLDOUT_SEASONS, VALIDATION_SEASON

FREEZE_DIRNAME = "backtests"
FREEZE_FILENAME = "phase4_holdout_freeze.json"
FREEZE_HASH_FILENAME = "phase4_holdout_freeze.sha256"

RIDGE_ALPHA = 5.0
LOGISTIC_C = 1.0

# Candidate models frozen for Phase 4, per the brief - primary six, plus the CORE-vs-FULL
# distinction Phase 3 found worth preserving (F_full underperformed leaner tiers, so both
# the winning E-tier AND the full F-tier ride along for direct comparison, exactly as
# Phase 3 compared them - not a new model, a preserved comparison).
CANDIDATE_MODELS: list[dict] = [
    {
        "model_id": "naive_v1", "model_type": "naive", "targets": ["home_margin", "total_points", "home_win"],
        "feature_set": None, "hyperparameters": {}, "preprocessing": "none (historical constants only)",
        "random_seed": None,
    },
    {
        "model_id": "elo_v2", "model_type": "elo", "targets": ["home_win", "home_margin"],
        "feature_set": None, "hyperparameters": None,  # filled in by freeze_elo_params()
        "preprocessing": "none (sequential rating update)", "random_seed": None,
    },
    {
        "model_id": "ridge_margin_E_v1", "model_type": "ridge", "targets": ["home_margin"],
        "feature_set": "E_add_personnel", "hyperparameters": {"alpha": RIDGE_ALPHA},
        "preprocessing": "median-impute+indicator, standardize (fit on training data only)",
        "random_seed": LINEAR_RANDOM_SEED,
    },
    {
        "model_id": "ridge_total_E_v1", "model_type": "ridge", "targets": ["total_points"],
        "feature_set": "E_add_personnel", "hyperparameters": {"alpha": RIDGE_ALPHA},
        "preprocessing": "median-impute+indicator, standardize (fit on training data only)",
        "random_seed": LINEAR_RANDOM_SEED,
        "calibration": {"correction_applied": False, "reason": "see docs/PHASE4_BACKTEST_REPORT.md#total-model-bias-investigation - bias varies in sign/magnitude by era (not stable), tracks the fit-vs-check period mean-total gap, and is not feature-set-specific; no robust additive/multiplicative correction is justified from development data alone. Left unchanged, documented."},
    },
    {
        "model_id": "logistic_win_E_v1", "model_type": "logistic", "targets": ["home_win"],
        "feature_set": "E_add_personnel", "hyperparameters": {"C": LOGISTIC_C},
        "preprocessing": "median-impute+indicator, standardize (fit on training data only)",
        "random_seed": LINEAR_RANDOM_SEED,
    },
    {
        "model_id": "lightgbm_F_v1", "model_type": "lightgbm", "targets": ["home_margin", "total_points", "home_win"],
        "feature_set": "F_full", "hyperparameters": LGBM_DEFAULT_PARAMS,
        "preprocessing": "none (LightGBM native missing-value handling)",
        "random_seed": LGBM_RANDOM_SEED,
    },
    # CORE-vs-FULL preserved comparison for margin specifically (where the ablation
    # difference was clearest in Phase 3) - not a new model family, the same Ridge recipe
    # run on the CORE feature list instead of E_add_personnel/F_full.
    {
        "model_id": "ridge_margin_CORE_v1", "model_type": "ridge", "targets": ["home_margin"],
        "feature_set": "CORE", "hyperparameters": {"alpha": RIDGE_ALPHA},
        "preprocessing": "median-impute+indicator, standardize (fit on training data only)",
        "random_seed": LINEAR_RANDOM_SEED,
    },
]

METRIC_DEFINITIONS = {
    "margin": ["mae", "rmse", "bias", "corr"],
    "total": ["mae", "rmse", "bias"],
    "win_probability": ["brier", "log_loss", "roc_auc", "accuracy", "calibration_bins"],
    "uncertainty": ["interval_coverage", "residual_std", "residual_mean"],
}

SUBGROUP_DEFINITIONS = [
    "season_2024", "season_2025", "combined_2024_2025",
    "season_type_REG", "season_type_POST",
    "weeks_1_4", "weeks_5_10", "weeks_11_plus",
]


@dataclass(frozen=True)
class FreezeManifestResult:
    manifest_path: Path
    hash_path: Path
    manifest_sha256: str


def _feature_config_hash() -> str | None:
    try:
        return compute_feature_registry_config_hash(PROJECT_ROOT)
    except FileNotFoundError:
        return None


def build_freeze_manifest(elo_config: EloConfig, elo_investigation_summary: dict, bias_investigation_summary: dict) -> dict:
    candidates = []
    for c in CANDIDATE_MODELS:
        entry = dict(c)
        if entry["model_type"] == "elo":
            entry["hyperparameters"] = {
                "k_factor": elo_config.k_factor, "home_field_advantage": elo_config.home_field_advantage,
                "season_regression_fraction": elo_config.season_regression_fraction, "initial_rating": elo_config.initial_rating,
            }
        if entry.get("feature_set"):
            entry["feature_names"] = ABLATIONS.get(entry["feature_set"]) or (CORE_FEATURES if entry["feature_set"] == "CORE" else None)
            entry["n_features"] = len(entry["feature_names"]) if entry["feature_names"] else None
        candidates.append(entry)

    return {
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_seasons_at_freeze_time": SEALED_HOLDOUT_SEASONS,
        "development_seasons": DEVELOPMENT_SEASONS,
        "phase3_validation_season": VALIDATION_SEASON,
        "feature_version": "v1",
        "feature_registry_config_hash": _feature_config_hash(),
        "git_revision": get_git_revision(PROJECT_ROOT),
        "candidate_models": candidates,
        "metric_definitions": METRIC_DEFINITIONS,
        "subgroup_definitions": SUBGROUP_DEFINITIONS,
        "elo_tuning_investigation": elo_investigation_summary,
        "total_bias_investigation": bias_investigation_summary,
    }


def freeze_manifest_paths() -> tuple[Path, Path]:
    directory = get_settings().data_dir / FREEZE_DIRNAME
    return directory / FREEZE_FILENAME, directory / FREEZE_HASH_FILENAME


def write_freeze_manifest(manifest: dict) -> FreezeManifestResult:
    manifest_path, hash_path = freeze_manifest_paths()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    manifest_path.write_text(serialized, encoding="utf-8")

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    hash_path.write_text(digest, encoding="utf-8")

    return FreezeManifestResult(manifest_path=manifest_path, hash_path=hash_path, manifest_sha256=digest)
