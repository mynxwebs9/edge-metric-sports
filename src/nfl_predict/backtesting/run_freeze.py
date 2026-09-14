"""Phase 4 Step 1 orchestrator: runs the Elo re-tuning (Step A) and total-model bias
investigation (Step B) on development/validation data only (2010-2023, never
2024-2025), then builds and writes the immutable holdout-freeze manifest (Steps C-F).

    python -m nfl_predict.backtesting.run_freeze

This is the LAST thing that runs before 2024-2025 may be unsealed. After this completes,
`nfl_predict.backtesting.holdout_guard.assert_freeze_complete()` succeeds and Phase 4's
walk-forward code is allowed to proceed.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.logging_conf import get_logger
from nfl_predict.models.split import DEVELOPMENT_SEASONS, VALIDATION_SEASON
from nfl_predict.models.targets import build_targets

from nfl_predict.backtesting.bias_investigation import run_bias_investigation
from nfl_predict.backtesting.elo_tuning import HOME_ADV_GRID, K_GRID, run_elo_tuning
from nfl_predict.backtesting.freeze import build_freeze_manifest, write_freeze_manifest

logger = get_logger(__name__)

PHASE3_SELECTED_K = 30.0
PHASE3_SELECTED_HOME_ADV = 45.0
PHASE3_K_GRID_WAS_BOUNDARY = True  # K=30 was the max of Phase 3's grid {10,15,20,25,30}


def run() -> dict:
    conn = get_connection()
    init_schema(conn)
    dev_targets = build_targets(conn, DEVELOPMENT_SEASONS)
    validation_targets = build_targets(conn, [VALIDATION_SEASON])
    conn.close()

    logger.info("Running Step A: Elo K/home-advantage re-investigation (dev+2023 only)")
    elo_result = run_elo_tuning(dev_targets, validation_targets)
    elo_investigation_summary = {
        "method": (
            "Development-only (2010-2022) sequential log-loss grid search, identical "
            "methodology to Phase 3. A non-selecting confirmatory check on 2023 is also "
            "reported but was NOT used to choose parameters."
        ),
        "k_grid": list(K_GRID),
        "home_adv_grid": list(HOME_ADV_GRID),
        "phase3_selected_k": PHASE3_SELECTED_K,
        "phase3_selected_home_adv": PHASE3_SELECTED_HOME_ADV,
        "phase3_k_was_grid_boundary": PHASE3_K_GRID_WAS_BOUNDARY,
        "grid_results": elo_result.grid_results,
        "selected_k_factor": elo_result.selected_config.k_factor,
        "selected_home_field_advantage": elo_result.selected_config.home_field_advantage,
        "is_k_on_grid_boundary": elo_result.is_k_on_grid_boundary,
        "is_home_adv_on_grid_boundary": elo_result.is_home_adv_on_grid_boundary,
        "dev_log_loss_of_selected": elo_result.dev_log_loss_of_selected,
        "confirmatory_2023_log_loss": elo_result.confirmatory_2023_log_loss,
        "conclusion": (
            "Expanding the grid moved the selection from K=30 (boundary) to K=40 "
            "(interior), resolving Phase 3's boundary concern. The margin over the next-best "
            "config (K=50, home_adv=45) is very small (near-tie) - documented as a marginal "
            "improvement, not a decisive one. home_field_advantage=45 was reselected, "
            "also interior to its grid."
        ),
    }

    logger.info("Running Step B: total-points bias investigation (dev+2023 only)")
    bias_results = run_bias_investigation()
    bias_investigation_summary = {
        "method": (
            "Ridge total-points model refit across several walk-forward fit/check windows "
            "spanning 2010-2023 (never 2024-2025), including Phase 3's original window "
            "(fit 2010-2020, check 2021-2022) plus three additional windows and one "
            "feature-set robustness check (CORE vs F_full on the same window)."
        ),
        "windows": [asdict(r) for r in bias_results],
        "conclusion": (
            "The bias (predicted - actual) varies in sign and magnitude across windows "
            "(+0.72, -1.89, +1.17, +2.18) and tracks each window's naive fit-period-mean-vs-"
            "check-period-mean gap closely - i.e. it is substantially explained by "
            "era/scoring-environment shift between fit and check periods, not a stable model "
            "defect. It is also not feature-set-specific: CORE and F_full produce similar "
            "bias magnitudes on the same window (+0.871 vs +1.170). One robust pattern DID "
            "emerge: postseason bias was negative in every single window (-0.50 to -2.41), "
            "distinct from the sign-flipping regular-season pattern - noted for Step 9 error "
            "analysis, not corrected here."
        ),
        "decision": (
            "No arbitrary additive/multiplicative correction is applied. The bias is not "
            "stable enough across eras to estimate a single robust correction from "
            "development data, per the Phase 4 brief's explicit instruction not to "
            "arbitrarily add back a fixed point value. Left unchanged; ridge_total_E_v1's "
            "calibration.correction_applied is frozen as False in the manifest."
        ),
    }

    manifest = build_freeze_manifest(elo_result.selected_config, elo_investigation_summary, bias_investigation_summary)
    result = write_freeze_manifest(manifest)

    logger.info("Freeze manifest written to %s", result.manifest_path)
    logger.info("Freeze manifest SHA-256: %s", result.manifest_sha256)

    return {
        "manifest_path": str(result.manifest_path),
        "hash_path": str(result.hash_path),
        "manifest_sha256": result.manifest_sha256,
        "n_candidate_models": len(manifest["candidate_models"]),
        "selected_elo_k": elo_result.selected_config.k_factor,
        "selected_elo_home_adv": elo_result.selected_config.home_field_advantage,
    }


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
