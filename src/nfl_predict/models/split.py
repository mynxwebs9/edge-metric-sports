"""Temporal data-split policy for Phase 3, and the guard that keeps 2024-2025 sealed.

NO RANDOM SHUFFLING. NFL games are temporal; the split below is fixed and documented, not
tuned. See docs/PHASE3_MODEL_REPORT.md for the rationale if it's ever revisited (revisiting
requires preserving at least two full seasons as a sealed Phase 4 holdout, per the Phase 3
brief - this module raises rather than allowing anything less).
"""

from __future__ import annotations

DEVELOPMENT_SEASONS: list[int] = list(range(2010, 2023))  # 2010-2022 inclusive: fit/train
VALIDATION_SEASON: int = 2023  # Phase 3's own held-out check
SEALED_HOLDOUT_SEASONS: list[int] = [2024, 2025]  # untouched until Phase 4

ALL_PHASE3_VISIBLE_SEASONS: list[int] = DEVELOPMENT_SEASONS + [VALIDATION_SEASON]


class SealedHoldoutError(Exception):
    """Raised when Phase 3 code attempts to read or evaluate against a sealed season."""


def assert_seasons_not_sealed(seasons: list[int]) -> None:
    """Call this at the entry point of any Phase 3 evaluation/reporting function that
    accepts a season list from a caller. Does NOT guard model training data selection
    inside this module's own constants (those are hardcoded safe) - it guards against a
    caller passing 2024/2025 in by mistake (e.g. a typo'd `--seasons 2010-2025` on a Phase 3
    script)."""
    sealed_requested = sorted(set(seasons) & set(SEALED_HOLDOUT_SEASONS))
    if sealed_requested:
        raise SealedHoldoutError(
            f"Refusing to read/evaluate sealed holdout season(s) {sealed_requested} during "
            "Phase 3. 2024-2025 are sealed for Phase 4 - see docs/PHASE3_MODEL_REPORT.md and "
            "src/nfl_predict/models/split.py. This guard exists specifically to prevent "
            "accidental holdout leakage, e.g. via a copy-pasted --seasons 2010-2025."
        )
