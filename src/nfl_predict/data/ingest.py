"""Ingestion CLI.

    python -m nfl_predict.data.ingest --dataset schedules --seasons 2010-2025
    python -m nfl_predict.data.ingest --dataset pbp --seasons 2023-2025
    python -m nfl_predict.data.ingest --all --seasons 2010-2025
    python -m nfl_predict.data.ingest --dataset teams

For each requested (dataset, season) pair: fetch via nflreadpy, write an immutable raw
snapshot with its provenance manifest, run validation, and - for teams/schedules/pbp, which
have a normalized-table target - promote to normalized storage if no FATAL issue was found.
Every other requested dataset is snapshotted and validated but not promoted to a dedicated
normalized table in Phase 1 (see docs/PHASE1_DATA_REPORT.md for why that's in-scope).

Never fails the whole run because one (dataset, season) pair is unavailable or has
validation issues - it records the outcome and moves on. Exit code is non-zero only if a
requested dataset key doesn't exist or the season spec is unparseable (a usage error), not
because some historical seasons lack optional data.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.ingestion import FetchOutcome, ingest_dataset
from nfl_predict.data.nflverse_loader import ALL_DATASET_SPECS, SOURCE_NAME, DatasetSpec
from nfl_predict.data.promote import promote_games, promote_pbp, promote_teams
from nfl_predict.data.repositories import ManifestsRepository, ValidationRepository
from nfl_predict.data.season_range import DEFAULT_HISTORICAL_SEASONS, parse_season_range
from nfl_predict.data.validation import (
    ValidationIssue,
    is_promotable,
    validate_critical_columns,
    validate_row_count_anomaly,
    validate_schema_change,
)
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)

# Datasets with a normalized-storage target in Phase 1 (see docs/PHASE1_DATA_REPORT.md's
# "Raw vs. normalized" section for why only these three). Every other requested dataset is
# snapshotted and validated but never promoted, so FATAL gating is a no-op for it.
PROMOTABLE_DATASETS = frozenset({"teams", "schedules", "pbp"})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nfl_predict.data.ingest",
        description="Fetch, snapshot, validate, and (where applicable) normalize nflverse data.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", choices=sorted(ALL_DATASET_SPECS), help="Ingest a single dataset.")
    target.add_argument("--all", action="store_true", help="Ingest every known dataset.")
    parser.add_argument(
        "--seasons", default=DEFAULT_HISTORICAL_SEASONS,
        help=f"Season range spec, e.g. '2010-2025' or '2023,2024,2025' (default: {DEFAULT_HISTORICAL_SEASONS}). "
             "Ignored for season-less datasets (currently: teams).",
    )
    return parser


def process_fetch_outcome(
    dataset_key: str,
    spec: DatasetSpec,
    outcome: FetchOutcome,
    conn: sqlite3.Connection,
    manifests_repo: ManifestsRepository,
    validation_repo: ValidationRepository,
) -> list[ValidationIssue]:
    """Validate a successfully-fetched outcome, record the results, and promote it to
    normalized storage IF AND ONLY IF no FATAL issue was found in that pre-promotion
    validation. This is the real orchestration path (the CLI's `run()` just loops over
    outcomes and calls this) - tests exercise it directly rather than only unit-testing
    `is_promotable()` in isolation, so a regression here is actually caught.

    Contract: a FATAL issue in the generic pre-promotion checks (critical columns, schema
    change, row-count anomaly) blocks the dataset-specific promote_* call entirely - it is
    never invoked, so it cannot write to normalized storage no matter what its own internal
    validation would have found. The raw snapshot and every validation issue are still
    recorded either way; only the normalized-storage write is skipped.
    """
    assert outcome.status == "fetched" and outcome.manifest is not None and outcome.dataframe is not None

    manifests_repo.record(outcome.manifest)
    df = outcome.dataframe
    issues = list(validate_critical_columns(dataset_key, df, spec.critical_columns))

    # Compare against the immediately preceding SEASON's LATEST CANONICAL schema
    # specifically - not "whatever manifest happens to already be on disk" (a full backfill
    # run writes every season's raw snapshot before this loop validates any of them, so
    # "most recently written" isn't "chronologically prior"), and not just "any" prior-season
    # manifest either, in case that season was re-fetched after an upstream correction and
    # has more than one canonical (non-duplicate) snapshot - get_latest_canonical_manifest
    # resolves that by `retrieved_at`, read from the database, not file-listing order.
    if outcome.season is not None:
        prior_season_manifest = manifests_repo.get_latest_canonical_manifest(
            SOURCE_NAME, dataset_key, outcome.season - 1
        )
        if prior_season_manifest is not None:
            issues += validate_schema_change(
                dataset_key,
                set(json.loads(prior_season_manifest["column_names"])),
                set(df.columns),
            )

    other_counts = [
        m["row_count"] for m in manifests_repo.by_dataset(SOURCE_NAME, dataset_key)
        if m["retrieval_id"] != outcome.manifest.retrieval_id
        and m["duplicate_of_retrieval_id"] is None
    ]
    issues += validate_row_count_anomaly(dataset_key, outcome.season, df.height, other_counts)
    validation_repo.record(dataset_key, outcome.season, outcome.manifest.retrieval_id, issues)

    if dataset_key not in PROMOTABLE_DATASETS:
        return issues

    if not is_promotable(issues):
        logger.error(
            "promotion blocked: pre-promotion validation found a FATAL issue",
            extra={
                "dataset_name": dataset_key, "season": outcome.season,
                "retrieval_id": outcome.manifest.retrieval_id,
                "fatal_codes": [i.code for i in issues if i.level == "FATAL"],
            },
        )
        print(
            f"[{dataset_key}] season={outcome.season}: PROMOTION BLOCKED - FATAL validation "
            f"issue(s) before promotion was attempted (retrieval_id={outcome.manifest.retrieval_id})"
        )
        return issues

    if dataset_key == "teams":
        issues += promote_teams(df, outcome.manifest.retrieval_id, conn)
    elif dataset_key == "schedules":
        issues += promote_games(df, outcome.manifest.retrieval_id, conn)
    elif dataset_key == "pbp":
        issues += promote_pbp(df, outcome.season, outcome.manifest.retrieval_id, conn)

    return issues


def run(dataset_keys: list[str], seasons: list[int]) -> int:
    conn = get_connection()
    init_schema(conn)
    manifests_repo = ManifestsRepository(conn)
    validation_repo = ValidationRepository(conn)

    total_fatal = 0
    for dataset_key in dataset_keys:
        spec = ALL_DATASET_SPECS[dataset_key]
        outcomes = ingest_dataset(dataset_key, seasons)

        for outcome in outcomes:
            if outcome.status != "fetched":
                print(f"[{dataset_key}] season={outcome.season}: {outcome.status} - {outcome.error_message}")
                continue

            issues = process_fetch_outcome(dataset_key, spec, outcome, conn, manifests_repo, validation_repo)

            fatal = sum(1 for i in issues if i.level == "FATAL")
            error = sum(1 for i in issues if i.level == "ERROR")
            warning = sum(1 for i in issues if i.level == "WARNING")
            total_fatal += fatal
            print(
                f"[{dataset_key}] season={outcome.season}: fetched {outcome.row_count} rows "
                f"(retrieval_id={outcome.manifest.retrieval_id}, "
                f"duplicate={outcome.manifest.duplicate_of_retrieval_id is not None}) "
                f"- issues: FATAL={fatal} ERROR={error} WARNING={warning}"
            )

    conn.close()
    return 1 if total_fatal else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    dataset_keys = sorted(ALL_DATASET_SPECS) if args.all else [args.dataset]
    seasons = parse_season_range(args.seasons)

    return run(dataset_keys, seasons)


if __name__ == "__main__":
    sys.exit(main())
