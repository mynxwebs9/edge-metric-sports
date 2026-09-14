"""Phase 1 data coverage report.

Produces:
    data/reports/phase1_coverage.csv   (machine-readable, one row per dataset x season)
    docs/PHASE1_DATA_REPORT.md         (human-readable summary, written separately by hand
                                         from this data - see the Phase 1 completion report)

Run as: python -m nfl_predict.data.coverage --seasons 2010-2025

## `fetched` vs `usable`

These are deliberately two different columns, not one ambiguous `available` flag:

- **fetched** = an upstream snapshot was successfully retrieved and preserved (a manifest
  exists for this dataset/season). Says nothing about whether the data is any good.
- **usable** = fetched, AND it passes the FATAL validation gate, AND it has at least one row.

The third condition is not implicit in "FATAL == 0" - a genuinely empty (0-row) snapshot
(e.g. snap_counts season=2012, which nflreadpy accepts as an in-range request but returns no
rows for) has zero validation issues today, because no validator currently treats "zero rows"
as a data-quality defect in itself. But a snapshot with no rows provides no analytical
observations, so calling it "usable" would be misleading regardless of its validation
status - this is a deliberate policy decision, not a validation bug, and it's enforced here
rather than by inventing a new FATAL check for "the data happens to be empty" (an empty
result can be entirely correct - the source just has nothing for that season).

So: out-of-range seasons -> fetched=False, usable=False. A season with a FATAL validation
issue (e.g. depth_charts 2025's missing season/week columns) -> fetched=True, usable=False.
A genuine zero-row snapshot (e.g. snap_counts 2012) -> fetched=True, usable=False. Everything
else that was actually fetched and has real rows and no FATAL issue -> fetched=True,
usable=True. See docs/PHASE1_DATA_REPORT.md for the terminology used consistently there.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings
from nfl_predict.data.db import get_connection, init_schema
from nfl_predict.data.nflverse_loader import ALL_DATASET_SPECS, SOURCE_NAME
from nfl_predict.data.repositories import ManifestsRepository, ValidationRepository
from nfl_predict.data.season_range import DEFAULT_HISTORICAL_SEASONS, parse_season_range

REPORT_COLUMNS = [
    "dataset_name", "season", "fetched", "usable", "row_count", "games_represented",
    "missing_critical_fields", "validation_fatal", "validation_error", "validation_warning",
    "validation_info", "retrieval_id", "schema_fingerprint", "duplicate_of_retrieval_id",
]


def _games_represented(raw_file_path: str, columns: list[str]) -> int | None:
    if "game_id" not in columns:
        return None
    try:
        return (
            pl.scan_parquet(raw_file_path)
            .select(pl.col("game_id").n_unique().alias("n"))
            .collect()
            .item()
        )
    except Exception:
        return None


def generate_coverage_rows(seasons: list[int]) -> list[dict]:
    conn = get_connection()
    init_schema(conn)
    manifests_repo = ManifestsRepository(conn)
    validation_repo = ValidationRepository(conn)

    rows: list[dict] = []
    for dataset_name, spec in sorted(ALL_DATASET_SPECS.items()):
        all_manifests = manifests_repo.by_dataset(SOURCE_NAME, dataset_name)
        by_season: dict[int | None, list] = {}
        for m in all_manifests:
            requested = json.loads(m["requested_seasons"])
            key = requested[0] if len(requested) == 1 else None
            by_season.setdefault(key, []).append(m)

        season_keys = [None] if not spec.takes_season else seasons
        for season in season_keys:
            manifests_for_season = by_season.get(season, [])
            canonical = next((m for m in reversed(manifests_for_season) if m["duplicate_of_retrieval_id"] is None), None)
            latest = manifests_for_season[-1] if manifests_for_season else None

            issues = validation_repo.by_dataset(dataset_name)
            issues_for_season = [i for i in issues if i["season"] == season] if spec.takes_season else issues
            level_counts = {"FATAL": 0, "ERROR": 0, "WARNING": 0, "INFO": 0}
            for i in issues_for_season:
                level_counts[i["level"]] = level_counts.get(i["level"], 0) + 1
            missing_critical = any(i["code"] == "missing_critical_columns" for i in issues_for_season)

            if latest is None:
                rows.append({
                    "dataset_name": dataset_name, "season": season, "fetched": False, "usable": False,
                    "row_count": None, "games_represented": None,
                    "missing_critical_fields": None, "validation_fatal": 0, "validation_error": 0,
                    "validation_warning": 0, "validation_info": 0, "retrieval_id": None,
                    "schema_fingerprint": None, "duplicate_of_retrieval_id": None,
                })
                continue

            columns = list(json.loads(latest["column_names"]))
            games_represented = None
            if canonical is not None:
                games_represented = _games_represented(canonical["raw_file_path"], columns)

            # See the module docstring's "fetched vs usable" section: usable requires both
            # a clean FATAL gate AND at least one row - a genuinely empty snapshot is fetched
            # but not usable, even though it has zero validation issues today.
            usable = level_counts["FATAL"] == 0 and (latest["row_count"] or 0) > 0

            rows.append({
                "dataset_name": dataset_name,
                "season": season,
                "fetched": True,
                "usable": usable,
                "row_count": latest["row_count"],
                "games_represented": games_represented,
                "missing_critical_fields": missing_critical,
                "validation_fatal": level_counts["FATAL"],
                "validation_error": level_counts["ERROR"],
                "validation_warning": level_counts["WARNING"],
                "validation_info": level_counts["INFO"],
                "retrieval_id": latest["retrieval_id"],
                "schema_fingerprint": latest["schema_fingerprint"],
                "duplicate_of_retrieval_id": latest["duplicate_of_retrieval_id"],
            })

    conn.close()
    return rows


def write_coverage_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_report_path() -> Path:
    return get_settings().data_dir / "reports" / "phase1_coverage.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nfl_predict.data.coverage")
    parser.add_argument("--seasons", default=DEFAULT_HISTORICAL_SEASONS)
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/reports/phase1_coverage.csv)")
    args = parser.parse_args(argv)

    seasons = parse_season_range(args.seasons)
    rows = generate_coverage_rows(seasons)
    out_path = Path(args.out) if args.out else default_report_path()
    write_coverage_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
