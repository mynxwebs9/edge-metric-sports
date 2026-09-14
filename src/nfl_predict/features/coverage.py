"""Phase 2 feature coverage report.

Produces data/reports/phase2_feature_coverage.csv: one row per registered feature, with
real per-feature statistics computed from the actual built team-game feature table - not
guessed or copied from the registry's `earliest_reliable_season` alone. Correctness of
inputs only - no betting/outcome statistics of any kind belong here or anywhere in Phase 2.

Run as: python -m nfl_predict.features.coverage --seasons 2010-2025
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings
from nfl_predict.data.season_range import DEFAULT_HISTORICAL_SEASONS, parse_season_range
from nfl_predict.features.registry import load_feature_registry
from nfl_predict.features.store import read_feature_table

REPORT_COLUMNS = [
    "feature_name", "family", "earliest_usable_season_registered", "earliest_season_actually_non_null",
    "non_null_count", "non_null_pct", "min", "max", "mean", "std", "rolling_window", "sample_count_behavior",
]


def _sample_count_behavior(entry: dict) -> str:
    name = entry["name"]
    if name.endswith(("_3g", "_5g", "_8g")):
        w = name.rsplit("_", 1)[-1].rstrip("g")
        return f"see n_games_trailing_{w}"
    if entry["category"] == "quarterback" and name != "qb_primary_id":
        return "see qb_starts_season"
    if name in ("off_redzone_td_rate_season", "def_redzone_td_rate_allowed_season"):
        return "see redzone_trips_n / redzone_trips_n_allowed"
    if name == "fg_pct_season":
        return "see fg_attempt_n"
    if name == "punt_yards_avg_season":
        return "see punt_n"
    if entry["rolling_window"].startswith("season-to-date"):
        return "see games_played_current_season"
    return "n/a (not a rolling/sample-dependent value)"


def generate_coverage_rows(feature_version: str, seasons: list[int]) -> list[dict]:
    tg = read_feature_table("team_game", feature_version, seasons)
    registry = load_feature_registry()
    rows: list[dict] = []

    for entry in registry:
        name = entry["name"]
        if name not in tg.columns:
            continue
        col = tg[name]
        non_null = tg.filter(pl.col(name).is_not_null())
        non_null_count = non_null.height
        non_null_pct = round(100.0 * non_null_count / tg.height, 2) if tg.height else None

        earliest_actual = None
        if non_null_count:
            earliest_actual = non_null["season"].min()

        is_numeric = col.dtype in (pl.Float64, pl.Int64, pl.Int32, pl.Float32)
        is_boolean = col.dtype == pl.Boolean
        stats = {"min": None, "max": None, "mean": None, "std": None}
        if is_numeric and non_null_count:
            stats["min"] = non_null[name].min()
            stats["max"] = non_null[name].max()
            stats["mean"] = non_null[name].mean()
            stats["std"] = non_null[name].std()
        elif is_boolean and non_null_count:
            true_share = non_null[name].cast(pl.Int64).mean()
            stats["mean"] = true_share  # share of True

        rows.append({
            "feature_name": name,
            "family": entry["category"],
            "earliest_usable_season_registered": entry["earliest_reliable_season"],
            "earliest_season_actually_non_null": earliest_actual,
            "non_null_count": non_null_count,
            "non_null_pct": non_null_pct,
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "std": stats["std"],
            "rolling_window": entry["rolling_window"],
            "sample_count_behavior": _sample_count_behavior(entry),
        })

    return rows


def write_coverage_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_report_path() -> Path:
    return get_settings().data_dir / "reports" / "phase2_feature_coverage.csv"


def main(argv: list[str] | None = None) -> int:
    from nfl_predict.config import get_feature_engine_config

    parser = argparse.ArgumentParser(prog="python -m nfl_predict.features.coverage")
    parser.add_argument("--seasons", default=DEFAULT_HISTORICAL_SEASONS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seasons = parse_season_range(args.seasons)
    feature_version = get_feature_engine_config()["feature_version"]
    rows = generate_coverage_rows(feature_version, seasons)
    out_path = Path(args.out) if args.out else default_report_path()
    write_coverage_csv(rows, out_path)
    print(f"Wrote {len(rows)} feature rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
