"""Orchestrates fetching + raw snapshotting for one dataset across one or more seasons.

This is the layer the CLI (src/nfl_predict/data/ingest.py) and the real-ingestion scripts
call. It never fails an entire run because one season of one optional dataset is
unavailable — see docs/PHASE1_DATA_REPORT.md and the "Do not fail the entire Phase 1
implementation" instruction this satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nfl_predict.data.nflverse_loader import (
    ALL_DATASET_SPECS,
    SOURCE_NAME,
    DatasetSpec,
    SeasonNotAvailableError,
    fetch_dataset_season,
    loader_package_version,
)
from nfl_predict.data.provenance import ProvenanceManifest
from nfl_predict.data.raw_store import write_raw_snapshot
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)

FetchStatus = Literal["fetched", "unavailable", "error"]


@dataclass(frozen=True)
class FetchOutcome:
    dataset_name: str
    season: int | None
    status: FetchStatus
    manifest: ProvenanceManifest | None
    row_count: int | None
    error_message: str | None
    dataframe: object | None = None  # pl.DataFrame, kept only for immediate in-process use
    # (e.g. by the CLI's validate/promote step) - never persisted as part of the outcome.


def ingest_dataset(dataset_key: str, seasons: list[int]) -> list[FetchOutcome]:
    """Fetch and snapshot `dataset_key` for each season in `seasons`.

    For a season-less dataset (currently only "teams"), `seasons` is still accepted (for a
    uniform CLI) but the fetch happens exactly once, tagged with the full requested list in
    its manifest for documentation purposes.
    """
    if dataset_key not in ALL_DATASET_SPECS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Known datasets: {sorted(ALL_DATASET_SPECS)}"
        )
    spec = ALL_DATASET_SPECS[dataset_key]

    if not spec.takes_season:
        return [_ingest_one(spec, season=None, requested_seasons=seasons)]

    return [_ingest_one(spec, season=season, requested_seasons=[season]) for season in seasons]


def _ingest_one(spec: DatasetSpec, season: int | None, requested_seasons: list[int]) -> FetchOutcome:
    log_ctx = {"dataset_name": spec.name, "season": season}
    try:
        df = fetch_dataset_season(spec, season)
    except SeasonNotAvailableError as exc:
        logger.warning("season not available for dataset", extra={**log_ctx, "reason": str(exc)})
        return FetchOutcome(spec.name, season, "unavailable", None, None, str(exc))
    except Exception as exc:  # noqa: BLE001 - real fetch errors are recorded, not swallowed
        logger.error("dataset fetch failed", extra={**log_ctx, "error": str(exc)})
        return FetchOutcome(spec.name, season, "error", None, None, str(exc))

    if df.height == 0:
        logger.warning("dataset fetch succeeded but returned zero rows", extra=log_ctx)

    manifest = write_raw_snapshot(
        df,
        source_name=SOURCE_NAME,
        dataset_name=spec.name,
        requested_seasons=requested_seasons,
        loader_function=spec.loader_name,
        loader_package_version=loader_package_version(),
        source_identifier=(
            f"nflverse-data GitHub release, via nflreadpy.{spec.loader_name}()"
        ),
        source_release_identifier=None,
        season_for_retrieval_id=season,
    )
    return FetchOutcome(spec.name, season, "fetched", manifest, df.height, None, dataframe=df)
