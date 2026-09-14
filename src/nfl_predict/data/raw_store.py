"""Immutable raw snapshot storage.

Layout (see docs/ARCHITECTURE.md#raw-data-provenance and config/sources.yaml's
raw_storage_convention for nflverse):

    data/raw/<source_name>/<dataset_name>/<retrieval_id>/data.parquet
    data/raw/<source_name>/<dataset_name>/<retrieval_id>/manifest.json

A snapshot directory, once written, is never modified or overwritten. A re-fetch always gets
a new retrieval_id; if its content hash matches an earlier snapshot's, the new manifest
records that (duplicate_of_retrieval_id) and reuses the existing file instead of writing a
byte-for-byte copy.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from nfl_predict.config import get_settings
from nfl_predict.data.provenance import (
    ProvenanceManifest,
    compute_schema_fingerprint,
    compute_sha256,
    read_manifest,
    write_manifest,
)
from nfl_predict.logging_conf import get_logger

logger = get_logger(__name__)

RAW_DIRNAME = "raw"
DATA_FILENAME = "data.parquet"
MANIFEST_FILENAME = "manifest.json"


def raw_root() -> Path:
    return get_settings().data_dir / RAW_DIRNAME


def dataset_dir(source_name: str, dataset_name: str) -> Path:
    return raw_root() / source_name / dataset_name


def new_retrieval_id(season: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{season}_{stamp}" if season is not None else stamp


def list_manifests(source_name: str, dataset_name: str) -> list[ProvenanceManifest]:
    """All manifests currently on disk for a dataset, oldest first by retrieval_id string
    sort (retrieval_id is timestamp-prefixed/suffixed so this is chronological)."""
    ddir = dataset_dir(source_name, dataset_name)
    if not ddir.is_dir():
        return []
    manifests = []
    for snap_dir in sorted(ddir.iterdir()):
        manifest_path = snap_dir / MANIFEST_FILENAME
        if manifest_path.is_file():
            manifests.append(read_manifest(manifest_path))
    return manifests


def find_duplicate_by_hash(
    source_name: str, dataset_name: str, content_sha256: str
) -> ProvenanceManifest | None:
    """Most recent existing manifest (for this dataset) whose content hash matches, if any.
    Only considers non-duplicate manifests as the canonical file holder, so chains of
    duplicates all point back to the one snapshot that actually holds the bytes."""
    for manifest in reversed(list_manifests(source_name, dataset_name)):
        if manifest.duplicate_of_retrieval_id is None and manifest.content_sha256 == content_sha256:
            return manifest
    return None


def write_raw_snapshot(
    df: pl.DataFrame,
    *,
    source_name: str,
    dataset_name: str,
    requested_seasons: list[int],
    loader_function: str,
    loader_package_version: str,
    source_identifier: str,
    source_release_identifier: str | None,
    season_for_retrieval_id: int | None = None,
) -> ProvenanceManifest:
    """Write a Polars DataFrame as an immutable raw Parquet snapshot with its manifest.

    If the serialized content is byte-identical to an existing snapshot for this dataset,
    the new manifest is recorded (so the fetch attempt itself is never lost) but points at
    the existing file rather than writing a duplicate copy.
    """
    retrieval_id = new_retrieval_id(season_for_retrieval_id)
    snap_dir = dataset_dir(source_name, dataset_name) / retrieval_id
    snap_dir.mkdir(parents=True, exist_ok=False)
    data_path = snap_dir / DATA_FILENAME

    df.write_parquet(data_path)
    content_hash = compute_sha256(data_path)

    column_names = list(df.columns)
    column_dtypes = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}
    schema_fingerprint = compute_schema_fingerprint(column_names, column_dtypes)

    duplicate = find_duplicate_by_hash(source_name, dataset_name, content_hash)
    raw_file_path = data_path
    duplicate_of_retrieval_id = None
    if duplicate is not None:
        duplicate_of_retrieval_id = duplicate.retrieval_id
        raw_file_path = Path(duplicate.raw_file_path)
        data_path.unlink()  # don't keep a byte-for-byte duplicate on disk
        logger.info(
            "raw snapshot content identical to an earlier fetch; reusing existing file",
            extra={
                "source_name": source_name,
                "dataset_name": dataset_name,
                "retrieval_id": retrieval_id,
                "duplicate_of_retrieval_id": duplicate_of_retrieval_id,
            },
        )

    manifest = ProvenanceManifest(
        source_name=source_name,
        dataset_name=dataset_name,
        requested_seasons=requested_seasons,
        loader_function=loader_function,
        loader_package_version=loader_package_version,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        source_identifier=source_identifier,
        source_release_identifier=source_release_identifier,
        raw_file_path=str(raw_file_path),
        content_sha256=content_hash,
        row_count=df.height,
        column_count=df.width,
        column_names=column_names,
        column_dtypes=column_dtypes,
        schema_fingerprint=schema_fingerprint,
        retrieval_id=retrieval_id,
        duplicate_of_retrieval_id=duplicate_of_retrieval_id,
    )
    write_manifest(manifest, snap_dir / MANIFEST_FILENAME)
    logger.info(
        "wrote raw snapshot",
        extra={
            "source_name": source_name,
            "dataset_name": dataset_name,
            "retrieval_id": retrieval_id,
            "row_count": df.height,
            "column_count": df.width,
            "duplicate": duplicate_of_retrieval_id is not None,
        },
    )
    return manifest


def read_raw_snapshot(manifest: ProvenanceManifest) -> pl.DataFrame:
    # memory_map=False: see pbp_store.read_pbp_season for why - raw snapshots are never
    # rewritten to the same path in production (immutability guarantees a fresh
    # retrieval_id/path per fetch), but reading eagerly here keeps every Parquet read in
    # this codebase consistent in how it releases its file handle, rather than leaving one
    # module memory-mapped and another not for no principled reason.
    return pl.read_parquet(manifest.raw_file_path, memory_map=False)


def purge_empty_snapshot_dir(source_name: str, dataset_name: str, retrieval_id: str) -> None:
    """Remove a snapshot directory that was started but never completed (e.g. the loader
    raised before any file was written). Never called on a directory containing a written
    manifest — that would violate immutability."""
    snap_dir = dataset_dir(source_name, dataset_name) / retrieval_id
    if snap_dir.is_dir() and not (snap_dir / MANIFEST_FILENAME).exists():
        shutil.rmtree(snap_dir)
